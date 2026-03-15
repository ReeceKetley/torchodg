from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class _EarChannelState:
    power_spectrum: torch.Tensor
    weighted_power_spectrum: torch.Tensor
    unsmeared_excitation: torch.Tensor
    excitation: torch.Tensor
    energy_threshold_reached: torch.Tensor
    frames: torch.Tensor


class BS1387BasicFrontEnd(torch.nn.Module):
    """
    Differentiable BS.1387-inspired basic front-end.

    This ports the GstPEAQ basic FFT ear-model signal flow and MOV equations
    closely enough for training use, while keeping all math in torch.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        playback_level_db: float = 92.0,
        n_fft: int = 2048,
        hop_length: int = 1024,
        band_count: int = 109,
        ehs_mode: str = "fast",
        max_frames: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if sample_rate != 48000:
            raise ValueError("BS1387BasicFrontEnd expects internal sample_rate=48000")
        if n_fft != 2048 or hop_length != 1024:
            raise ValueError("BS1387BasicFrontEnd currently supports n_fft=2048, hop_length=1024")
        if band_count != 109:
            raise ValueError("BS1387BasicFrontEnd currently supports band_count=109 (basic mode)")

        self.sample_rate = sample_rate
        self.playback_level_db = float(playback_level_db)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.band_count = band_count
        if ehs_mode not in ("fast", "exact"):
            raise ValueError("ehs_mode must be 'fast' or 'exact'")
        self.ehs_mode = ehs_mode
        self.max_frames = max_frames
        self.eps = eps
        self.maxlag = 256

        gamma = 0.84971762641205
        level_factor = (10.0 ** (self.playback_level_db / 10.0)) / (
            (8.0 / 3.0) * ((gamma / 4.0) * (self.n_fft - 1)) ** 2
        )

        k = torch.arange(self.n_fft, dtype=torch.float32)
        hann = math.sqrt(8.0 / 3.0) * 0.5 * (1.0 - torch.cos(2.0 * math.pi * k / (self.n_fft - 1)))
        self.register_buffer("window", hann, persistent=False)
        self.register_buffer("level_factor", torch.tensor(level_factor, dtype=torch.float32), persistent=False)

        f = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32) * (self.sample_rate / self.n_fft)
        ear_weight = self._ear_weight(f)
        self.register_buffer("outer_middle_ear_weight", ear_weight.square(), persistent=False)

        fc, delta_z = self._band_centers(self.band_count)
        self.register_buffer("fc", fc, persistent=False)
        self.register_buffer("delta_z", torch.tensor(delta_z, dtype=torch.float32), persistent=False)

        band_matrix = self._build_band_matrix(fc, delta_z)
        self.register_buffer("band_matrix", band_matrix, persistent=False)

        internal_noise = torch.pow(10.0, 0.4 * 0.364 * torch.pow(fc / 1000.0, -0.8))
        excitation_threshold = torch.pow(10.0, 0.364 * torch.pow(fc / 1000.0, -0.8))
        threshold_index = torch.pow(
            10.0,
            0.1
            * (
                -2.0
                - 2.05 * torch.atan(fc / 4000.0)
                - 0.75 * torch.atan(torch.pow(fc / 1600.0, 2.0))
            ),
        )
        loudness_scale = 1.07664
        loudness_factor = loudness_scale * torch.pow(
            excitation_threshold / (1e4 * threshold_index + self.eps), 0.23
        )

        self.register_buffer("internal_noise", internal_noise, persistent=False)
        self.register_buffer("excitation_threshold", excitation_threshold, persistent=False)
        self.register_buffer("threshold_index", threshold_index, persistent=False)
        self.register_buffer("loudness_factor", loudness_factor, persistent=False)

        tau_ear = 0.008 + 100.0 / fc * (0.030 - 0.008)
        a_ear = torch.exp(torch.tensor(self.hop_length / -48000.0, dtype=torch.float32) / tau_ear)
        self.register_buffer("a_ear", a_ear, persistent=False)

        tau_mod = 0.008 + 100.0 / fc * (0.050 - 0.008)
        a_mod = torch.exp(torch.tensor(self.hop_length / -48000.0, dtype=torch.float32) / tau_mod)
        self.register_buffer("a_mod", a_mod, persistent=False)

        masking_difference = torch.pow(
            10.0, torch.where(torch.arange(self.band_count) * delta_z <= 12.0, 3.0, 0.25 * torch.arange(self.band_count) * delta_z) / 10.0
        )
        self.register_buffer("masking_difference", masking_difference.to(torch.float32), persistent=False)

        lower_spreading = torch.tensor(10.0 ** (-2.7 * delta_z), dtype=torch.float32)
        self.register_buffer("lower_spreading", lower_spreading, persistent=False)
        self.register_buffer("lower_spreading_exp", torch.pow(lower_spreading, 0.4), persistent=False)
        a_uc = torch.pow(10.0, (-2.4 - 23.0 / fc) * delta_z)
        self.register_buffer("a_uc", a_uc, persistent=False)
        g_il = (1.0 - torch.pow(lower_spreading, torch.arange(self.band_count, dtype=torch.float32) + 1.0)) / (
            1.0 - lower_spreading
        )
        self.register_buffer("g_il", g_il, persistent=False)

        # Helpers for vectorized frequency spreading.
        idx = torch.arange(self.band_count, dtype=torch.float32)
        i_idx = idx.view(-1, 1)
        j_idx = idx.view(1, -1)
        upper_mask = (j_idx > i_idx).to(torch.float32)
        upper_delta = torch.clamp(j_idx - i_idx, min=0.0)
        self.register_buffer("spread_upper_mask", upper_mask, persistent=False)
        self.register_buffer("spread_upper_delta", upper_delta, persistent=False)

        # Lower-recursive component matrix: L[i, k] = aLe^(k-i) for k >= i else 0.
        a_le = torch.pow(lower_spreading, 0.4)
        lower_delta = torch.clamp(j_idx - i_idx, min=0.0)
        lower_mask = (j_idx >= i_idx).to(torch.float32)
        lower_mat = torch.pow(a_le, lower_delta) * lower_mask
        self.register_buffer("spread_lower_mat", lower_mat, persistent=False)

        spread_norm = self._frequency_spread(
            torch.ones(self.band_count, dtype=torch.float32), normalize=False
        )
        self.register_buffer("spreading_normalization", spread_norm, persistent=False)

        w = torch.arange(self.maxlag, dtype=torch.float32)
        corr_window = 0.81649658092773 * (1.0 - torch.cos(2.0 * math.pi * w / (self.maxlag - 1.0))) / self.maxlag
        self.register_buffer("corr_window", corr_window, persistent=False)

        # Pattern-adaptation neighborhood averaging matrix (eq. 50/51 in BS.1387 path).
        avg_mat = torch.zeros(self.band_count, self.band_count, dtype=torch.float32)
        bdiv36 = self.band_count // 36
        bdiv25 = self.band_count // 25
        for k_idx in range(self.band_count):
            m1 = min(k_idx, bdiv36)
            m2 = min(self.band_count - k_idx - 1, bdiv25)
            sl = slice(k_idx - m1, k_idx + m2 + 1)
            avg_mat[k_idx, sl] = 1.0 / (m1 + m2 + 1)
        self.register_buffer("pattern_avg_matrix", avg_mat, persistent=False)

    def forward(self, ref: torch.Tensor, test: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        ref = self._as_bct(ref).to(dtype=torch.float32)
        test = self._as_bct(test).to(dtype=torch.float32)
        if ref.shape != test.shape:
            raise ValueError(f"ref/test shapes must match, got {tuple(ref.shape)} vs {tuple(test.shape)}")
        if sample_rate is not None and sample_rate != self.sample_rate:
            ref = self._resample_linear(ref, sample_rate, self.sample_rate)
            test = self._resample_linear(test, sample_rate, self.sample_rate)

        batch_movs = []
        for b in range(ref.shape[0]):
            batch_movs.append(self._compute_pair_movs(ref[b], test[b]))
        return torch.stack(batch_movs, dim=0)

    def _compute_pair_movs(self, ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        channels = ref.shape[0]
        ref_states = [self._process_channel(ref[c]) for c in range(channels)]
        test_states = [self._process_channel(test[c]) for c in range(channels)]
        frames = ref_states[0].power_spectrum.shape[0]
        above = self._frame_above_threshold(ref_states)

        mod_ref, avg_loud_ref = [], []
        mod_test, avg_loud_test = [], []
        adapted_ref, adapted_test = [], []
        loud_reached = []
        for c in range(channels):
            mref, aref = self._modulation(ref_states[c].unsmeared_excitation)
            mtest, atest = self._modulation(test_states[c].unsmeared_excitation)
            mod_ref.append(mref)
            avg_loud_ref.append(aref)
            mod_test.append(mtest)
            avg_loud_test.append(atest)
            ar, at = self._level_adapt(ref_states[c].excitation, test_states[c].excitation)
            adapted_ref.append(ar)
            adapted_test.append(at)
            lref = self._loudness(ref_states[c].excitation)
            ltest = self._loudness(test_states[c].excitation)
            idx = (lref > 0.1) & (ltest > 0.1)
            loud_reached.append(int(torch.argmax(idx.float()).item()) if idx.any() else frames + 1)

        mov_bw_ref_num = ref.new_tensor(0.0)
        mov_bw_ref_den = ref.new_tensor(0.0)
        mov_bw_test_num = ref.new_tensor(0.0)
        mov_bw_test_den = ref.new_tensor(0.0)
        mov_nmr_linear_sum = ref.new_tensor(0.0)
        mov_nmr_den = ref.new_tensor(0.0)
        mov_rdf_num = ref.new_tensor(0.0)
        mov_rdf_den = ref.new_tensor(0.0)

        m1_num = ref.new_tensor(0.0)
        m1_den = ref.new_tensor(0.0)
        m2_num = ref.new_tensor(0.0)
        m2_den = ref.new_tensor(0.0)
        win_values = []
        nl_sq_sum = ref.new_tensor(0.0)
        nl_den = ref.new_tensor(0.0)
        ehs_sum = ref.new_tensor(0.0)
        ehs_den = ref.new_tensor(0.0)

        mfpd_filt = ref.new_tensor(0.0)
        mfpd_max = ref.new_tensor(0.0)
        adb_num = ref.new_tensor(0.0)
        adb_den = ref.new_tensor(0.0)

        for t in range(frames):
            gate = above[t].to(ref.dtype)
            if gate <= 0:
                continue

            # Bandwidth, NMR, RDF
            bw_ref_acc = ref.new_tensor(0.0)
            bw_test_acc = ref.new_tensor(0.0)
            bw_count = ref.new_tensor(0.0)
            nmr_acc = ref.new_tensor(0.0)
            rdf_acc = ref.new_tensor(0.0)
            for c in range(channels):
                pw_ref = ref_states[c].power_spectrum[t]
                pw_test = test_states[c].power_spectrum[t]
                zt = torch.max(pw_test[921:1024])
                idx_ref = torch.where(pw_ref[:921] > 10.0 * zt)[0]
                if idx_ref.numel() > 0:
                    bw_ref = (idx_ref[-1] + 1).to(ref.dtype)
                    if bw_ref > 346:
                        idx_test = torch.where(pw_test[: int(bw_ref.item())] >= 3.16227766016838 * zt)[0]
                        bw_test = (idx_test[-1] + 1).to(ref.dtype) if idx_test.numel() > 0 else ref.new_tensor(0.0)
                        bw_ref_acc = bw_ref_acc + bw_ref
                        bw_test_acc = bw_test_acc + bw_test
                        bw_count = bw_count + 1.0

                wref = ref_states[c].weighted_power_spectrum[t]
                wtest = test_states[c].weighted_power_spectrum[t]
                noise_spec = wref - 2.0 * torch.sqrt(torch.clamp(wref * wtest, min=self.eps)) + wtest
                noise_bands = torch.clamp(noise_spec @ self.band_matrix.T, min=1e-12)
                mask = ref_states[c].excitation[t] / (self.masking_difference + self.eps)
                curr_nmr = noise_bands / (mask + self.eps)
                nmr_acc = nmr_acc + curr_nmr.mean()
                rdf_acc = rdf_acc + (curr_nmr.max() > 1.41253754462275).to(ref.dtype)

            if bw_count > 0:
                mov_bw_ref_num = mov_bw_ref_num + bw_ref_acc / bw_count
                mov_bw_ref_den = mov_bw_ref_den + 1.0
                mov_bw_test_num = mov_bw_test_num + bw_test_acc / bw_count
                mov_bw_test_den = mov_bw_test_den + 1.0

            mov_nmr_linear_sum = mov_nmr_linear_sum + nmr_acc / channels
            mov_nmr_den = mov_nmr_den + 1.0
            mov_rdf_num = mov_rdf_num + rdf_acc / channels
            mov_rdf_den = mov_rdf_den + 1.0

            # Modulation diffs
            if t >= 24:
                for c in range(channels):
                    mr = mod_ref[c][t]
                    mt = mod_test[c][t]
                    diff = torch.abs(mt - mr)
                    md1 = 100.0 / self.band_count * torch.sum(diff / (1.0 + mr))
                    wt = torch.where(mt >= mr, torch.ones_like(mt), torch.full_like(mt, 0.1))
                    md2 = 100.0 / self.band_count * torch.sum(wt * diff / (0.01 + mr))
                    tw = torch.sum(avg_loud_ref[c][t] / (avg_loud_ref[c][t] + 100.0 * torch.pow(self.internal_noise, 0.3)))
                    m1_num = m1_num + md1 * tw
                    m1_den = m1_den + tw
                    m2_num = m2_num + md2 * tw
                    m2_den = m2_den + tw
                    win_values.append(md1)

            # Noise loudness
            if t >= 24:
                for c in range(channels):
                    if t - 3 >= loud_reached[c]:
                        nl = self._noise_loudness(
                            ref_mod=mod_ref[c][t],
                            test_mod=mod_test[c][t],
                            ref_adapted=adapted_ref[c][t],
                            test_adapted=adapted_test[c][t],
                            alpha=1.5,
                            thres_fac=0.15,
                            s0=0.5,
                            nl_min=0.0,
                        )
                        nl_sq_sum = nl_sq_sum + nl * nl
                        nl_den = nl_den + 1.0

            # Prob detect (binaural), vectorized across bands/channels.
            ref_exc_t = torch.stack([ref_states[c].excitation[t] for c in range(channels)], dim=0)
            test_exc_t = torch.stack([test_states[c].excitation[t] for c in range(channels)], dim=0)
            eref_db = 10.0 * torch.log10(ref_exc_t + self.eps)
            etest_db = 10.0 * torch.log10(test_exc_t + self.eps)
            l = 0.3 * torch.maximum(eref_db, etest_db) + 0.7 * etest_db
            l_pos = torch.clamp(l, min=1e-6)
            s = torch.where(
                l > 0,
                5.95072 * torch.pow(6.39468 / l_pos, 1.71332)
                + 9.01033e-11 * torch.pow(l_pos, 4.0)
                + 5.05622e-6 * torch.pow(l_pos, 3.0)
                - 0.00102438 * torch.pow(l_pos, 2.0)
                + 0.0550197 * l_pos
                - 0.198719,
                torch.full_like(l, 1e30),
            )
            e = eref_db - etest_db
            ratio = e / (s + self.eps)
            ratio_pow = torch.where(eref_db > etest_db, ratio**4, ratio**6)
            pc = 1.0 - torch.pow(0.5, ratio_pow)
            qc = torch.abs(torch.trunc(e)) / (s + self.eps)
            pmax = torch.amax(pc, dim=0)
            qmax = torch.amax(qc, dim=0)
            pbin = 1.0 - torch.prod(1.0 - pmax)
            qbin = torch.sum(qmax)
            if pbin > 0.5:
                adb_num = adb_num + qbin
                adb_den = adb_den + 1.0
            mfpd_filt = 0.9 * mfpd_filt + 0.1 * pbin
            mfpd_max = torch.maximum(mfpd_max, mfpd_filt)

            # EHS
            ehs_valid = any(bool(ref_states[c].energy_threshold_reached[t].item()) or bool(test_states[c].energy_threshold_reached[t].item()) for c in range(channels))
            if ehs_valid:
                for c in range(channels):
                    if self.ehs_mode == "exact":
                        ehs_curr = self._ehs_frame_exact(
                            ref_states[c].weighted_power_spectrum[t],
                            test_states[c].weighted_power_spectrum[t],
                        )
                    else:
                        ehs_curr = self._ehs_frame_fast(
                            ref_states[c].weighted_power_spectrum[t],
                            test_states[c].weighted_power_spectrum[t],
                        )
                    ehs_sum = ehs_sum + ehs_curr
                    ehs_den = ehs_den + 1.0

        bw_ref = mov_bw_ref_num / (mov_bw_ref_den + self.eps)
        bw_test = mov_bw_test_num / (mov_bw_test_den + self.eps)
        total_nmr = 10.0 * torch.log10(mov_nmr_linear_sum / (mov_nmr_den + self.eps) + self.eps)
        win_mod = self._avg_window(win_values)
        adb = self._adb_value(adb_num, adb_den)
        ehs = ehs_sum / (ehs_den + self.eps)
        avg_mod1 = m1_num / (m1_den + self.eps)
        avg_mod2 = m2_num / (m2_den + self.eps)
        rms_noise_loud = torch.sqrt(nl_sq_sum / (nl_den + self.eps) + self.eps)
        mfpd = mfpd_max
        rdf = mov_rdf_num / (mov_rdf_den + self.eps)

        return torch.stack(
            [bw_ref, bw_test, total_nmr, win_mod, adb, ehs, avg_mod1, avg_mod2, rms_noise_loud, mfpd, rdf],
            dim=0,
        )

    def _process_channel(self, x: torch.Tensor) -> _EarChannelState:
        frames = x.unfold(dimension=-1, size=self.n_fft, step=self.hop_length).contiguous()
        if self.max_frames is not None:
            frames = frames[: self.max_frames]
        win = self.window.to(x.device, x.dtype)
        xw = frames * win
        fft = torch.fft.rfft(xw, n=self.n_fft, dim=-1)
        power = (fft.real.square() + fft.imag.square()) * self.level_factor.to(x.device, x.dtype)
        weighted = power * self.outer_middle_ear_weight.to(x.device, x.dtype)
        band_power = torch.clamp(weighted @ self.band_matrix.T.to(x.device, x.dtype), min=1e-12)
        noisy_band = band_power + self.internal_noise.to(x.device, x.dtype)

        unsmeared = []
        for t in range(noisy_band.shape[0]):
            unsmeared.append(self._frequency_spread(noisy_band[t]))
        unsmeared = torch.stack(unsmeared, dim=0)

        a = self.a_ear.to(x.device, x.dtype)
        filt = torch.zeros(self.band_count, device=x.device, dtype=x.dtype)
        exc = []
        for t in range(unsmeared.shape[0]):
            filt = a * filt + (1.0 - a) * unsmeared[t]
            exc.append(torch.maximum(filt, unsmeared[t]))
        excitation = torch.stack(exc, dim=0)

        energy_tail = frames[:, self.n_fft // 2 :].square().sum(dim=-1)
        energy_reached = energy_tail >= (8000.0 / (32768.0 * 32768.0))

        return _EarChannelState(
            power_spectrum=power,
            weighted_power_spectrum=weighted,
            unsmeared_excitation=unsmeared,
            excitation=excitation,
            energy_threshold_reached=energy_reached,
            frames=frames,
        )

    def _modulation(self, unsmeared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.a_mod.to(unsmeared.device, unsmeared.dtype)
        derivative_factor = float(self.sample_rate) / float(self.hop_length)
        prev_loud = torch.zeros(self.band_count, device=unsmeared.device, dtype=unsmeared.dtype)
        filt_loud = torch.zeros_like(prev_loud)
        filt_der = torch.zeros_like(prev_loud)
        mods, avgs = [], []
        for t in range(unsmeared.shape[0]):
            loud = torch.pow(torch.clamp(unsmeared[t], min=0.0), 0.3)
            loud_der = derivative_factor * torch.abs(loud - prev_loud)
            filt_der = a * filt_der + (1.0 - a) * loud_der
            filt_loud = a * filt_loud + (1.0 - a) * loud
            mod = filt_der / (1.0 + filt_loud / 0.3)
            prev_loud = loud
            mods.append(mod)
            avgs.append(filt_loud)
        return torch.stack(mods, dim=0), torch.stack(avgs, dim=0)

    def _level_adapt(self, ref_exc: torch.Tensor, test_exc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.a_mod.to(ref_exc.device, ref_exc.dtype)
        rf = torch.zeros(self.band_count, device=ref_exc.device, dtype=ref_exc.dtype)
        tf = torch.zeros_like(rf)
        fn = torch.zeros_like(rf)
        fd = torch.zeros_like(rf)
        pr = torch.zeros_like(rf)
        pt = torch.zeros_like(rf)
        out_r, out_t = [], []
        avg_mat = self.pattern_avg_matrix.to(ref_exc.device, ref_exc.dtype)
        for t in range(ref_exc.shape[0]):
            re = ref_exc[t]
            te = test_exc[t]
            rf = a * rf + (1.0 - a) * re
            tf = a * tf + (1.0 - a) * te
            num = torch.sum(torch.sqrt(torch.clamp(rf * tf, min=self.eps)))
            den = torch.sum(tf) + self.eps
            lev_corr = (num * num) / (den * den + self.eps)
            lc_gt = lev_corr > 1.0
            r_corr = torch.where(lc_gt, re / (lev_corr + self.eps), re)
            t_corr = torch.where(lc_gt, te, te * lev_corr)
            fn = a * fn + t_corr * r_corr
            fd = a * fd + r_corr * r_corr
            pa_ref = torch.where(fn >= fd, torch.ones_like(fn), fn / (fd + self.eps))
            pa_test = torch.where(fn >= fd, fd / (fn + self.eps), torch.ones_like(fn))
            ra_ref = avg_mat @ pa_ref
            ra_test = avg_mat @ pa_test
            pr = a * pr + (1.0 - a) * ra_ref
            pt = a * pt + (1.0 - a) * ra_test
            out_r.append(r_corr * pr)
            out_t.append(t_corr * pt)
        return torch.stack(out_r, dim=0), torch.stack(out_t, dim=0)

    def _noise_loudness(
        self,
        ref_mod: torch.Tensor,
        test_mod: torch.Tensor,
        ref_adapted: torch.Tensor,
        test_adapted: torch.Tensor,
        alpha: float,
        thres_fac: float,
        s0: float,
        nl_min: float,
    ) -> torch.Tensor:
        sref = thres_fac * ref_mod + s0
        stest = thres_fac * test_mod + s0
        ep_ref = ref_adapted
        ep_test = test_adapted
        beta = torch.exp(-alpha * (ep_test - ep_ref) / (ep_ref + self.eps))
        t = torch.clamp(stest * ep_test - sref * ep_ref, min=0.0) / (
            self.internal_noise.to(ep_ref.device, ep_ref.dtype) + sref * ep_ref * beta + self.eps
        )
        nl = torch.sum(torch.pow(self.internal_noise.to(ep_ref.device, ep_ref.dtype) / (stest + self.eps), 0.23) * (torch.pow(1.0 + t, 0.23) - 1.0))
        nl = nl * (24.0 / self.band_count)
        return torch.where(nl < nl_min, torch.zeros_like(nl), nl)

    def _ehs_frame_fast(self, ref_weighted: torch.Tensor, test_weighted: torch.Tensor) -> torch.Tensor:
        # Fast differentiable proxy using FFT-based autocorrelation on log error.
        d = torch.log(torch.clamp(test_weighted, min=self.eps) / torch.clamp(ref_weighted, min=self.eps))
        d = d - d.mean()
        den = torch.sum(d * d) + self.eps
        n = d.shape[0]
        nfft = 1
        while nfft < 2 * n:
            nfft *= 2
        spec = torch.fft.rfft(d, n=nfft)
        ac = torch.fft.irfft(spec * torch.conj(spec), n=nfft)[:33]
        vals = torch.relu(ac[1:] / den)
        return 1000.0 * vals.amax()

    def _ehs_frame_exact(self, ref_weighted: torch.Tensor, test_weighted: torch.Tensor) -> torch.Tensor:
        n = 2 * self.maxlag
        rr = torch.clamp(ref_weighted[:n], min=self.eps)
        tt = torch.clamp(test_weighted[:n], min=self.eps)
        d = torch.log(tt / rr)
        c = torch.empty(self.maxlag, device=d.device, dtype=d.dtype)
        for i in range(self.maxlag):
            c[i] = torch.sum(d[: self.maxlag] * d[i : i + self.maxlag])
        d0 = c[0]
        dk = d0
        cnorm = torch.empty_like(c)
        for i in range(self.maxlag):
            cnorm[i] = c[i] / torch.sqrt(torch.clamp(d0 * dk, min=self.eps))
            dk = dk + d[i + self.maxlag] * d[i + self.maxlag] - d[i] * d[i]
        cavg = cnorm.mean()
        cwin = (cnorm - cavg) * self.corr_window.to(d.device, d.dtype)
        cfft = torch.fft.rfft(cwin, n=self.maxlag)
        magsq = cfft.real.square() + cfft.imag.square()
        ehs = torch.tensor(0.0, device=d.device, dtype=d.dtype)
        s = magsq[0]
        for i in range(1, magsq.shape[0]):
            ns = magsq[i]
            if bool((ns > s).item()) and bool((ns > ehs).item()):
                ehs = ns
            s = ns
        return 1000.0 * ehs

    def _loudness(self, excitation: torch.Tensor) -> torch.Tensor:
        term = torch.pow(
            1.0
            - self.threshold_index.to(excitation.device, excitation.dtype)
            + self.threshold_index.to(excitation.device, excitation.dtype)
            * excitation
            / (self.excitation_threshold.to(excitation.device, excitation.dtype) + self.eps),
            0.23,
        )
        vals = self.loudness_factor.to(excitation.device, excitation.dtype) * (term - 1.0)
        vals = torch.clamp(vals, min=0.0)
        return (24.0 / self.band_count) * vals.sum(dim=-1)

    def _avg_window(self, values: list[torch.Tensor]) -> torch.Tensor:
        if len(values) < 4:
            return values[0].new_tensor(0.0) if values else self.fc.new_tensor(0.0)
        num = values[0].new_tensor(0.0)
        den = values[0].new_tensor(0.0)
        sq = [torch.sqrt(torch.clamp(v, min=self.eps)) for v in values]
        for i in range(3, len(sq)):
            ws = 0.25 * (sq[i] + sq[i - 1] + sq[i - 2] + sq[i - 3])
            num = num + ws**4
            den = den + 1.0
        return torch.sqrt(num / (den + self.eps) + self.eps)

    @staticmethod
    def _adb_value(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        if den <= 0:
            return num.new_tensor(0.0)
        if num == 0:
            return num.new_tensor(-0.5)
        return torch.log10(num / den)

    def _frame_above_threshold(self, states: list[_EarChannelState]) -> torch.Tensor:
        # Vectorized equivalent of gstpeaq.c is_frame_above_threshold on reference channels.
        thr = 200.0 / 32768.0
        kernel = torch.ones(1, 1, 5, device=states[0].frames.device, dtype=states[0].frames.dtype)
        per_channel_flags = []
        for st in states:
            x = torch.abs(st.frames).unsqueeze(1)  # [F,1,N]
            if x.shape[-1] < 5:
                per_channel_flags.append(torch.zeros(x.shape[0], dtype=torch.bool, device=x.device))
                continue
            sums = F.conv1d(x, kernel)  # [F,1,N-4]
            per_channel_flags.append(torch.amax(sums, dim=-1).squeeze(1) >= thr)
        return torch.stack(per_channel_flags, dim=0).any(dim=0)

    def _frequency_spread(self, pp: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        pp = torch.clamp(pp, min=1e-12)
        a_uce = self.a_uc.to(pp.device, pp.dtype) * torch.pow(pp, 0.2 * self.delta_z.to(pp.device, pp.dtype))
        num = 1.0 - torch.pow(a_uce, torch.arange(self.band_count, 0, -1, device=pp.device, dtype=pp.dtype))
        den = 1.0 - a_uce
        g_iu = num / (den + self.eps)
        en = pp / (self.g_il.to(pp.device, pp.dtype) + g_iu - 1.0 + self.eps)
        a_ucee = torch.pow(a_uce, 0.4)
        ene = torch.pow(torch.clamp(en, min=1e-20), 0.4)

        # Lower-spreading recursion component, vectorized.
        lower = self.spread_lower_mat.to(pp.device, pp.dtype) @ ene
        # Upper-spreading component: sum_i ene[i] * a_ucee[i]^(j-i) for j>i.
        upper = (
            ene[:, None]
            * torch.pow(a_ucee[:, None], self.spread_upper_delta.to(pp.device, pp.dtype))
            * self.spread_upper_mask.to(pp.device, pp.dtype)
        ).sum(dim=0)
        e2 = lower + upper
        e2 = torch.pow(torch.clamp(e2, min=1e-20), 1.0 / 0.4)
        if not normalize:
            return e2
        return e2 / self.spreading_normalization.to(pp.device, pp.dtype)

    def _band_centers(self, band_count: int) -> tuple[torch.Tensor, float]:
        z_l = 7.0 * math.asinh(80.0 / 650.0)
        z_u = 7.0 * math.asinh(18000.0 / 650.0)
        delta_z = 27.0 / (band_count - 1)
        z = z_l + (torch.arange(band_count, dtype=torch.float32) + 0.5) * delta_z
        fc = 650.0 * torch.sinh(z / 7.0)
        return fc, float(delta_z)

    def _build_band_matrix(self, fc: torch.Tensor, delta_z: float) -> torch.Tensor:
        z_l = 7.0 * math.asinh(80.0 / 650.0)
        z_u = 7.0 * math.asinh(18000.0 / 650.0)
        mat = torch.zeros(self.band_count, self.n_fft // 2 + 1, dtype=torch.float32)
        for b in range(self.band_count):
            zl = z_l + b * delta_z
            zu = min(z_u, z_l + (b + 1) * delta_z)
            fl = 650.0 * math.sinh(zl / 7.0)
            fu = 650.0 * math.sinh(zu / 7.0)
            lo = int(round(fl / self.sample_rate * self.n_fft))
            hi = int(round(fu / self.sample_rate * self.n_fft))
            lo = max(0, min(lo, self.n_fft // 2))
            hi = max(0, min(hi, self.n_fft // 2))
            upper_freq = (2 * lo + 1) / 2 * self.sample_rate / self.n_fft
            upper_freq = min(upper_freq, fu)
            lw = (upper_freq - fl) * self.n_fft / self.sample_rate
            uw = 0.0
            if lo != hi:
                lower_freq = (2 * hi - 1) / 2 * self.sample_rate / self.n_fft
                uw = (fu - lower_freq) * self.n_fft / self.sample_rate
            mat[b, lo] += max(0.0, float(lw))
            if hi != lo:
                mat[b, hi] += max(0.0, float(uw))
                if hi - lo > 1:
                    mat[b, lo + 1 : hi] = 1.0
        return mat

    @staticmethod
    def _ear_weight(freq: torch.Tensor) -> torch.Tensor:
        f_khz = torch.clamp(freq / 1000.0, min=1e-6)
        w_db = -0.6 * 3.64 * torch.pow(f_khz, -0.8) + 6.5 * torch.exp(-0.6 * torch.pow(f_khz - 3.3, 2.0)) - 1e-3 * torch.pow(f_khz, 3.6)
        return torch.pow(10.0, w_db / 20.0)

    @staticmethod
    def _as_bct(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim == 3:
            return x
        raise ValueError(f"expected shape (B, T) or (B, C, T), got {tuple(x.shape)}")

    @staticmethod
    def _resample_linear(x: torch.Tensor, src_rate: int, dst_rate: int) -> torch.Tensor:
        if src_rate == dst_rate:
            return x
        scale = dst_rate / src_rate
        out_len = max(1, int(round(x.shape[-1] * scale)))
        b, c, t = x.shape
        y = F.interpolate(x.reshape(b * c, 1, t), size=out_len, mode="linear", align_corners=False)
        return y.reshape(b, c, out_len)
