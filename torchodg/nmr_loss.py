"""
NMRLoss — Lightweight differentiable Noise-to-Mask Ratio loss.

Extracts only the TotalNMR computation from the BS.1387 Basic ear model,
skipping modulation, EHS, ADB, MFPD, and noise loudness. ~60% cheaper than
running the full BS1387BasicFrontEnd.

NMR (dB) = 10 * log10( mean_over_frames( mean_over_bands( noise_bands / mask ) ) )

Negative NMR means watermark is below the masking threshold (inaudible).
Loss = relu(NMR - target_nmr_db)^2 — zero when NMR is below target.

Usage:
    loss_fn = NMRLoss(target_nmr_db=-25.0).to(device)
    loss = loss_fn(ref_wav, test_wav)   # ref/test: [B, C, T] or [B, T] at 48kHz
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NMRLoss(nn.Module):
    """
    Differentiable NMR hinge loss targeting a maximum NMR of ``target_nmr_db``.

    Args:
        target_nmr_db:  Maximum allowed NMR in dB. Default -25.0.
                        Loss is zero when NMR < target_nmr_db.
        sample_rate:    Must be 48000.
        n_fft:          FFT size. Must be 2048.
        hop_length:     Hop. Must be 1024.
        playback_level_db: PEAQ playback level (default 92 dB SPL).
        band_count:     Bark bands. Must be 109.
        max_frames:     Cap frames processed per chunk (speeds up training).
        eps:            Numerical floor.
    """

    def __init__(
        self,
        target_nmr_db: float = -25.0,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 1024,
        playback_level_db: float = 92.0,
        band_count: int = 109,
        max_frames: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.target_nmr_db = target_nmr_db
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.band_count = band_count
        self.max_frames = max_frames
        self.eps = eps

        # Hann window (BS.1387 normalised)
        gamma = 0.84971762641205
        level_factor = (10.0 ** (playback_level_db / 10.0)) / (
            (8.0 / 3.0) * ((gamma / 4.0) * (n_fft - 1)) ** 2
        )
        k = torch.arange(n_fft, dtype=torch.float32)
        hann = math.sqrt(8.0 / 3.0) * 0.5 * (1.0 - torch.cos(2.0 * math.pi * k / (n_fft - 1)))
        self.register_buffer("window", hann, persistent=False)
        self.register_buffer("level_factor", torch.tensor(level_factor, dtype=torch.float32), persistent=False)

        # Outer/middle ear weighting
        f = torch.arange(n_fft // 2 + 1, dtype=torch.float32) * (sample_rate / n_fft)
        f_khz = torch.clamp(f / 1000.0, min=1e-6)
        w_db = (-0.6 * 3.64 * torch.pow(f_khz, -0.8)
                + 6.5 * torch.exp(-0.6 * torch.pow(f_khz - 3.3, 2.0))
                - 1e-3 * torch.pow(f_khz, 3.6))
        ear_weight = torch.pow(10.0, w_db / 20.0)
        self.register_buffer("outer_middle_ear_weight", ear_weight.square(), persistent=False)

        # Bark band centres and spacing
        z_l = 7.0 * math.asinh(80.0 / 650.0)
        z_u = 7.0 * math.asinh(18000.0 / 650.0)
        delta_z = 27.0 / (band_count - 1)
        z = z_l + (torch.arange(band_count, dtype=torch.float32) + 0.5) * delta_z
        fc = 650.0 * torch.sinh(z / 7.0)
        self.register_buffer("fc", fc, persistent=False)
        self.register_buffer("delta_z", torch.tensor(delta_z, dtype=torch.float32), persistent=False)

        # Band matrix [109, 1025]
        band_matrix = self._build_band_matrix(fc, delta_z, n_fft, sample_rate, z_l, z_u, band_count)
        self.register_buffer("band_matrix", band_matrix, persistent=False)

        # Internal noise and masking difference
        internal_noise = torch.pow(10.0, 0.4 * 0.364 * torch.pow(fc / 1000.0, -0.8))
        self.register_buffer("internal_noise", internal_noise, persistent=False)

        masking_difference = torch.pow(
            10.0,
            torch.where(
                torch.arange(band_count) * delta_z <= 12.0,
                torch.full((band_count,), 3.0),
                0.25 * torch.arange(band_count, dtype=torch.float32) * delta_z,
            ) / 10.0,
        )
        self.register_buffer("masking_difference", masking_difference.float(), persistent=False)

        # Temporal smoothing coefficients for excitation
        tau_ear = 0.008 + 100.0 / fc * (0.030 - 0.008)
        a_ear = torch.exp(torch.tensor(hop_length / -48000.0, dtype=torch.float32) / tau_ear)
        self.register_buffer("a_ear", a_ear, persistent=False)

        # Frequency spreading buffers
        a_uc = torch.pow(10.0, (-2.4 - 23.0 / fc) * delta_z)
        self.register_buffer("a_uc", a_uc, persistent=False)

        lower_spreading = torch.tensor(10.0 ** (-2.7 * delta_z), dtype=torch.float32)
        g_il = (1.0 - torch.pow(lower_spreading, torch.arange(band_count, dtype=torch.float32) + 1.0)) / (
            1.0 - lower_spreading
        )
        self.register_buffer("g_il", g_il, persistent=False)

        a_le = torch.pow(lower_spreading, 0.4)
        idx = torch.arange(band_count, dtype=torch.float32)
        i_idx = idx.view(-1, 1)
        j_idx = idx.view(1, -1)
        lower_delta = torch.clamp(j_idx - i_idx, min=0.0)
        lower_mask = (j_idx >= i_idx).float()
        spread_lower_mat = torch.pow(a_le, lower_delta) * lower_mask
        self.register_buffer("spread_lower_mat", spread_lower_mat, persistent=False)

        upper_mask = (j_idx > i_idx).float()
        upper_delta = torch.clamp(j_idx - i_idx, min=0.0)
        self.register_buffer("spread_upper_mask", upper_mask, persistent=False)
        self.register_buffer("spread_upper_delta", upper_delta, persistent=False)

        spread_norm = self._frequency_spread(torch.ones(band_count, dtype=torch.float32), normalize=False)
        self.register_buffer("spreading_normalization", spread_norm, persistent=False)

    # ── Public interface ──────────────────────────────────────────────────────

    def forward(self, ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref:  Reference (original) audio [B, C, T] or [B, T] at 48 kHz, float32.
            test: Degraded (watermarked) audio, same shape as ref.
        Returns:
            Scalar loss — relu(NMR - target_nmr_db)^2, mean over batch.
        """
        ref  = self._as_bct(ref).float()
        test = self._as_bct(test).float()

        losses = []
        for b in range(ref.shape[0]):
            nmr_db = self._compute_nmr(ref[b], test[b])
            losses.append(F.relu(nmr_db - self.target_nmr_db).pow(2))
        return torch.stack(losses).mean()

    def nmr_db(self, ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        """Return raw NMR in dB (no hinge) — useful for logging."""
        ref  = self._as_bct(ref).float()
        test = self._as_bct(test).float()
        return self._compute_nmr(ref[0], test[0])

    # ── Core NMR computation ──────────────────────────────────────────────────

    def _compute_nmr(self, ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        """ref/test: [C, T]. Returns scalar NMR in dB."""
        C = ref.shape[0]
        ref_w_ps, ref_exc = zip(*[self._process_channel(ref[c]) for c in range(C)])
        test_w_ps = [self._process_channel(test[c])[0] for c in range(C)]

        frames = ref_w_ps[0].shape[0]
        if self.max_frames is not None:
            frames = min(frames, self.max_frames)

        nmr_sum = ref.new_tensor(0.0)
        nmr_den = ref.new_tensor(0.0)

        # Frame-level above-threshold gate (mirrors BS.1387 basic gating)
        thr = 200.0 / 32768.0
        kernel = torch.ones(1, 1, 5, device=ref.device, dtype=ref.dtype)
        ref_frames_raw = ref[0].unfold(-1, self.n_fft, self.hop_length)[:frames]
        x = ref_frames_raw.abs().unsqueeze(1)
        if x.shape[-1] >= 5:
            sums = F.conv1d(x, kernel)
            above = torch.amax(sums, dim=-1).squeeze(1) >= thr  # [T]
        else:
            above = torch.ones(frames, dtype=torch.bool, device=ref.device)

        for t in range(frames):
            if not above[t]:
                continue
            acc = ref.new_tensor(0.0)
            for c in range(C):
                wref  = ref_w_ps[c][t]
                wtest = test_w_ps[c][t]
                noise_spec = wref - 2.0 * torch.sqrt(torch.clamp(wref * wtest, min=self.eps)) + wtest
                noise_bands = torch.clamp(noise_spec @ self.band_matrix.T, min=1e-12)
                mask = ref_exc[c][t] / (self.masking_difference.to(ref.device, ref.dtype) + self.eps)
                acc = acc + (noise_bands / (mask + self.eps)).mean()
            nmr_sum = nmr_sum + acc / C
            nmr_den = nmr_den + 1.0

        return 10.0 * torch.log10(nmr_sum / (nmr_den + self.eps) + self.eps)

    def _process_channel(self, x: torch.Tensor):
        """Returns (weighted_power_spectrum [T, F], excitation [T, B])."""
        dev, dt = x.device, x.dtype
        frames = x.unfold(-1, self.n_fft, self.hop_length)
        if self.max_frames is not None:
            frames = frames[:self.max_frames]

        win = self.window.to(dev, dt)
        xw = frames * win
        fft = torch.fft.rfft(xw, n=self.n_fft, dim=-1)
        power   = (fft.real.square() + fft.imag.square()) * self.level_factor.to(dev, dt)
        weighted = power * self.outer_middle_ear_weight.to(dev, dt)

        # Bark band power + internal noise
        band_power = torch.clamp(weighted @ self.band_matrix.T.to(dev, dt), min=1e-12)
        noisy_band = band_power + self.internal_noise.to(dev, dt)

        # Frequency spreading (per frame loop — required for masking threshold)
        unsmeared = torch.stack([self._frequency_spread(noisy_band[t]) for t in range(noisy_band.shape[0])], dim=0)

        # Temporal smoothing → excitation
        a = self.a_ear.to(dev, dt)
        filt = torch.zeros(self.band_count, device=dev, dtype=dt)
        exc = []
        for t in range(unsmeared.shape[0]):
            filt = a * filt + (1.0 - a) * unsmeared[t]
            exc.append(torch.maximum(filt, unsmeared[t]))
        excitation = torch.stack(exc, dim=0)

        return weighted, excitation

    def _frequency_spread(self, pp: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        dev, dt = pp.device, pp.dtype
        pp = torch.clamp(pp, min=1e-12)
        a_uc   = self.a_uc.to(dev, dt)
        g_il   = self.g_il.to(dev, dt)
        dz     = self.delta_z.to(dev, dt)
        n      = self.band_count

        a_uce  = torch.clamp(a_uc * torch.pow(pp, 0.2 * dz), max=0.9999)
        num    = 1.0 - torch.pow(a_uce, torch.arange(n, 0, -1, device=dev, dtype=dt))
        den    = 1.0 - a_uce
        g_iu   = torch.nan_to_num(num / (den + self.eps), nan=0.0, posinf=1e6, neginf=0.0)
        en     = torch.nan_to_num(pp / (g_il + g_iu - 1.0 + self.eps), nan=0.0, posinf=0.0, neginf=0.0)
        a_ucee = torch.pow(a_uce, 0.4)
        ene    = torch.pow(torch.clamp(en, min=1e-20), 0.4)

        lower = self.spread_lower_mat.to(dev, dt) @ ene
        upper = (ene[:, None]
                 * torch.pow(a_ucee[:, None], self.spread_upper_delta.to(dev, dt))
                 * self.spread_upper_mask.to(dev, dt)).sum(dim=0)
        e2 = torch.pow(torch.clamp(lower + upper, min=1e-20), 1.0 / 0.4)
        if not normalize:
            return e2
        return e2 / self.spreading_normalization.to(dev, dt)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_band_matrix(fc, delta_z, n_fft, sample_rate, z_l, z_u, band_count):
        mat = torch.zeros(band_count, n_fft // 2 + 1, dtype=torch.float32)
        for b in range(band_count):
            zl = z_l + b * delta_z
            zu = min(z_u, z_l + (b + 1) * delta_z)
            fl = 650.0 * math.sinh(zl / 7.0)
            fu = 650.0 * math.sinh(zu / 7.0)
            lo = int(round(fl / sample_rate * n_fft))
            hi = int(round(fu / sample_rate * n_fft))
            lo = max(0, min(lo, n_fft // 2))
            hi = max(0, min(hi, n_fft // 2))
            upper_freq = (2 * lo + 1) / 2 * sample_rate / n_fft
            upper_freq = min(upper_freq, fu)
            lw = (upper_freq - fl) * n_fft / sample_rate
            uw = 0.0
            if lo != hi:
                lower_freq = (2 * hi - 1) / 2 * sample_rate / n_fft
                uw = (fu - lower_freq) * n_fft / sample_rate
            mat[b, lo] += max(0.0, float(lw))
            if hi != lo:
                mat[b, hi] += max(0.0, float(uw))
                if hi - lo > 1:
                    mat[b, lo + 1:hi] = 1.0
        return mat

    @staticmethod
    def _as_bct(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(0).unsqueeze(0)
        if x.ndim == 2:
            return x.unsqueeze(0)
        if x.ndim == 3:
            return x
        raise ValueError(f"expected 1D/2D/3D tensor, got shape {tuple(x.shape)}")
