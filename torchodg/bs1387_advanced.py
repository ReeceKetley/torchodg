from __future__ import annotations

import torch
import torch.nn.functional as F


class BS1387AdvancedFrontEnd(torch.nn.Module):
    """
    Differentiable advanced-path front-end producing 5 MOVs:
    - RmsModDiff1
    - RmsNoiseLoudAsym
    - SegmentalNMR
    - EHS
    - AvgLinDist

    This is a practical advanced skeleton for training/eval iteration.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 1024,
        eps: float = 1e-8,
        max_frames: int | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.max_frames = max_frames
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, ref: torch.Tensor, test: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        ref = self._as_bct(ref).to(dtype=torch.float32)
        test = self._as_bct(test).to(dtype=torch.float32)
        if ref.shape != test.shape:
            raise ValueError(f"ref/test shapes must match, got {tuple(ref.shape)} vs {tuple(test.shape)}")
        if sample_rate is not None and sample_rate != self.sample_rate:
            ref = self._resample_linear(ref, sample_rate, self.sample_rate)
            test = self._resample_linear(test, sample_rate, self.sample_rate)

        # Use mono for primary MOV statistics.
        ref_mono = ref.mean(dim=1)
        test_mono = test.mean(dim=1)
        err_mono = test_mono - ref_mono

        ref_spec = torch.stft(
            ref_mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        test_spec = torch.stft(
            test_mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        if self.max_frames is not None:
            ref_spec = ref_spec[..., : self.max_frames]
            test_spec = test_spec[..., : self.max_frames]

        ref_pow = ref_spec.abs().pow(2) + self.eps
        test_pow = test_spec.abs().pow(2) + self.eps
        err_pow = torch.abs(test_spec - ref_spec).pow(2) + self.eps

        # Band-wise temporal modulation.
        ref_log = torch.log(ref_pow)
        test_log = torch.log(test_pow)
        ref_mod = torch.abs(torch.diff(ref_log, dim=-1))
        test_mod = torch.abs(torch.diff(test_log, dim=-1))

        # RmsModDiff1 (RMS form over frames/bands).
        mod_diff = 100.0 * torch.mean(torch.abs(test_mod - ref_mod) / (1.0 + ref_mod), dim=(1, 2))
        rms_mod_diff1 = torch.sqrt(torch.mean(mod_diff**2, dim=0, keepdim=False) + self.eps).reshape(-1)

        # SegmentalNMR in dB (framewise, then average).
        frame_ref = torch.mean(ref_pow, dim=1) + self.eps
        frame_noise = torch.mean(err_pow, dim=1) + self.eps
        frame_nmr = 10.0 * torch.log10(frame_noise / frame_ref + self.eps)
        segmental_nmr = torch.mean(frame_nmr, dim=1)

        # EHS proxy: strongest short-lag autocorrelation peak of log error spectrum.
        log_err = torch.log(err_pow)
        log_err = log_err - torch.mean(log_err, dim=-1, keepdim=True)
        band_mean = torch.mean(log_err, dim=1)  # [B, T]
        ehs = self._ehs_fast(band_mean)

        # AvgLinDist proxy: level-adapted linear spectral distance.
        # Apply framewise scalar level correction.
        corr = torch.sqrt(torch.sum(ref_pow, dim=1) / (torch.sum(test_pow, dim=1) + self.eps))  # [B,T]
        test_corr = test_pow * corr.unsqueeze(1)
        lin_dist = torch.abs(torch.log((test_corr + self.eps) / (ref_pow + self.eps)))
        avg_lin_dist = torch.mean(lin_dist, dim=(1, 2))

        # RmsNoiseLoudAsym proxy: asymmetric loudness-like penalty.
        # Penalize added components + missing components asymmetrically.
        add = torch.relu(test_pow - ref_pow)
        miss = torch.relu(ref_pow - test_pow)
        nl = torch.sqrt(torch.mean(add, dim=(1, 2)) + self.eps)
        mc = torch.sqrt(torch.mean(miss, dim=(1, 2)) + self.eps)
        rms_noise_loud_asym = nl + 0.5 * mc

        movs = torch.stack(
            [rms_mod_diff1, rms_noise_loud_asym, segmental_nmr, ehs, avg_lin_dist],
            dim=-1,
        )
        return movs

    def _ehs_fast(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        b, t = x.shape
        nfft = 1
        while nfft < 2 * t:
            nfft *= 2
        spec = torch.fft.rfft(x, n=nfft, dim=-1)
        ac = torch.fft.irfft(spec * torch.conj(spec), n=nfft, dim=-1)[:, :33]
        den = torch.clamp(ac[:, :1], min=self.eps)
        vals = torch.relu(ac[:, 1:] / den)
        return 1000.0 * torch.amax(vals, dim=-1)

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
