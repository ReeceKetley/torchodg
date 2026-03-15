from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .bs1387_advanced import BS1387AdvancedFrontEnd
from .bs1387_basic import BS1387BasicFrontEnd
from .nn import PEAQNetwork


@dataclass(frozen=True)
class PEAQProxyLossOutput:
    loss: torch.Tensor
    odg: torch.Tensor
    di: torch.Tensor
    movs: torch.Tensor
    odg_loss: torch.Tensor | None = None
    waveform_l1: torch.Tensor | None = None
    stft_l1: torch.Tensor | None = None


def _as_bct(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise ValueError(f"expected shape (B, T) or (B, C, T), got {tuple(x.shape)}")


def _ensure_same_shape(ref: torch.Tensor, test: torch.Tensor) -> None:
    if ref.shape != test.shape:
        raise ValueError(f"ref/test shapes must match, got {tuple(ref.shape)} vs {tuple(test.shape)}")


def _resample_linear(x: torch.Tensor, src_rate: int, dst_rate: int) -> torch.Tensor:
    if src_rate == dst_rate:
        return x
    scale = dst_rate / src_rate
    out_len = max(1, int(round(x.shape[-1] * scale)))
    b, c, t = x.shape
    y = F.interpolate(x.reshape(b * c, 1, t), size=out_len, mode="linear", align_corners=False)
    return y.reshape(b, c, out_len)


class PEAQProxyFrontEnd(torch.nn.Module):
    """
    Differentiable proxy front-end mapping waveforms to 11 basic-mode MOVs.

    This is not yet a strict BS.1387 implementation. It is designed as a stable,
    gradient-friendly approximation for training losses.
    """

    def __init__(
        self,
        target_sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        bandwidth_quantile: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bandwidth_quantile = bandwidth_quantile
        self.eps = eps
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(
        self,
        ref: torch.Tensor,
        test: torch.Tensor,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        ref = _as_bct(ref).to(dtype=torch.float32)
        test = _as_bct(test).to(dtype=torch.float32)
        _ensure_same_shape(ref, test)

        if sample_rate is not None and sample_rate != self.target_sample_rate:
            ref = _resample_linear(ref, sample_rate, self.target_sample_rate)
            test = _resample_linear(test, sample_rate, self.target_sample_rate)

        b, c, _ = ref.shape
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
        err_spec = test_spec - ref_spec

        ref_pow = ref_spec.abs().pow(2)
        test_pow = test_spec.abs().pow(2)
        err_pow = err_spec.abs().pow(2)

        ref_bw = self._estimate_bandwidth_hz(ref_pow)
        test_bw = self._estimate_bandwidth_hz(test_pow)

        ref_total = ref_mono.pow(2).mean(dim=-1) + self.eps
        err_total = err_mono.pow(2).mean(dim=-1) + self.eps
        total_nmr = 10.0 * torch.log10(err_total / ref_total)

        ref_log = torch.log1p(ref_pow)
        test_log = torch.log1p(test_pow)
        ref_mod = torch.abs(torch.diff(ref_log, dim=-1))
        test_mod = torch.abs(torch.diff(test_log, dim=-1))

        mod_delta = torch.abs(test_mod - ref_mod)
        win_mod_diff1 = 100.0 * (mod_delta / (1.0 + ref_mod)).mean(dim=(1, 2))
        avg_mod_diff1 = win_mod_diff1
        avg_mod_diff2 = 100.0 * (mod_delta / (0.01 + ref_mod)).mean(dim=(1, 2))

        frame_ref_energy = ref_pow.mean(dim=1)
        frame_err_energy = err_pow.mean(dim=1)
        frame_snr = 10.0 * torch.log10((frame_ref_energy + self.eps) / (frame_err_energy + self.eps))

        detection_prob = torch.sigmoid((5.0 - frame_snr) / 2.0)
        rdf = detection_prob.mean(dim=-1)
        mfpd = (detection_prob * torch.relu(-frame_snr)).mean(dim=-1)

        adb = torch.relu((10.0 * torch.log10((frame_err_energy + self.eps) / (frame_ref_energy + self.eps))) + 12.0)
        adb = adb.mean(dim=-1)

        err_env = err_pow.mean(dim=1)
        ehs = self._harmonic_proxy(err_env)

        rms_noise_loud = torch.sqrt(err_total)

        movs = torch.stack(
            [
                ref_bw,
                test_bw,
                total_nmr,
                win_mod_diff1,
                adb,
                ehs,
                avg_mod_diff1,
                avg_mod_diff2,
                rms_noise_loud,
                mfpd,
                rdf,
            ],
            dim=-1,
        )
        return movs.view(b, 11)

    def _estimate_bandwidth_hz(self, power_spectrogram: torch.Tensor) -> torch.Tensor:
        # power_spectrogram: [B, F, T]
        mean_spec = power_spectrogram.mean(dim=-1)
        cdf = torch.cumsum(mean_spec, dim=-1)
        total = cdf[:, -1:] + self.eps
        thresh = self.bandwidth_quantile * total
        soft_step = torch.sigmoid((cdf - thresh) / (0.01 * total + self.eps))
        idx = soft_step.argmax(dim=-1).to(mean_spec.dtype)
        hz_per_bin = self.target_sample_rate / self.n_fft
        return idx * hz_per_bin

    def _harmonic_proxy(self, env: torch.Tensor) -> torch.Tensor:
        # env: [B, T]
        x = env - env.mean(dim=-1, keepdim=True)
        denom = x.pow(2).sum(dim=-1) + self.eps
        lags = (1, 2, 3, 4, 5, 6, 7, 8)
        corrs = []
        for lag in lags:
            if x.shape[-1] <= lag:
                corrs.append(torch.zeros_like(denom))
                continue
            num = (x[:, :-lag] * x[:, lag:]).sum(dim=-1)
            corrs.append(torch.relu(num / denom))
        return torch.stack(corrs, dim=-1).amax(dim=-1)


class PEAQProxyLoss(torch.nn.Module):
    """
    Differentiable waveform-domain proxy of PEAQ basic ODG for training.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        clamp_movs: bool = True,
        reduction: str = "mean",
        backend: str = "proxy",
        bs_ehs_mode: str = "fast",
        bs_max_frames: int | None = None,
        crop_samples: int | None = None,
        align_max_lag_samples: int | None = None,
        loudness_match: bool = False,
        odg_min_for_loss: float = -4.0,
        odg_max_for_loss: float = 0.5,
        odg_margin_target: float | None = None,
        waveform_l1_weight: float = 0.0,
        stft_l1_weight: float = 0.0,
        stft_n_fft: int = 1024,
        stft_hop_length: int = 256,
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean"):
            raise ValueError("reduction must be 'none' or 'mean'")
        if backend not in ("proxy", "bs1387_basic", "bs1387_advanced"):
            raise ValueError("backend must be 'proxy', 'bs1387_basic' or 'bs1387_advanced'")
        if odg_min_for_loss >= odg_max_for_loss:
            raise ValueError("odg_min_for_loss must be < odg_max_for_loss")
        if odg_margin_target is not None and not (odg_min_for_loss <= odg_margin_target <= odg_max_for_loss):
            raise ValueError("odg_margin_target must lie in [odg_min_for_loss, odg_max_for_loss]")
        if waveform_l1_weight < 0.0 or stft_l1_weight < 0.0:
            raise ValueError("regularizer weights must be >= 0")
        self.reduction = reduction
        self.backend = backend
        self.crop_samples = crop_samples
        self.align_max_lag_samples = align_max_lag_samples
        self.loudness_match = loudness_match
        self.odg_min_for_loss = odg_min_for_loss
        self.odg_max_for_loss = odg_max_for_loss
        self.odg_margin_target = odg_margin_target
        self.waveform_l1_weight = waveform_l1_weight
        self.stft_l1_weight = stft_l1_weight
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.register_buffer("stft_window", torch.hann_window(stft_n_fft), persistent=False)
        if backend == "proxy":
            self.frontend = PEAQProxyFrontEnd(target_sample_rate=sample_rate)
            nn_mode = "basic"
        else:
            if backend == "bs1387_basic":
                self.frontend = BS1387BasicFrontEnd(
                    sample_rate=sample_rate,
                    ehs_mode=bs_ehs_mode,
                    max_frames=bs_max_frames,
                )
                nn_mode = "basic"
            else:
                self.frontend = BS1387AdvancedFrontEnd(
                    sample_rate=sample_rate,
                    max_frames=bs_max_frames,
                )
                nn_mode = "advanced"
        self.network = PEAQNetwork(mode=nn_mode, clamp_movs=clamp_movs, dtype=torch.float32)

    def forward(
        self,
        ref: torch.Tensor,
        test: torch.Tensor,
        sample_rate: int | None = None,
        return_details: bool = False,
    ):
        ref, test = self._maybe_crop(ref, test)
        ref, test = self._preprocess_pair(ref, test)
        movs = self.frontend(ref, test, sample_rate=sample_rate)
        di, odg = self.network(movs, return_di=True)

        odg_clamped = odg.clamp(min=self.odg_min_for_loss, max=self.odg_max_for_loss)
        if self.odg_margin_target is None:
            odg_loss = -odg_clamped
        else:
            odg_loss = torch.relu(self.odg_margin_target - odg_clamped)

        waveform_l1 = self._batch_waveform_l1(ref, test) if self.waveform_l1_weight > 0.0 else None
        stft_l1 = self._batch_stft_mag_l1(ref, test) if self.stft_l1_weight > 0.0 else None

        loss = odg_loss
        if waveform_l1 is not None:
            loss = loss + self.waveform_l1_weight * waveform_l1
        if stft_l1 is not None:
            loss = loss + self.stft_l1_weight * stft_l1

        if self.reduction == "mean":
            loss = loss.mean()
        if return_details:
            return PEAQProxyLossOutput(
                loss=loss,
                odg=odg,
                di=di,
                movs=movs,
                odg_loss=odg_loss,
                waveform_l1=waveform_l1,
                stft_l1=stft_l1,
            )
        return loss

    def _maybe_crop(self, ref: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.crop_samples is None:
            return ref, test
        if ref.shape[-1] != test.shape[-1]:
            raise ValueError("ref/test length mismatch")
        t = ref.shape[-1]
        if t <= self.crop_samples:
            return ref, test
        start = (t - self.crop_samples) // 2
        end = start + self.crop_samples
        return ref[..., start:end], test[..., start:end]

    def _preprocess_pair(self, ref: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ref = _as_bct(ref)
        test = _as_bct(test)
        if self.align_max_lag_samples:
            ref, test = self._align_by_xcorr(ref, test, self.align_max_lag_samples)
        if self.loudness_match:
            test = self._match_loudness(ref, test)
        return ref, test

    @staticmethod
    def _match_loudness(ref: torch.Tensor, test: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        # Per-item RMS match of test to reference.
        ref_rms = torch.sqrt(torch.mean(ref * ref, dim=(-2, -1), keepdim=True) + eps)
        test_rms = torch.sqrt(torch.mean(test * test, dim=(-2, -1), keepdim=True) + eps)
        gain = ref_rms / (test_rms + eps)
        return test * gain

    @staticmethod
    def _align_by_xcorr(
        ref: torch.Tensor, test: torch.Tensor, max_lag: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Integer lag search using mono cross-correlation; alignment is for eval realism.
        if max_lag <= 0:
            return ref, test
        b, c, t = ref.shape
        mono_ref = ref.mean(dim=1)
        mono_test = test.mean(dim=1)
        lags = range(-max_lag, max_lag + 1)
        scores = []
        for lag in lags:
            if lag > 0:
                r = mono_ref[:, lag:]
                s = mono_test[:, : t - lag]
            elif lag < 0:
                k = -lag
                r = mono_ref[:, : t - k]
                s = mono_test[:, k:]
            else:
                r = mono_ref
                s = mono_test
            scores.append(torch.sum(r * s, dim=-1))
        score_mat = torch.stack(scores, dim=-1)  # [B, L]
        best_idx = torch.argmax(score_mat, dim=-1)
        best_lags = torch.tensor(list(lags), device=ref.device, dtype=torch.long)[best_idx]

        out_ref = []
        out_test = []
        for i in range(b):
            lag = int(best_lags[i].item())
            if lag > 0:
                out_ref.append(ref[i, :, lag:])
                out_test.append(test[i, :, : t - lag])
            elif lag < 0:
                k = -lag
                out_ref.append(ref[i, :, : t - k])
                out_test.append(test[i, :, k:])
            else:
                out_ref.append(ref[i])
                out_test.append(test[i])
        min_len = min(x.shape[-1] for x in out_ref)
        out_ref = torch.stack([x[..., :min_len] for x in out_ref], dim=0)
        out_test = torch.stack([x[..., :min_len] for x in out_test], dim=0)
        return out_ref, out_test

    @staticmethod
    def _batch_waveform_l1(ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(test - ref), dim=(-2, -1))

    def _batch_stft_mag_l1(self, ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        b, c, t = ref.shape
        ref_flat = ref.reshape(b * c, t)
        test_flat = test.reshape(b * c, t)
        ref_spec = torch.stft(
            ref_flat,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            window=self.stft_window,
            return_complex=True,
        )
        test_spec = torch.stft(
            test_flat,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            window=self.stft_window,
            return_complex=True,
        )
        mag_l1 = torch.mean(torch.abs(test_spec.abs() - ref_spec.abs()), dim=(1, 2))
        return mag_l1.reshape(b, c).mean(dim=1)
