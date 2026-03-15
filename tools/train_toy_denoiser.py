from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import torch
import torch.nn as nn

from torchodg.loss import PEAQProxyLoss


def load_wav_mono(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sr = w.getframerate()
        n = w.getnframes()
        data = w.readframes(n)
    x = torch.frombuffer(bytearray(data), dtype=torch.int16).float().reshape(-1, ch).t() / 32768.0
    x = x.mean(dim=0, keepdim=True)  # [1, T]
    return x, sr


def add_noise_at_snr(clean: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise = torch.randn_like(clean)
    clean_rms = torch.sqrt(torch.mean(clean * clean) + 1e-9)
    noise_rms = torch.sqrt(torch.mean(noise * noise) + 1e-9)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    return clean + noise * (target_noise_rms / (noise_rms + 1e-9))


class TinyDenoiser(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv1d(channels, 1, kernel_size=9, padding=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual denoising keeps optimization stable in short runs.
        return x - self.net(x)


def random_crop_pair(clean: torch.Tensor, noisy: torch.Tensor, crop: int, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    t = clean.shape[-1]
    if crop >= t:
        c = clean.expand(batch, -1)
        n = noisy.expand(batch, -1)
        return c, n
    starts = torch.randint(0, t - crop + 1, (batch,))
    clean_chunks = [clean[:, s : s + crop] for s in starts.tolist()]
    noisy_chunks = [noisy[:, s : s + crop] for s in starts.tolist()]
    return torch.cat(clean_chunks, dim=0), torch.cat(noisy_chunks, dim=0)


@torch.no_grad()
def eval_odg(loss_fn: PEAQProxyLoss, ref: torch.Tensor, test: torch.Tensor, sr: int) -> float:
    out = loss_fn(ref, test, sample_rate=sr, return_details=True)
    return float(out.odg.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny denoiser sanity run using PEAQProxyLoss.")
    parser.add_argument("--wav", type=Path, default=Path("test_input.wav"))
    parser.add_argument("--backend", choices=["proxy", "bs1387_basic"], default="proxy")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-samples", type=int, default=12288)
    parser.add_argument("--snr-db", type=float, default=14.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 keeps PyTorch default.")
    parser.add_argument(
        "--profile",
        choices=["default", "bs_basic_cpu", "proxy_fast"],
        default="default",
        help="Applies practical defaults for different backends/devices.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.profile == "bs_basic_cpu":
        if args.backend != "bs1387_basic":
            raise ValueError("--profile bs_basic_cpu requires --backend bs1387_basic")
        if args.steps == 60:
            args.steps = 80
        if args.batch_size == 4:
            args.batch_size = 2
        if args.crop_samples == 12288:
            args.crop_samples = 8192
        if args.lr == 2e-4:
            args.lr = 1e-4
    elif args.profile == "proxy_fast":
        if args.backend != "proxy":
            raise ValueError("--profile proxy_fast requires --backend proxy")
        if args.steps == 60:
            args.steps = 40
        if args.batch_size == 4:
            args.batch_size = 6
        if args.crop_samples == 12288:
            args.crop_samples = 12288
        if args.lr == 2e-4:
            args.lr = 3e-4

    device = torch.device("cpu")
    if args.device == "auto":
        # Keep this explicitly CPU-friendly for laptops without CUDA.
        device = torch.device("cpu")

    clean, sr = load_wav_mono(args.wav)
    clean = clean.to(device)
    noisy = add_noise_at_snr(clean, args.snr_db).clamp(-1.0, 1.0)

    model = TinyDenoiser(channels=args.channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = PEAQProxyLoss(
        sample_rate=48000,
        reduction="mean",
        backend=args.backend,
        bs_max_frames=64 if args.backend == "bs1387_basic" else None,
        crop_samples=args.crop_samples if args.backend == "bs1387_basic" else None,
        odg_min_for_loss=-4.0,
        odg_max_for_loss=0.5,
        waveform_l1_weight=0.05 if args.backend == "bs1387_basic" else 0.02,
        stft_l1_weight=0.004 if args.backend == "bs1387_basic" else 0.002,
    ).to(device)

    with torch.no_grad():
        baseline_odg = eval_odg(loss_fn, clean, noisy, sr)

    print(
        f"device={device.type} backend={args.backend} steps={args.steps} batch={args.batch_size} "
        f"crop={args.crop_samples} lr={args.lr:g} threads={torch.get_num_threads()}"
    )
    print(f"Baseline ODG (noisy vs clean): {baseline_odg:.4f}")

    for step in range(1, args.steps + 1):
        ref_b, noisy_b = random_crop_pair(clean, noisy, args.crop_samples, args.batch_size)
        pred_b = model(noisy_b.unsqueeze(1)).squeeze(1).clamp(-1.0, 1.0)

        out = loss_fn(ref_b, pred_b, sample_rate=sr, return_details=True)
        loss = out.loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if step == 1 or step % args.print_every == 0 or step == args.steps:
            odg_step = float(out.odg.mean().item())
            odg_loss = float(out.odg_loss.mean().item()) if out.odg_loss is not None else math.nan
            print(f"step={step:03d} loss={float(loss.item()):.4f} odg={odg_step:.4f} odg_loss={odg_loss:.4f}")

    with torch.no_grad():
        pred_full = model(noisy.unsqueeze(1)).squeeze(1).clamp(-1.0, 1.0)
        final_odg = eval_odg(loss_fn, clean, pred_full, sr)

    print(f"Final ODG (denoised vs clean): {final_odg:.4f}")
    print(f"Delta ODG: {final_odg - baseline_odg:+.4f}")


if __name__ == "__main__":
    main()
