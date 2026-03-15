from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path
from statistics import mean

import torch

from torchodg.loss import PEAQProxyLoss


def parse_eaqual_silent(path: Path) -> tuple[float, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vals = lines[2].split()
    return float(vals[0]), float(vals[1])


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sr = w.getframerate()
        n = w.getnframes()
        data = w.readframes(n)
    x = torch.frombuffer(data, dtype=torch.int16).float().reshape(-1, ch).t() / 32768.0
    return x.unsqueeze(0), sr


def rankdata(xs: list[float]) -> list[float]:
    order = sorted((x, i) for i, x in enumerate(xs))
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k][1]] = r
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mx = mean(x)
    my = mean(y)
    vx = [a - mx for a in x]
    vy = [b - my for b in y]
    num = sum(a * b for a, b in zip(vx, vy))
    den = math.sqrt(sum(a * a for a in vx) * sum(b * b for b in vy) + 1e-12)
    return num / den


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def mae(x: list[float], y: list[float]) -> float:
    return mean(abs(a - b) for a, b in zip(x, y))


def rmse(x: list[float], y: list[float]) -> float:
    return math.sqrt(mean((a - b) ** 2 for a, b in zip(x, y)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate attack suite against EAQUAL references.")
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--eaqual-results", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=Path("results/eval_attack_suite.csv"))
    parser.add_argument("--bs-max-frames", type=int, default=256)
    parser.add_argument("--crop-samples", type=int, default=262144)
    args = parser.parse_args()

    ref, ref_sr = load_wav(args.ref)
    proxy = PEAQProxyLoss(
        sample_rate=48000,
        backend="proxy",
        reduction="none",
        clamp_movs=True,
        align_max_lag_samples=1024,
        loudness_match=True,
    )
    basic = PEAQProxyLoss(
        sample_rate=48000,
        backend="bs1387_basic",
        reduction="none",
        clamp_movs=True,
        bs_ehs_mode="fast",
        bs_max_frames=args.bs_max_frames,
        crop_samples=args.crop_samples,
        align_max_lag_samples=1024,
        loudness_match=True,
    )
    advanced = PEAQProxyLoss(
        sample_rate=48000,
        backend="bs1387_advanced",
        reduction="none",
        clamp_movs=True,
        bs_max_frames=args.bs_max_frames,
        crop_samples=args.crop_samples,
        align_max_lag_samples=1024,
        loudness_match=True,
    )

    rows: list[dict[str, float | str]] = []
    for wav_path in sorted(args.attacks.glob("*.wav")):
        ea_file = args.eaqual_results / f"eaqual_{wav_path.stem}.txt"
        if not ea_file.exists():
            continue
        ea_odg, ea_di = parse_eaqual_silent(ea_file)
        test, sr = load_wav(wav_path)
        p = proxy(ref, test, sample_rate=sr, return_details=True)
        b = basic(ref, test, sample_rate=sr, return_details=True)
        a = advanced(ref, test, sample_rate=sr, return_details=True)
        rows.append(
            {
                "file": wav_path.name,
                "eaqual_odg": ea_odg,
                "eaqual_di": ea_di,
                "proxy_odg": float(p.odg.item()),
                "proxy_di": float(p.di.item()),
                "bs_basic_odg": float(b.odg.item()),
                "bs_basic_di": float(b.di.item()),
                "bs_adv_odg": float(a.odg.item()),
                "bs_adv_di": float(a.di.item()),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ea = [float(r["eaqual_odg"]) for r in rows]
    p = [float(r["proxy_odg"]) for r in rows]
    b = [float(r["bs_basic_odg"]) for r in rows]
    a = [float(r["bs_adv_odg"]) for r in rows]

    print(f"Wrote {args.out_csv}")
    for name, vals in [("proxy", p), ("bs_basic", b), ("bs_advanced", a)]:
        print(
            f"{name:11s} MAE={mae(vals, ea):.4f} RMSE={rmse(vals, ea):.4f} "
            f"Pearson={pearson(vals, ea):.4f} Spearman={spearman(vals, ea):.4f}"
        )


if __name__ == "__main__":
    main()
