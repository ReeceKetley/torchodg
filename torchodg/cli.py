from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .nn import ADVANCED_MOV_NAMES, BASIC_MOV_NAMES, PEAQNetwork


def _parse_csv(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _load_json(path: Path, mov_names: tuple[str, ...]) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [float(v) for v in data]
    if isinstance(data, dict):
        missing = [name for name in mov_names if name not in data]
        if missing:
            raise ValueError(f"missing MOV keys in JSON: {missing}")
        return [float(data[name]) for name in mov_names]
    raise ValueError("JSON input must be a list or object")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="torchodg",
        description="Compute DI/ODG from MOV vectors using PyTorch GstPEAQ constants.",
    )
    parser.add_argument(
        "--mode",
        choices=("basic", "advanced"),
        default="basic",
        help="PEAQ mode controlling required MOV vector.",
    )
    parser.add_argument(
        "--movs",
        help="Comma-separated MOV values in GstPEAQ order for selected mode.",
    )
    parser.add_argument(
        "--movs-json",
        type=Path,
        help="JSON file containing MOV list or keyed object.",
    )
    parser.add_argument(
        "--clamp-movs",
        action="store_true",
        help="Clamp normalized MOVs to [0, 1] before DI computation.",
    )
    args = parser.parse_args()

    if bool(args.movs) == bool(args.movs_json):
        parser.error("provide exactly one of --movs or --movs-json")

    mov_names = BASIC_MOV_NAMES if args.mode == "basic" else ADVANCED_MOV_NAMES
    values = _parse_csv(args.movs) if args.movs else _load_json(args.movs_json, mov_names)
    if len(values) != len(mov_names):
        parser.error(f"--mode {args.mode} expects {len(mov_names)} MOV values, got {len(values)}")

    model = PEAQNetwork(mode=args.mode, clamp_movs=args.clamp_movs)
    mov_tensor = torch.tensor(values, dtype=torch.float64).unsqueeze(0)
    di, odg = model(mov_tensor, return_di=True)
    print(f"Mode: {args.mode}")
    print(f"DI: {di.item():.6f}")
    print(f"ODG: {odg.item():.6f}")


if __name__ == "__main__":
    main()
