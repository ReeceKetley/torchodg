# torchodg

`torchodg` is a PyTorch-native approximation of the PEAQ / BS.1387 scoring
pipeline, focused on producing differentiable training losses and reusable
building blocks for audio ML experiments.

It's based on the GstPEAQ repo and public available documents on the BS.1387 standard.

Today it includes:

- PyTorch implementations of the PEAQ neural-network stage for basic and
  advanced MOV sets
- Differentiable waveform-domain proxy losses for model training
- Early BS.1387-inspired front ends intended to move toward a more complete
  approximation over time

This project is a work in progress. It is not a standards-compliant or fully
validated PEAQ implementation yet, and some parts still need optimization
(including Python loops in performance-sensitive sections).

## Status

- Intended use: research, experimentation, and training-loss prototyping
- Current focus: PyTorch-native ODG/DI approximation rather than bit-exact
  conformance
- Validation: compared against EAQUAL in a limited set of cases, with known
  discrepancies still to be resolved
- Performance: some performance-sensitive paths still rely on Python loops and
  need further optimization

## Early comparison vs EAQUAL

The repository includes a small comparison set in
`results/eval_attack_suite.csv` covering 9 processed examples. On that set,
the current BS.1387 basic front end is the closest match of the included
backends, but none of them are fully correct yet.

| Backend | ODG MAE vs EAQUAL | DI MAE vs EAQUAL |
| --- | ---: | ---: |
| `proxy` | 1.152 | 1.675 |
| `bs1387_basic` | 0.577 | 1.004 |
| `bs1387_advanced` | 0.914 | 1.616 |

Those numbers are meant as a progress snapshot, not a claim of conformance.

## Install

```bash
pip install -e .
```

## CLI usage

Basic mode (11 MOVs):

```bash
torchodg --mode basic --movs "600,590,0,20,0.5,2,20,100,1.2,0.1,0.4"
```

Advanced mode (5 MOVs):

```bash
torchodg --mode advanced --movs "50,1,-2,0.5,1"
```

JSON input:

```bash
torchodg --mode basic --movs-json path/to/movs.json
```

`--movs-json` accepts either:

- A list in the exact GstPEAQ order
- An object keyed by MOV names

## MOV order

Basic MOV order:

1. `BandwidthRef`
2. `BandwidthTest`
3. `TotalNMR`
4. `WinModDiff1`
5. `ADB`
6. `EHS`
7. `AvgModDiff1`
8. `AvgModDiff2`
9. `RmsNoiseLoud`
10. `MFPD`
11. `RelDistFrames`

Advanced MOV order:

1. `RmsModDiff1`
2. `RmsNoiseLoudAsym`
3. `SegmentalNMR`
4. `EHS`
5. `AvgLinDist`

## Differentiable training loss

```python
import torch
from torchodg import PEAQProxyLoss

loss_fn = PEAQProxyLoss(
    sample_rate=48000,
    clamp_movs=True,
    reduction="mean",
    backend="proxy",           # or "bs1387_basic" / "bs1387_advanced"
    bs_max_frames=256,         # optional speed cap for bs1387_basic
    crop_samples=16384,        # optional center crop for faster training
    align_max_lag_samples=1024,# optional sync compensation
    loudness_match=True,       # optional RMS matching before scoring
    odg_min_for_loss=-4.0,     # clamp ODG range used by the loss
    odg_max_for_loss=0.5,
    odg_margin_target=None,    # optional hinge target instead of plain -ODG
    waveform_l1_weight=0.02,   # optional waveform regularizer
    stft_l1_weight=0.002,      # optional spectral regularizer
)

ref = torch.randn(4, 48000)              # [batch, time]
test = ref + 0.01 * torch.randn_like(ref)
test.requires_grad_(True)

loss = loss_fn(ref, test)
loss.backward()                          # gradients flow to test waveform
```

If your input audio is not 48 kHz, pass `sample_rate=...` in `forward`; the
module applies differentiable linear resampling to 48 kHz internally.

Toy sanity run:

```bash
python tools/train_toy_denoiser.py --wav test_input.wav --backend proxy --steps 40
```

CPU-focused run:

```bash
python tools/train_toy_denoiser.py --wav test_input.wav --backend bs1387_basic --profile bs_basic_cpu --device cpu --threads 8 --print-every 5
```

## Repository notes

- `results/` contains comparison and evaluation outputs used during development
- `tools/` contains scripts for quick experiments and sanity checks
- `tests/` contains the current regression coverage
