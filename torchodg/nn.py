from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


BASIC_MOV_NAMES = (
    "BandwidthRef",
    "BandwidthTest",
    "TotalNMR",
    "WinModDiff1",
    "ADB",
    "EHS",
    "AvgModDiff1",
    "AvgModDiff2",
    "RmsNoiseLoud",
    "MFPD",
    "RelDistFrames",
)

ADVANCED_MOV_NAMES = (
    "RmsModDiff1",
    "RmsNoiseLoudAsym",
    "SegmentalNMR",
    "EHS",
    "AvgLinDist",
)


@dataclass(frozen=True)
class _NetworkConstants:
    amin: tuple[float, ...]
    amax: tuple[float, ...]
    wx: tuple[tuple[float, ...], ...]
    wxb: tuple[float, ...]
    wy: tuple[float, ...]
    wyb: float
    mov_names: tuple[str, ...]


_BASIC_CONSTANTS = _NetworkConstants(
    amin=(
        393.916656,
        361.965332,
        -24.045116,
        1.110661,
        -0.206623,
        0.074318,
        1.113683,
        0.950345,
        0.029985,
        0.000101,
        0.0,
    ),
    amax=(
        921.0,
        881.131226,
        16.212030,
        107.137772,
        2.886017,
        13.933351,
        63.257874,
        1145.018555,
        14.819740,
        1.0,
        1.0,
    ),
    wx=(
        (-0.502657, 0.436333, 1.219602),
        (4.307481, 3.246017, 1.123743),
        (4.984241, -2.211189, -0.192096),
        (0.051056, -1.762424, 4.331315),
        (2.321580, 1.789971, -0.754560),
        (-5.303901, -3.452257, -10.814982),
        (2.730991, -6.111805, 1.519223),
        (0.624950, -1.331523, -5.955151),
        (3.102889, 0.871260, -5.922878),
        (-1.051468, -0.939882, -0.142913),
        (-1.804679, -0.503610, -0.620456),
    ),
    wxb=(-2.518254, 0.654841, -2.207228),
    wy=(-3.817048, 4.107138, 4.629582),
    wyb=-0.307594,
    mov_names=BASIC_MOV_NAMES,
)

_ADVANCED_CONSTANTS = _NetworkConstants(
    amin=(13.298751, 0.041073, -25.018791, 0.061560, 0.02452),
    amax=(2166.5, 13.24326, 13.46708, 10.226771, 14.224874),
    wx=(
        (21.211773, -39.013052, -1.382553, -14.545348, -0.320899),
        (-8.981803, 19.956049, 0.935389, -1.686586, -3.238586),
        (1.633830, -2.877505, -7.442935, 5.606502, -1.783120),
        (6.103821, 19.587435, -0.240284, 1.088213, -0.511314),
        (11.556344, 3.892028, 9.720441, -3.287205, -11.031250),
    ),
    wxb=(1.330890, 2.686103, 2.096598, -1.327851, 3.087055),
    wy=(-4.696996, -3.289959, 7.004782, 6.651897, 4.009144),
    wyb=-1.360308,
    mov_names=ADVANCED_MOV_NAMES,
)

_BMIN = -3.98
_BMAX = 0.22


class PEAQNetwork(torch.nn.Module):
    """PyTorch implementation of GstPEAQ DI/ODG mapping from MOV vectors."""

    def __init__(
        self,
        mode: str = "basic",
        clamp_movs: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if mode not in ("basic", "advanced"):
            raise ValueError("mode must be 'basic' or 'advanced'")
        const = _BASIC_CONSTANTS if mode == "basic" else _ADVANCED_CONSTANTS

        self.mode = mode
        self.clamp_movs = clamp_movs
        self.mov_names = const.mov_names

        self.register_buffer("amin", torch.tensor(const.amin, dtype=dtype))
        self.register_buffer("amax", torch.tensor(const.amax, dtype=dtype))
        self.register_buffer("wx", torch.tensor(const.wx, dtype=dtype))
        self.register_buffer("wxb", torch.tensor(const.wxb, dtype=dtype))
        self.register_buffer("wy", torch.tensor(const.wy, dtype=dtype))
        self.register_buffer("wyb", torch.tensor(const.wyb, dtype=dtype))
        self.register_buffer("bmin", torch.tensor(_BMIN, dtype=dtype))
        self.register_buffer("bmax", torch.tensor(_BMAX, dtype=dtype))

    @property
    def mov_count(self) -> int:
        return len(self.mov_names)

    def forward(self, movs: torch.Tensor, return_di: bool = False):
        movs = torch.as_tensor(movs, dtype=self.amin.dtype, device=self.amin.device)
        if movs.shape[-1] != self.mov_count:
            raise ValueError(
                f"expected last dim to be {self.mov_count} for mode={self.mode}, "
                f"got {movs.shape[-1]}"
            )

        m = (movs - self.amin) / (self.amax - self.amin)
        if self.clamp_movs:
            m = torch.clamp(m, 0.0, 1.0)

        x = self.wxb + m @ self.wx
        di = self.wyb + torch.sigmoid(x) @ self.wy
        odg = self.bmin + (self.bmax - self.bmin) * torch.sigmoid(di)

        if return_di:
            return di, odg
        return odg


def _to_2d_tensor(movs: Sequence[Sequence[float]] | Sequence[float], dim: int) -> torch.Tensor:
    x = torch.as_tensor(movs, dtype=torch.float64)
    if x.ndim == 1:
        if x.shape[0] != dim:
            raise ValueError(f"expected {dim} MOVs, got {x.shape[0]}")
        return x.unsqueeze(0)
    if x.ndim == 2 and x.shape[1] == dim:
        return x
    raise ValueError(f"expected shape ({dim},) or (N, {dim}), got {tuple(x.shape)}")


def compute_di_basic(movs: Sequence[Sequence[float]] | Sequence[float], clamp_movs: bool = False):
    model = PEAQNetwork(mode="basic", clamp_movs=clamp_movs)
    x = _to_2d_tensor(movs, model.mov_count)
    di, _ = model(x, return_di=True)
    return di.squeeze(0) if di.numel() == 1 else di


def compute_di_advanced(
    movs: Sequence[Sequence[float]] | Sequence[float], clamp_movs: bool = False
):
    model = PEAQNetwork(mode="advanced", clamp_movs=clamp_movs)
    x = _to_2d_tensor(movs, model.mov_count)
    di, _ = model(x, return_di=True)
    return di.squeeze(0) if di.numel() == 1 else di


def compute_odg_from_di(di: Iterable[float] | float):
    x = torch.as_tensor(di, dtype=torch.float64)
    odg = _BMIN + (_BMAX - _BMIN) * torch.sigmoid(x)
    return odg.item() if odg.numel() == 1 else odg
