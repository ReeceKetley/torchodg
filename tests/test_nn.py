import math

import torch

from torchodg.nn import PEAQNetwork, compute_odg_from_di


def _manual_network(movs, amin, amax, wx, wxb, wy, wyb):
    normalized = [(m - lo) / (hi - lo) for m, lo, hi in zip(movs, amin, amax)]
    x = list(wxb)
    for i in range(len(normalized)):
        for j in range(len(x)):
            x[j] += wx[i][j] * normalized[i]
    return wyb + sum(w * (1.0 / (1.0 + math.exp(-v))) for w, v in zip(wy, x))


def test_basic_network_matches_manual_formula():
    model = PEAQNetwork(mode="basic")
    movs = [
        600.0,
        590.0,
        0.0,
        20.0,
        0.5,
        2.0,
        20.0,
        100.0,
        1.2,
        0.1,
        0.4,
    ]
    di, odg = model(torch.tensor(movs, dtype=torch.float64).unsqueeze(0), return_di=True)
    di_manual = _manual_network(
        movs,
        model.amin.tolist(),
        model.amax.tolist(),
        model.wx.tolist(),
        model.wxb.tolist(),
        model.wy.tolist(),
        float(model.wyb.item()),
    )
    assert math.isclose(di.item(), di_manual, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(odg.item(), compute_odg_from_di(di_manual), rel_tol=0.0, abs_tol=1e-12)


def test_advanced_network_accepts_batch_shape():
    model = PEAQNetwork(mode="advanced")
    batch = torch.tensor(
        [
            [50.0, 1.0, -2.0, 0.5, 1.0],
            [80.0, 2.0, 1.0, 2.0, 3.0],
        ],
        dtype=torch.float64,
    )
    di, odg = model(batch, return_di=True)
    assert tuple(di.shape) == (2,)
    assert tuple(odg.shape) == (2,)

