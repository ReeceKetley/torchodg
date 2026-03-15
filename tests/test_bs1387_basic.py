import torch

from torchodg.loss import PEAQProxyLoss


def test_bs1387_basic_backend_degrades_with_noise():
    torch.manual_seed(3)
    ref = torch.randn(1, 2, 16384, dtype=torch.float32)
    noisy = ref + 0.02 * torch.randn_like(ref)

    crit = PEAQProxyLoss(
        sample_rate=48000,
        clamp_movs=True,
        reduction="none",
        backend="bs1387_basic",
        bs_ehs_mode="fast",
    )
    a = crit(ref, ref, return_details=True)
    b = crit(ref, noisy, return_details=True)
    assert a.odg.item() >= b.odg.item()


def test_bs1387_basic_backend_backward():
    torch.manual_seed(4)
    ref = torch.randn(1, 2, 12288, dtype=torch.float32)
    test = (ref + 0.01 * torch.randn_like(ref)).detach().requires_grad_(True)

    crit = PEAQProxyLoss(
        sample_rate=48000,
        clamp_movs=True,
        reduction="mean",
        backend="bs1387_basic",
        bs_ehs_mode="fast",
    )
    loss = crit(ref, test)
    loss.backward()
    assert test.grad is not None
    assert torch.isfinite(test.grad).all()
