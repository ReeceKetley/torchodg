import torch

from torchodg.loss import PEAQProxyLoss


def test_proxy_loss_backward_has_finite_grad():
    torch.manual_seed(0)
    b, t = 2, 48000
    ref = torch.randn(b, t, dtype=torch.float32)
    test = (ref + 0.01 * torch.randn_like(ref)).detach().requires_grad_(True)

    criterion = PEAQProxyLoss(sample_rate=48000, clamp_movs=True, reduction="mean")
    loss = criterion(ref, test)
    loss.backward()

    assert test.grad is not None
    assert torch.isfinite(test.grad).all()
    assert test.grad.abs().sum().item() > 0.0


def test_identity_has_higher_odg_than_noisy():
    torch.manual_seed(1)
    b, t = 1, 48000
    ref = torch.randn(b, t, dtype=torch.float32)
    noisy = ref + 0.03 * torch.randn_like(ref)

    criterion = PEAQProxyLoss(sample_rate=48000, clamp_movs=True, reduction="none")
    out_same = criterion(ref, ref, return_details=True)
    out_noisy = criterion(ref, noisy, return_details=True)

    assert out_same.odg.shape == out_noisy.odg.shape
    assert out_same.odg.item() >= out_noisy.odg.item()


def test_alignment_option_handles_small_delay():
    torch.manual_seed(2)
    b, t = 1, 12000
    ref = torch.randn(b, t, dtype=torch.float32)
    delay = 128
    test = torch.cat([torch.zeros(b, delay), ref[:, :-delay]], dim=-1)

    plain = PEAQProxyLoss(sample_rate=48000, clamp_movs=True, reduction="none")
    aligned = PEAQProxyLoss(
        sample_rate=48000,
        clamp_movs=True,
        reduction="none",
        align_max_lag_samples=256,
    )
    out_plain = plain(ref, test, return_details=True)
    out_aligned = aligned(ref, test, return_details=True)
    assert out_aligned.odg.item() >= out_plain.odg.item()


def test_regularized_loss_returns_components_and_grad():
    torch.manual_seed(3)
    b, t = 2, 16384
    ref = torch.randn(b, t, dtype=torch.float32)
    test = (ref + 0.02 * torch.randn_like(ref)).detach().requires_grad_(True)

    criterion = PEAQProxyLoss(
        sample_rate=48000,
        clamp_movs=True,
        reduction="mean",
        waveform_l1_weight=0.1,
        stft_l1_weight=0.01,
    )
    out = criterion(ref, test, return_details=True)
    out.loss.backward()

    assert out.odg_loss is not None
    assert out.waveform_l1 is not None
    assert out.stft_l1 is not None
    assert test.grad is not None
    assert torch.isfinite(test.grad).all()


def test_margin_mode_nonnegative_per_item_loss():
    torch.manual_seed(4)
    b, t = 1, 8192
    ref = torch.randn(b, t, dtype=torch.float32)
    test = ref + 0.1 * torch.randn_like(ref)

    criterion = PEAQProxyLoss(
        sample_rate=48000,
        reduction="none",
        odg_margin_target=-1.0,
    )
    out = criterion(ref, test, return_details=True)
    assert out.odg_loss is not None
    assert out.odg_loss.shape == (b,)
    assert torch.all(out.odg_loss >= 0.0)
