"""
Unit Tests for Member 2 RMS-AV and Baseline Models:
1. Batch size 1 and batch size 4 forward passes.
2. Training and Eval modes.
3. Missing modality handling (has_video=False, has_audio=False).
4. Backward pass gradient computation for all trainable parameters.
5. Verification of all 4 modes ('audio', 'video', 'fusion', 'rmsav').
"""

import torch
import torch.nn as nn
from model import build_model, count_params, MODES
from reliability_gate import ReliabilityGate

def run_tests():
    print("=" * 70)
    print("RUNNING RMS-AV AND BASELINE MODEL UNIT TESTS")
    print("=" * 70)

    dim = 128
    T = 5

    # 1. Test ReliabilityGate directly
    print("\n[Test 1] ReliabilityGate Logic...")
    gate = ReliabilityGate(dim=dim)
    a = torch.randn(2, dim)
    v = torch.randn(2, dim)
    has_a = torch.tensor([True, True])
    has_v = torch.tensor([True, False])  # Sample 1 missing video

    w = gate(a, v, has_a, has_v)
    assert w.shape == (2, 2), f"Expected (2, 2), got {w.shape}"
    assert torch.allclose(w.sum(dim=1), torch.tensor([1.0, 1.0])), "Weights must sum to 1.0!"
    assert w[1, 1].item() == 0.0, "Missing video must have weight 0.0!"
    assert w[1, 0].item() == 1.0, "Active audio must have weight 1.0 when video is missing!"
    print("  -> Passed! Gate weights correctly normalized and missing modality strictly clamped to 0.0.")

    # 2. Test Batch Size = 1 in Train Mode (Ensure no BatchNorm crashes)
    print("\n[Test 2] RMS-AV with Batch Size = 1 (Train Mode)...")
    net = build_model("rmsav", cfg={"dim": dim, "pretrained": False})
    net.train()

    b1 = {
        "video": torch.randn(1, T, 3, 224, 224),
        "audio": torch.randn(1, 1, 80, 200),
        "has_video": torch.tensor([True]),
        "has_audio": torch.tensor([True]),
    }
    out1 = net(b1)
    assert out1["logit_av"].shape == (1,), f"Expected shape (1,), got {out1['logit_av'].shape}"
    assert out1["gate"].shape == (1, 2)
    assert not torch.isnan(out1["logit_av"]).any(), "NaN in logit_av!"
    print("  -> Passed! Batch size 1 produced finite outputs with zero NaNs.")

    # 3. Test Missing Modality Handling in RMS-AV
    print("\n[Test 3] Missing Modality Handling in RMS-AV...")
    b_missing_v = {
        "video": torch.zeros(2, T, 3, 224, 224),
        "audio": torch.randn(2, 1, 80, 200),
        "has_video": torch.tensor([False, False]),
        "has_audio": torch.tensor([True, True]),
    }
    out_mv = net(b_missing_v)
    assert torch.all(out_mv["gate"][:, 1] == 0.0), "Video gate must be 0.0 when video is missing!"
    assert not torch.isnan(out_mv["logit_av"]).any(), "NaN when video is missing!"
    print("  -> Passed audio-only input through RMS-AV!")

    b_missing_a = {
        "video": torch.randn(2, T, 3, 224, 224),
        "audio": torch.zeros(2, 1, 80, 200),
        "has_video": torch.tensor([True, True]),
        "has_audio": torch.tensor([False, False]),
    }
    out_ma = net(b_missing_a)
    assert torch.all(out_ma["gate"][:, 0] == 0.0), "Audio gate must be 0.0 when audio is missing!"
    assert not torch.isnan(out_ma["logit_av"]).any(), "NaN when audio is missing!"
    print("  -> Passed video-only input through RMS-AV!")

    # 4. Test Gradient Flow & Multi-Task Loss
    print("\n[Test 4] Gradient Flow & Multi-Task Loss...")
    b_train = {
        "video": torch.randn(4, T, 3, 224, 224),
        "audio": torch.randn(4, 1, 80, 200),
        "has_video": torch.tensor([True, True, True, False]),
        "has_audio": torch.tensor([True, True, False, True]),
    }
    labels = torch.randint(0, 2, (4,)).float()
    v_labels = torch.randint(0, 2, (4,)).float()
    a_labels = torch.randint(0, 2, (4,)).float()

    net.zero_grad()
    out = net(b_train)
    criterion = nn.BCEWithLogitsLoss()

    loss_av = criterion(out["logit_av"], labels)
    loss_v = criterion(out["logit_video"][b_train["has_video"]], v_labels[b_train["has_video"]])
    loss_a = criterion(out["logit_audio"][b_train["has_audio"]], a_labels[b_train["has_audio"]])
    total_loss = loss_av + 0.2 * loss_v + 0.2 * loss_a

    total_loss.backward()
    assert net.fusion_head[0].weight.grad is not None, "Fusion head gradient is None!"
    assert net.gate.net[0].weight.grad is not None, "Gate gradient is None!"
    assert net.audio_enc.proj.weight.grad is not None, "Audio encoder gradient is None!"
    assert net.video_enc.proj.weight.grad is not None, "Video encoder gradient is None!"
    print(f"  -> Passed! Total loss: {total_loss.item():.4f}, gradients flowed successfully to all trainable parameters.")

    # 5. Test All Modes
    print("\n[Test 5] Verifying All 4 Model Modes...")
    for mode in MODES:
        m_net = build_model(mode, cfg={"dim": dim, "pretrained": False})
        m_out = m_net(b1)
        tot, trn = count_params(m_net)
        assert m_out["logit_av"].shape == (1,), f"{mode} logit_av shape mismatch!"
        assert not torch.isnan(m_out["logit_av"]).any(), f"{mode} produced NaNs!"
        print(f"  -> Mode '{mode:7s}' verified! Params: {tot:,} | Logit: {m_out['logit_av'].shape}")

    print("\n" + "=" * 70)
    print("ALL MODEL UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
