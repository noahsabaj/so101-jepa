"""hjepa/lewam_plus.py: with its changes off, or the proprio token dropped, it computes what LeWAM does."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hjepa"))
lewam_plus = pytest.importorskip("lewam_plus")

CFG = dict(fs=5, action_raw_dim=2, img_size=64, encoder_size="tiny", encoder_backbone="resnet18dp", z_dim=32,
           proj_hidden=64, d_model=64, n_heads=4, depth=2, dropout=0.0, n_flow_steps=4, num_actions_pred=10,
           num_states_pred=1, history_len=2, history_stride=5, goal_type="sampled", H_max=10, action_head="flow",
           state_head="mse", flow_alpha_act=1.0, flow_alpha_state=1.0, num_views=1, goal_view_indices=[0])


def models(**plus):
    torch.manual_seed(0)
    base = lewam_plus._build_model(dict(CFG)).eval()
    with torch.no_grad():  # adaLN-zero starts with the slots' gates at 0 (no attention): open them
        for block in base.blocks:
            block.ada_action[-1].weight.normal_(0, 0.05)
    model = lewam_plus.build_model({**CFG, "plus": {**plus, "proprio_dim": 3} if plus.get("proprio_keys") else plus})
    missing, unexpected = model.load_state_dict(
        {k: v for k, v in base.state_dict().items() if k != "attends"}, strict=False)
    assert not unexpected and set(missing) <= {"attends", "proprio_proj.weight", "proprio_proj.bias", "proprio_type",
                                                "step_proj.weight", "step_proj.bias"}, missing
    return base, model.eval()


def inputs(B=3):
    g = torch.Generator().manual_seed(1)
    z = torch.randn(B, 2, 32, generator=g)
    pad = torch.tensor([[True, False], [False, False], [False, False]])[:B]
    clean = torch.randn(B, 5, 2, generator=g)
    noisy = torch.randn(B, 10, 2, generator=g)
    tau = torch.rand(B, generator=g)
    goal = torch.randn(B, 32, generator=g)
    h = torch.rand(B, generator=g)
    return z, pad, clean, noisy, tau, goal, h


@pytest.mark.parametrize("plus", [{}, {"proprio_keys": ["p"]}, {"shortcut": True}])
def test_forward_matches_lewam(plus):
    base, model = models(**plus)
    z, pad, clean, noisy, tau, goal, h = inputs()
    keep = torch.tensor([True, False, True])
    want = base.forward_mot(z, pad, clean, noisy, None, tau, None, goal, h, keep)
    got = model.forward_mot(z, pad, clean, noisy, None, tau, None, goal, h, keep)  # proprio dropped (no context)
    for a, b in zip(want, got):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_proprio_token_changes_outputs():
    _, model = models(proprio_keys=["p"])
    z, pad, clean, noisy, tau, goal, h = inputs()
    prop = torch.randn(3, 3)
    with torch.no_grad():
        model.proprio_proj.weight.normal_()
    off = model.forward_mot(z, pad, clean, noisy, None, tau, None, goal, h, None, prop, torch.zeros(3, dtype=torch.bool))
    on = model.forward_mot(z, pad, clean, noisy, None, tau, None, goal, h, None, prop, torch.ones(3, dtype=torch.bool))
    assert not torch.allclose(off[0], on[0]) and not torch.allclose(off[1], on[1])
    ti, at = model.token_indices, model.token_indices["proprio"][0]
    assert not model.attends[ti["hist"] + ti["goal"], at].any()
    assert model.attends[ti["state_slots"] + ti["clean_actions"] + ti["action_slots"], at].all()
    assert model.attends[at].nonzero().flatten().tolist() == ti["hist"] + [at]
    assert model.layout[0] == ti["clean_actions"][0] == at + 1  # proprio ends the state stream


def test_predict_actions_and_guidance():
    base, model = models()
    z, pad, _, _, _, goal, h = inputs()
    noise = torch.randn(3, 10, 2, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        torch.testing.assert_close(base.predict_actions(z, pad, goal, h, noise), model.predict_actions(z, pad, goal, h, noise))
        model.ctx = {"cfg_w": 0.0}  # w = 0: the goal-free velocity only
        free = model.predict_actions(z, pad, goal, h, noise)
        model.ctx = {}
        keep = torch.zeros(3, dtype=torch.bool)
        x = noise.clone()
        for i in range(4):
            v, _ = base.forward_mot(z, pad, torch.zeros(3, 5, 2), x, torch.zeros(3, 1, 32), torch.full((3,), i / 4),
                                   torch.ones(3), goal, h, keep)
            x = x + v / 4
        torch.testing.assert_close(free, x, rtol=1e-5, atol=1e-5)


def test_training_terms_run():
    _, model = models(shortcut=True, roll_k=3, proprio_keys=["p"])
    model.train()
    z, pad, clean, noisy, tau, goal, h = inputs()
    target = torch.randn(3, 10, 2)
    loss = model.shortcut_loss(z, pad, target, torch.ones(3, 10), goal, h, None, torch.randn(3, 3), torch.ones(3, dtype=torch.bool))
    first = torch.randn(3, 1, 32, requires_grad=True)
    roll = model.rollout_loss(z, pad, first, torch.randn(3, 15, 2), torch.randn(3, 2, 32), torch.ones(3, 3))
    (loss + roll).backward()
    assert first.grad.abs().sum() > 0 and model.step_proj.weight.grad is not None
