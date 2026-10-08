"""The goal-reaching value (hjepa/value.py): the terminal boundary of its targets, its rank metric and
its saved artifacts (codex audit findings 9, 13, 19, 30)."""

import hashlib
import json

import numpy as np
import torch

import value


def test_a_step_into_the_goal_does_not_bootstrap():
    # Two episodes of 2 frames: the only state with a next frame is frame 0 of each (starts).
    z = torch.arange(4, dtype=torch.float32)[:, None]
    last = np.array([1, 1, 3, 3])
    starts = np.flatnonzero(np.arange(len(last)) < last)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s, s2, g, r, cont = value.sample(rng, z, last, starts, 64, device="cpu")
        idx, goal = s[:, 0].long(), g[:, 0].long()
        at_goal, next_is_goal = goal == idx, goal == idx + 1
        assert torch.all(r[at_goal] == 0) and torch.all(cont[at_goal] == 0)
        assert torch.all(r[next_is_goal] == -1) and torch.all(cont[next_is_goal] == 0)  # target -1, not -1 + 0.99 V(g, g)
        other = ~(at_goal | next_is_goal)
        assert torch.all(r[other] == -1) and torch.all(cont[other] == 1)
        assert next_is_goal.any()


def test_spearman_ties_and_constant_predictions():
    steps = np.array([6, 6, 7, 7])
    assert value.spearman(np.zeros(4), steps) is None  # a constant prediction has no rank correlation
    assert value.spearman(np.array([1.0, 1.0, 2.0, 2.0]), steps) == 1.0
    assert abs(value.spearman(np.array([1.0, 2.0, 3.0, 4.0]), steps) - 0.894) < 1e-3  # average ranks for ties


def test_saved_value_is_bound_to_its_encoder_and_published_whole(tmp_path):
    ckpt = tmp_path / "m_object.ckpt"
    ckpt.write_bytes(b"encoder weights")
    sha = value.sha256_file(ckpt)
    assert sha == hashlib.sha256(b"encoder weights").hexdigest()
    head = value.GoalValue(8)
    value.save(tmp_path, head, {"ckpt": str(ckpt), "encoder_ckpt_sha256": sha})
    blob = torch.load(tmp_path / "value.pt", weights_only=False)
    assert blob["encoder_ckpt_sha256"] == sha and blob["report"]["encoder_ckpt_sha256"] == sha
    assert json.loads((tmp_path / "value.json").read_text())["encoder_ckpt_sha256"] == sha
    assert not list(tmp_path.glob("*.tmp"))
    assert value.load_value(tmp_path / "value.pt", device="cpu").cfg == head.cfg
