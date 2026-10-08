"""Evaluation integrity: closed-loop run identity, pick measurements, rung verdicts, paired comparisons
and the jump test's common cases (codex audit findings 6, 7, 8, 10, 12)."""

import json

import numpy as np
import pytest

import closed_loop
import compare
import jump_test


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- 6: closed-loop resume -------------------------------------------------------------------

def test_resume_skips_only_rows_of_the_same_run(tmp_path):
    out = tmp_path / "reach.jsonl"
    _write_rows(out, [dict(task="reach", seed=s, success=True, run_id="aaa") for s in (5000, 5001)])
    assert closed_loop.done_seeds(out, "aaa") == {5000, 5001}
    with pytest.raises(SystemExit):
        closed_loop.done_seeds(out, "bbb")  # another model or planner: stop, never mix


def test_resume_refuses_rows_without_identity(tmp_path):
    out = tmp_path / "reach.jsonl"
    _write_rows(out, [dict(task="reach", seed=5000, success=True)])
    with pytest.raises(SystemExit):
        closed_loop.done_seeds(out, "aaa")


def _fake_run(tmp_path, ckpt_bytes):
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    (d / "m_object.ckpt").write_bytes(ckpt_bytes)
    (d / "normalizer.pt").write_bytes(b"norm")
    (d / "config.yaml").write_text("level1: {}\n")
    return d / "m_object.ckpt"


def test_run_identity_follows_checkpoint_bytes_config_and_task(tmp_path):
    ckpt = _fake_run(tmp_path, b"weights-seed42")
    a = closed_loop.run_identity("reach", ckpt, "so101_flat_cem")
    assert a == closed_loop.run_identity("reach", ckpt, "so101_flat_cem")
    assert a["ckpt_sha256"] and a["planner_config_sha256"]
    assert a["run_id"] != closed_loop.run_identity("pick", ckpt, "so101_flat_cem")["run_id"]
    assert a["run_id"] != closed_loop.run_identity("reach", ckpt, "so101_flat_cem", ["solver.n_steps=5"])["run_id"]
    assert a["run_id"] != closed_loop.run_identity("reach", ckpt, "so101_flat_cem_fast")["run_id"]
    ckpt = _fake_run(tmp_path, b"weights-seed43")  # same path, another training seed
    assert a["run_id"] != closed_loop.run_identity("reach", ckpt, "so101_flat_cem")["run_id"]


# --- 8: pick measurements --------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    return closed_loop.SO101Env(randomize=False)


def _place_and_wait(env, xy, steps=6):
    env.reset(5000)
    env.set_cube(xy, 0.0)
    track = []
    for _ in range(steps):
        env.step(np.zeros(6))
        track.append(env.cube_pose()[:3])
    return track


def test_pushed_cube_passes_the_fixed_mark_but_not_the_strict_one(env):
    xy = np.array([0.2, 0.0])
    track = _place_and_wait(env, xy)
    goal = dict(cube=np.array([xy[0] + 0.01, xy[1], 0.0]))
    r = closed_loop.pick_outcome(env, goal, picked_up=False, track=track)  # it never left the table
    assert r["success"] and r["resting"] and r["on_table"] and r["released"] and r["settled"]
    assert not r["picked_up"] and not r["lifted"] and not r["success_strict"]
    json.dumps(r)  # plain Python values only
    r = closed_loop.pick_outcome(env, goal, picked_up=True, track=track)
    assert r["success_strict"]


class _NonfinitePlanner:
    """The planner raises when no candidate has a finite cost (hjepa/planner.py)."""

    def seed(self, seed):
        pass

    def reset(self):
        pass

    def record(self, obs, action):
        pass

    def plan(self, *args, **kwargs):
        raise FloatingPointError("no finite plan")


def test_a_nonfinite_plan_is_a_failed_trial_not_a_crash(env):
    r = closed_loop.run("reach", _NonfinitePlanner(), env, 1000)
    assert r["nonfinite_plan"] and r["replans"] == 0 and not r["success"]


def test_sliding_cube_is_not_settled(env):
    xy = np.array([0.2, 0.0])
    track = _place_and_wait(env, xy)
    env.data.qvel[env.model.joint("cube").dofadr[0]] = 0.3  # sliding through the target at the last step
    r = closed_loop.pick_outcome(env, dict(cube=np.array([*xy, 0.0])), picked_up=True, track=track)
    assert not r["settled"] and not r["success_strict"]


# --- 7: rung verdicts ------------------------------------------------------------------------

def _rows(task, seeds, successes, strict=None):
    """strict: how many of the successes are real picks (success_strict); None leaves the field out."""
    rows = {s: dict(task=task, seed=s, success=i < successes, error_cm=0.5) for i, s in enumerate(seeds)}
    if strict is not None:
        for i, s in enumerate(seeds):
            rows[s]["success_strict"] = i < strict
    return rows


def test_verdict_only_on_the_exact_test_seeds():
    assert compare.verdict(_rows("reach", range(5000, 5100), 95)).endswith("PASS")
    assert compare.verdict(_rows("reach", range(5000, 5100), 89)).endswith("fail")
    assert "no verdict" in compare.verdict(_rows("reach", range(5000, 5200), 90))  # 90/200 is not 90/100
    assert "no verdict" in compare.verdict(_rows("reach", range(1000, 1100), 100))  # tuning seeds
    assert "incomplete" in compare.verdict(_rows("pick", range(5000, 5030), 30))
    # rung 2 counts real picks only (Noah, 2026-10-08): 30 pushes into the goal place do not pass
    assert compare.verdict(_rows("pick", range(5000, 5050), 30, strict=30)).endswith("PASS")
    assert compare.verdict(_rows("pick", range(5000, 5050), 50, strict=29)).endswith("fail")
    assert "no verdict" in compare.verdict(_rows("pick", range(5000, 5050), 30))  # rows from before the field
    mixed = {**_rows("reach", range(5000, 5050), 50), **_rows("pick", range(5050, 5100), 50)}
    assert "no verdict" in compare.verdict(mixed)


def test_a_file_with_two_runs_is_refused(tmp_path):
    out = tmp_path / "reach.jsonl"
    _write_rows(out, [dict(task="reach", seed=5000, success=True, run_id="a"),
                      dict(task="reach", seed=5001, success=True, run_id="b")])
    with pytest.raises(SystemExit):
        compare.trials(out)


# --- 12: offline pairing ---------------------------------------------------------------------

def _report(ids, vals, fp="data1", t="expert_rank_k5"):
    return {"case_ids": ids, "data_fingerprint": fp, t: [float(np.mean(vals)), 0.1], f"{t}_cases": vals}


def test_pairing_joins_on_case_ids():
    a = _report([[0, 3], [1, 4], [2, 5]], [1.0, 2.0, 3.0])
    b = _report([[2, 5], [0, 3], [1, 4]], [30.0, 10.0, 20.0])
    x, y = compare.paired(a, b, "expert_rank_k5")
    assert (y - x).tolist() == [9.0, 18.0, 27.0]


def test_pairing_uses_the_case_prefix_of_shorter_tests():
    a = _report([[0, 3], [1, 4], [2, 5]], [1.0, 2.0])
    b = _report([[0, 3], [1, 4], [2, 5]], [2.0, 4.0])
    x, y = compare.paired(a, b, "expert_rank_k5")
    assert (y - x).tolist() == [1.0, 2.0]


@pytest.mark.parametrize("b", [
    _report([[0, 3], [1, 4]], [1.0, 2.0], fp="data2"),  # another dataset, same length
    {"expert_rank_k5": [1.5, 0.1], "expert_rank_k5_cases": [1.0, 2.0]},  # no identities
])
def test_pairing_refuses_unmatched_reports(b):
    a = _report([[0, 3], [1, 4]], [1.0, 2.0])
    with pytest.raises(ValueError):
        compare.paired(a, b, "expert_rank_k5")


def test_offline_comparison_stops_unless_unpaired(tmp_path, capsys):
    pa, pb = tmp_path / "offline_a.json", tmp_path / "offline_b.json"
    pa.write_text(json.dumps(_report([[0, 3], [1, 4]], [1.0, 2.0])))
    pb.write_text(json.dumps(_report([[0, 3], [1, 4]], [1.0, 3.0], fp="data2")))
    with pytest.raises(SystemExit):
        compare.offline([str(pa), str(pb)])
    compare.offline([str(pa), str(pb)], unpaired=True)
    assert "(unpaired)" in capsys.readouterr().out


# --- 10: jump test cases ---------------------------------------------------------------------

def test_jump_cases_are_common_and_admissible():
    lens = np.array([300, 40, 120, 300, 51, 52])
    ids = jump_test.common_cases(lens, 200)
    assert ids == jump_test.common_cases(lens, 200)  # the same for every model and stride
    assert len(set(ids)) == len(ids) == 200 and ids == sorted(ids)
    for e, t in ids:
        assert t - 3 * jump_test.CASE_STRIDE >= 0 and t + max(jump_test.HORIZONS) <= lens[e] - 1
