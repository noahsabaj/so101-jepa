"""Is LeWAM's compiled training step right? One training step of LeWAM's own batch code (run_batch: encoder,
predictor loss, SIGReg) on a random batch, its parameter gradients under each speed mode against eager fp32, with
the same random numbers (inductor fallback_random) and no dropout. Before 2026-10-10 Inductor's GroupNorm backward
gave every GroupNorm weight gradient a 0.6 error at 512 images (lewam_common.GN_ATEN); this checks the whole step.

    python hjepa/jobs/step_ref.py MODEL BATCH [BATCH ...]
    SR_MODES=compiled,graphs,gninductor  SR_CALLS=3  SR_AMP=1 (bf16 autocast; then eager bf16 is shown too)

MODEL: a config in hjepa/config/lewam/ (e.g. repro-cube). Prints per mode and call the loss, the gradient norm and
the worst parameters (relative error of the gradient against eager fp32). Pass: about eager bf16's error in bf16,
under 1e-2 in fp32 (GroupNorm sums over a whole batch of maps).
"""
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewam_bench import T, load_cfg  # noqa: E402
from lewam_common import AFTER_CROP, COMPILED, GN_ATEN, after_crop  # noqa: E402

SKIP = ("encoder.kp_conv.bias", "encoder.projector.net.0.bias")  # zero in exact math (softmax, BatchNorm)


def random_batch(cfg, B, action_dim, gen):
    m, size = cfg.model, cfg.data.img_size
    frames = lambda *shape: torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)  # noqa: E731
    return tuple(x.cuda() for x in (
        frames(B, m.history_len, 3, size, size), torch.zeros(B, m.history_len, dtype=torch.bool),
        torch.randn(B, m.num_actions_pred, action_dim, generator=gen), torch.ones(B, m.num_actions_pred),
        frames(B, m.num_states_pred, 3, size, size), torch.ones(B, m.num_states_pred),
        frames(B, 3, size, size), torch.randint(1, m.H_max + 1, (B,), generator=gen).float() / m.H_max))


def set_mode(model, loss_eager, mode):
    """eager | compiled (the training default when compiling) | graphs (all three as CUDA graphs) | gninductor
    (compiled, Inductor's own GroupNorm: the old bug)"""
    torch._dynamo.reset()
    COMPILED["fns"] = {}
    GN_ATEN["on"] = mode in ("compiled", "graphs")
    graphs = "reduce-overhead" if mode == "graphs" else None
    model.loss = loss_eager
    AFTER_CROP["fn"] = after_crop
    COMPILED["on"], COMPILED["mode"] = mode != "eager", graphs
    if mode == "eager":
        return
    encode = torch.compile(after_crop, fullgraph=True, dynamic=False, mode=graphs)
    AFTER_CROP["fn"] = (lambda encoder, pixels: encode(encoder, pixels).clone()) if graphs else encode
    loss = torch.compile(loss_eager, mode=graphs)
    model.loss = (lambda *a, **k: tuple(t.clone() for t in loss(*a, **k))) if graphs else loss


def step(model, sigreg, cfg, batch, seed, amp):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    ctx = torch.autocast("cuda", torch.bfloat16) if amp else nullcontext()
    with ctx:
        loss, terms, _ = T["run_batch"](batch, model, None, sigreg, cfg, "cuda", train=True)
    loss.backward()
    torch.cuda.synchronize()
    grads = {n: p.grad.detach().double().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss), grads


def main(model_name, sizes):
    inductor_config.fallback_random = True
    cfg = load_cfg(model_name)
    modes = os.environ.get("SR_MODES", "compiled,graphs,gninductor").split(",")
    calls, amp = int(os.environ.get("SR_CALLS", "3")), os.environ.get("SR_AMP") == "1"
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, {model_name}, amp {amp}", flush=True)
    for B in sizes:
        torch.manual_seed(0)
        model = T["build_model"](T["model_config"](cfg, 5)).cuda().train()
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
            if hasattr(mod, "attn_dropout"):
                mod.attn_dropout = 0.0
        loss_eager, sigreg = model.loss, T["SIGReg"]().cuda()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        batch = random_batch(cfg, B, 5, torch.Generator().manual_seed(1))
        set_mode(model, loss_eager, "eager")
        ref = [step(model, sigreg, cfg, batch, 100 + c, False) for c in range(calls)]
        for mode in (["eager"] if amp else []) + modes:
            model.load_state_dict(state)
            set_mode(model, loss_eager, mode)
            for c in range(calls):
                loss, got = step(model, sigreg, cfg, batch, 100 + c, amp)
                r = ref[c][1]
                err = {n: ((got[n] - g).norm() / (g.norm() + 1e-30)).item() for n, g in r.items() if n not in SKIP}
                worst = sorted(err, key=err.get, reverse=True)[:3]
                norm = torch.sqrt(sum((g ** 2).sum() for g in got.values())).item()
                print(f"B {B} {mode:11s} call {c}: loss {loss:.5f} (eager fp32 {ref[c][0]:.5f}), grad norm {norm:.4g}, "
                      "worst " + ", ".join(f"{n.removeprefix('encoder.')} {err[n]:.1e}" for n in worst), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(a) for a in sys.argv[2:]] or [16])
