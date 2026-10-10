"""Do CUDA graphs (lewam_train.py LEWAM_GRAPHS: torch.compile mode reduce-overhead) change LeWAM's training math?
Kaggle test 1010-122303-3n: with graphs the epoch-1 train state loss was 0.36-0.59, without 0.014-0.022.

    python hjepa/jobs/graphs_check.py rng
    python hjepa/jobs/graphs_check.py train MODEL DATA.h5 VARIANT STEPS OUTDIR
    python hjepa/jobs/graphs_check.py compare OUTDIR

rng: do the random draws inside a compiled function change from call to call, with and without graphs?
train: STEPS training steps (LeWAM's run_batch, clip, AdamW) from one seed on the same 4 batches, the seed set
before each step and fallback_random on (compiled code draws eager's random numbers): every variant should give
eager's losses up to rounding. VARIANT: eager, or a +-joined set of lewam_bench flags (wenc genc csig gsig closs gloss).
compare: each variant's largest relative difference from eager per term, and the first step off by more than 1%.
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rng():
    fns = {
        "rand(B) ** (1/a), as the flow times": lambda x: torch.rand(x.shape[0], device=x.device) ** 0.5 + x[:, 0],
        "randn_like, as the flow noise": lambda x: torch.randn_like(x) + x,
        "randn(D), as SIGReg's projections": lambda x: x + torch.randn(x.shape[-1], device=x.device),
    }
    for grad in (False, True):
        for mode in (None, "reduce-overhead"):
            for name, f in fns.items():
                torch._dynamo.reset()
                g = torch.compile(f, mode=mode, dynamic=False)
                outs = []
                for _ in range(6):
                    x = torch.zeros(64, 8, device="cuda", requires_grad=grad)
                    y = g(x)
                    if grad:
                        y.sum().backward()
                    outs.append(y.detach().clone())
                same = sum(torch.equal(outs[i], outs[i - 1]) for i in range(1, 6))
                print(f"[rng] grad={grad} mode={mode} {name}: {same} of 5 calls repeat the call before"
                      f"{'  <-- FROZEN' if same else ''}", flush=True)


def train(model_name, data, variant, steps, outdir):
    import torch._inductor.config as inductor_config

    inductor_config.fallback_random = True
    from lewam_bench import T, load_cfg, make_data
    from lewam_common import AFTER_CROP, COMPILED, after_crop

    flags = set(variant.split("+")) - {"eager"}
    cfg = load_cfg(model_name)
    cfg.dataset_path, cfg.data.source, cfg.train.batch_size, cfg.train.precision = data, "h5", int(os.environ.get("GC_BS", "4")), "fp32"
    torch.manual_seed(0)
    loader, actions, _ = make_data(cfg, data, 0)
    it = iter(loader)
    batches = [tuple(x.cuda() for x in next(it)) for _ in range(4)]
    torch.manual_seed(0)
    model = T["build_model"](T["model_config"](cfg, actions.shape[1])).cuda()
    # as lewam_train.speedups (and lewam_bench.bench_model)
    AFTER_CROP["fn"] = after_crop
    if flags & {"wenc", "genc"}:
        encode = torch.compile(after_crop, fullgraph=True, dynamic=False,
                               mode="reduce-overhead" if "genc" in flags else None)
        AFTER_CROP["fn"] = (lambda encoder, pixels: encode(encoder, pixels).clone()) if "genc" in flags else encode
    COMPILED["on"], COMPILED["fns"] = bool(flags & {"csig", "gsig"}), {}
    COMPILED["mode"] = "reduce-overhead" if "gsig" in flags else None
    if flags & {"closs", "gloss"}:
        loss_fn = torch.compile(model.loss, mode="reduce-overhead" if "gloss" in flags else None)
        model.loss = (lambda *a, **k: tuple(t.clone() for t in loss_fn(*a, **k))) if "gloss" in flags else loss_fn
    sigreg = T["SIGReg"]().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    rows = []
    for i in range(steps):
        torch.manual_seed(1000 + i)
        model.train()
        loss, terms, _ = T["run_batch"](batches[i % 4], model, None, sigreg, cfg, "cuda", train=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rows.append({k: float(v) for k, v in terms.items()} | {"gnorm": float(gnorm)})
        print(f"[train] {variant} step {i}: " + " ".join(f"{k}={v:.5f}" for k, v in rows[-1].items()), flush=True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{variant}.json").write_text(json.dumps(rows))
    print(f"[train] {variant}: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


def compare(outdir):
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(Path(outdir).glob("*.json"))}
    base = runs.pop("eager")
    for name, rows in runs.items():
        out = []
        for k in base[0]:
            rel = [abs(r[k] - b[k]) / max(abs(b[k]), 1e-6) for r, b in zip(rows, base)]
            first = next((i for i, d in enumerate(rel) if d > 0.01), None)
            out.append(f"{k} max {max(rel):.1e} first>1% {first}")
        print(f"[compare] {name}: " + "; ".join(out), flush=True)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "rng":
        rng()
    elif cmd == "train":
        train(args[0], args[1], args[2], int(args[3]), args[4])
    else:
        compare(args[0])
