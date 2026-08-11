"""Headless training for the clean refine_v3 sudoku solver.

Two datasets, each with a faithful-baseline default and an our-recipe variant,
selected by ``--preset``:

  palm          our arch on Palm et al.'s exact 216k materialised set (uniform
                17..34 givens), no revision — the honest comparison to their 94.1%
                @32 / 96.6% @64.
  palm-ours     + self-revision (revision_prob 0.5), warm-restart LR.
  palm-repro    pure Palm baseline: flat per-pass CE, settledness head off,
                lr 2e-4 cosine. Reproduces their loss inside this arch.

  extreme       our arch on 1000 base Sudoku-Extreme puzzles (HRM/TRM's protocol),
                band+digit augmentation, no transpose, no revision.
  extreme-ours  + self-revision and transpose augmentation.

The faithful-match presets keep clue grading ON: refine_v3 with hard feedback
needs an input-readable gradient (grade the clues, or feed guesses back via
revision) or the readout collapses to a constant predictor. Grading the givens —
a copy task that self-anneals once learned — supplies it without revision.

    python3 train.py --preset extreme --steps 8000 --run-name extreme-clean
    python3 train.py --preset palm --steps 300000 --run-name palm-clean
    python3 train.py --resume logs/<run>/best.pt --steps 20000

Everything the run needs is written under logs/<run-name>/: config.json,
train.log, metrics.jsonl, latest.pt, best.pt.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from model import RefinementUT, ModelConfig, accuracy, count_params, target_mask
from losses import compute_loss
from data import load_base_pool, materialize_pool, Batcher
from evaluate import load_pool, load_ratings, solve

HERE = Path(__file__).resolve().parent  # dir holding these scripts


def _resolve(p: str) -> str:
    """Resolve a data path in a layout-agnostic way: as given (cwd-relative),
    then next to the scripts (standalone repo), then one level up (data/ beside a
    clean/ subdir)."""
    if Path(p).is_absolute() or Path(p).exists():
        return p
    for base in (HERE, HERE.parent):
        if (base / p).exists():
            return str(base / p)
    return p


MODEL_DEFAULTS = dict(refine_steps=32, settle_threshold=0.995, halt_bias_init=-3.0,
                      dropout=0.05)
LOSS_DEFAULTS = dict(settle_weight=1.0, pass_weight=0.1, min_settle_weight=0.158,
                     grade_clues=True, clue_share=0.5, clue_denom_blanks=True)
LOSS_OFF = dict(settle_weight=0.0, pass_weight=0.0, min_settle_weight=0.0,
                grade_clues=False, clue_share=0.0, clue_denom_blanks=False)

# `materialize` N > 0: expand the base pool into a fixed set of N augmentations
# per base puzzle once, then train over that set (HRM/TRM's protocol). 0: use the
# pool as-is (Palm's set is already materialised), unless `augment` resamples a
# symmetry per draw (our variant).
PRESETS: dict[str, dict] = {
    "palm": dict(
        pool="data/mixed-givens-17-34-216000-heldout.npz",
        eval="data/eval-17clue-64.npz",
        d_model=96, n_heads=4, d_ff=384, budget=32,
        materialize=0, augment=False, transpose=False, revision_prob=0.0,
        lr=1e-3, lr_schedule="cosine",
    ),
    "palm-ours": dict(
        pool="data/mixed-givens-17-34-216000-heldout.npz",
        eval="data/eval-17clue-64.npz",
        d_model=96, n_heads=4, d_ff=384, budget=32,
        materialize=0, augment=False, transpose=False, revision_prob=0.5,
        lr=1e-3, lr_schedule="cycles",
    ),
    "palm-repro": dict(
        pool="data/mixed-givens-17-34-216000-heldout.npz",
        eval="data/eval-17clue-64.npz",
        d_model=96, n_heads=4, d_ff=384, budget=32,
        materialize=0, augment=False, transpose=False, revision_prob=0.0,
        lr=2e-4, lr_schedule="cosine", loss=LOSS_OFF,
    ),
    "extreme": dict(  # faithful match: 1000 base × 1000 materialised augmentations
        pool="data/extreme-train-1000.npz",
        eval="data/sudoku-extreme-test.csv",
        d_model=96, n_heads=4, d_ff=384, budget=32,
        materialize=1000, augment=False, transpose=False, revision_prob=0.0,
        lr=1e-3, lr_schedule="cycles",
    ),
    "extreme-ours": dict(  # our variant: per-draw augmentation + revision + transpose
        pool="data/extreme-train-1000.npz",
        eval="data/sudoku-extreme-test.csv",
        d_model=96, n_heads=4, d_ff=384, budget=32,
        materialize=0, augment=True, transpose=True, revision_prob=0.5,
        lr=1e-3, lr_schedule="cycles",
    ),
    "extreme-full": dict(  # full 3.83M-puzzle Sudoku-Extreme train split, no
                           # materialisation needed — the pool is already diverse
        pool="data/extreme-full-train.npz",
        eval="data/sudoku-extreme-test.csv",
        d_model=192, n_heads=8, d_ff=768, budget=32,
        materialize=0, augment=False, transpose=False, revision_prob=0.0,
        lr=1e-3, lr_schedule="cycles",
    ),
}


class LR:
    """Two schedules. cosine: warmup → one cosine decay to the floor. cycles:
    warmup → exponential half-life decay, with a warm restart each time held-out
    solve rate stalls for ``patience`` evals (peak decays 0.7x per restart)."""

    def __init__(self, base: float, schedule: str, warmup: int, max_steps: int,
                 min_frac: float = 0.1, half_life: int = 1000, peak_decay: float = 0.7,
                 patience: int = 150):
        self.base, self.schedule, self.warmup = base, schedule, warmup
        self.max_steps, self.min_frac = max_steps, min_frac
        self.half_life, self.peak_decay, self.patience = half_life, peak_decay, patience
        self.floor = base * min_frac
        self.cycle_start = 0
        self.cycle_peak = base
        self.best = 0.0
        self.stall = 0

    def at(self, step: int) -> float:
        if self.schedule == "cosine":
            if step < self.warmup:
                return self.base * (step + 1) / self.warmup
            span = max(1, self.max_steps - self.warmup)
            prog = min(1.0, (step - self.warmup) / span)
            return self.floor + 0.5 * (self.base - self.floor) * (1 + math.cos(math.pi * prog))
        d = step - self.cycle_start
        if d < self.warmup:
            return self.cycle_peak * (d + 1) / self.warmup
        return self.floor + (self.cycle_peak - self.floor) * 0.5 ** ((d - self.warmup) / self.half_life)

    def on_eval(self, step: int, metric: float, log: logging.Logger) -> None:
        """cycles only: track the best held-out metric and restart on a plateau."""
        if self.schedule != "cycles" or self.patience <= 0:
            return
        if metric > self.best + 1e-4:
            self.best, self.stall = metric, 0
            return
        self.stall += 1
        if self.stall >= self.patience:
            cur = self.at(step)
            self.cycle_peak = max(self.floor, min(self.peak_decay * self.cycle_peak, 2.0 * cur))
            self.cycle_start = step
            self.stall = 0
            self.best = metric
            log.info("lr restart (plateau) | peak %.2e", self.cycle_peak)


def build(preset: dict, args) -> tuple[ModelConfig, dict, dict]:
    mcfg = ModelConfig(
        d_model=args.d_model or preset["d_model"],
        n_heads=args.heads or preset["n_heads"],
        d_ff=args.d_ff or preset["d_ff"],
        **MODEL_DEFAULTS,
    )
    loss_kw = dict(preset.get("loss", LOSS_DEFAULTS))
    data_kw = dict(
        augment=preset["augment"],
        transpose=preset["transpose"] if args.transpose is None else args.transpose,
        revision_prob=preset["revision_prob"] if args.revision_prob is None else args.revision_prob,
    )
    return mcfg, loss_kw, data_kw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="extreme")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None, help="pass budget T (train + eval)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--revision-prob", type=float, default=None)
    ap.add_argument("--transpose", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    ap.add_argument("--materialize", type=int, default=None,
                    help="augmentations per base puzzle to materialise once (0 = off)")
    ap.add_argument("--pool", default=None, help="override the base pool npz")
    ap.add_argument("--eval-data", default=None, help="override the held-out eval set (npz or CSV)")
    ap.add_argument("--eval-ratings", default=None, help="ratings npz/CSV aligned to --eval-data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--loss-scale-budget", type=float, default=32.0,
                    help="reference budget the loss weights were tuned at; the "
                         "deduction CE (blanks), settledness, pass, and min-settle "
                         "terms are scaled by this/budget so BPTT gradient "
                         "magnitude stays comparable across budgets. Clue grading "
                         "is left at full strength (early bootstrap signal).")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    budget = args.budget or preset["budget"]
    mcfg, loss_kw, data_kw = build(preset, args)
    mcfg.refine_steps = budget
    lr_base = args.lr or preset["lr"]
    dev = args.device
    budget_scale = args.loss_scale_budget / budget

    run = args.run_name or f"{args.preset}-{datetime.now():%Y%m%d-%H%M%S}"
    log_dir = Path("logs") / run
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(run)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(log_dir / "train.log")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s  %(message)s", "%H:%M:%S")
    for hnd in (fh, sh):
        hnd.setFormatter(fmt)
        log.addHandler(hnd)
    metrics = (log_dir / "metrics.jsonl").open("a")

    def emit(row: dict) -> None:
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()

    # model, optimiser, scaler
    model = RefinementUT(mcfg).to(dev)
    use_cuda = dev.startswith("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr_base, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay, fused=use_cuda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    sched = LR(lr_base, preset["lr_schedule"], args.warmup, args.steps)

    # data
    pool_path = _resolve(args.pool or preset["pool"])
    puz, sol = load_base_pool(pool_path)
    materialize = preset.get("materialize", 0) if args.materialize is None else args.materialize
    if materialize > 0:
        t0m = time.time()
        n_base = len(puz)
        log.info("materialising %d augmentations/base from %d puzzles (transpose=%s)…",
                 materialize, n_base, data_kw["transpose"])
        puz, sol = materialize_pool(puz, sol, materialize,
                                    np.random.default_rng(args.seed),
                                    transpose=data_kw["transpose"], log=log)
        log.info("materialised %d base → %d puzzles (%.1fs)", n_base, len(puz), time.time() - t0m)
        data_kw["augment"] = False  # the set is now fixed; do not resample per draw
    batcher = Batcher(puz, sol, dev, seed=args.seed, wrong_prob=0.25, **data_kw)

    # held-out eval set (optional — skipped with a warning if absent)
    eval_path = _resolve(args.eval_data or preset["eval"])
    eval_set = None
    if Path(eval_path).exists():
        ep, es = load_pool(eval_path) if eval_path.endswith(".npz") else _load_csv(eval_path)
        rat = load_ratings(args.eval_ratings) if args.eval_ratings else None
        if rat is None and eval_path.endswith(".csv"):
            rat = load_ratings(eval_path)
        eval_set = (ep[:args.eval_n], es[:args.eval_n], None if rat is None else rat[:args.eval_n])
    else:
        log.warning("eval set %s not found — training without held-out eval "
                    "(best.pt will track latest)", eval_path)

    start = 0
    best = 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model_state"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "scaler" in ck and use_cuda:
            scaler.load_state_dict(ck["scaler"])
        if "batcher_rng" in ck:
            batcher.rng.bit_generator.state = ck["batcher_rng"]
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"].cpu())
        if use_cuda and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["cuda_rng"]])
        if "lr_state" in ck:
            sched.cycle_start = ck["lr_state"]["cycle_start"]
            sched.cycle_peak = ck["lr_state"]["cycle_peak"]
            sched.best = ck["lr_state"]["best"]
            sched.stall = ck["lr_state"]["stall"]
        start = int(ck.get("step", 0))
        best = float(ck.get("best_grid", 0.0))
        log.info("resumed %s at step %d", args.resume, start)

    cfg_json = {"preset": args.preset, "model": mcfg.to_dict(),
                "loss": loss_kw, "data": {**data_kw, "materialize": materialize},
                "train": {"budget": budget, "lr": lr_base, "batch_size": args.batch_size,
                          "steps": args.steps, "weight_decay": args.weight_decay,
                          "grad_clip": args.grad_clip, "pool": pool_path, "seed": args.seed,
                          "lr_schedule": preset["lr_schedule"],
                          "loss_scale_budget": args.loss_scale_budget, "budget_scale": budget_scale}}
    (log_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))
    log.info("run %s | device=%s | params=%d | %s", run, dev, count_params(model),
             f"d{mcfg.d_model}/h{mcfg.n_heads} budget {budget} preset {args.preset} "
             f"budget_scale {budget_scale:.3f}")

    def save(name: str) -> None:
        torch.save({"model_state": model.state_dict(), "model_config": mcfg.to_dict(),
                    "train_config": cfg_json["train"], "loss_config": loss_kw,
                    "step": step, "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best_grid": best,
                    "batcher_rng": batcher.rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if use_cuda else None,
                    "lr_state": {"cycle_start": sched.cycle_start, "cycle_peak": sched.cycle_peak,
                                 "best": sched.best, "stall": sched.stall},
                    }, log_dir / name)

    def evaluate(step: int) -> None:
        nonlocal best
        if eval_set is None:
            save("latest.pt")
            return
        ep, es, rat = eval_set
        res = solve(model, ep, es, budget, dev, batch_size=args.batch_size)
        gg = float(res["solved"].mean())
        row = {"kind": "eval", "step": step, "solved": gg,
               "finished": float(res["finished"].mean()),
               "passes": float(res["passes"].mean())}
        emit(row)
        log.info("eval  step %6d | solved %.4f | finished %.2f | passes %.2f",
                 step, gg, row["finished"], row["passes"])
        sched.on_eval(step, gg, log)
        if gg > best:
            best = gg
            save("best.pt")

    model.train()
    t0 = time.time()
    seen = 0
    for step in range(start, args.steps):
        lr = sched.at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        board, s, clues = batcher.batch(args.batch_size)
        seen += args.batch_size
        with torch.autocast("cuda" if use_cuda else "cpu", enabled=use_cuda):
            out = model(board, clues=clues, steps=budget, run_full=True)
            loss, stats = compute_loss(out, board, s, clues=clues, budget=budget,
                                       budget_scale=budget_scale, **loss_kw)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip
                                               if args.grad_clip > 0 else float("inf"))
        scaler.step(opt)
        scaler.update()

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                acc = accuracy(out, board, s, clues=clues)
            its = (step + 1 - start) / max(time.time() - t0, 1e-9)
            row = {"kind": "train", "step": step + 1, "loss": float(loss.detach()),
                   "lr": lr, "grad_norm": float(gnorm),
                   "cell": float(acc["cell_acc"]), "grid": float(acc["grid_acc"]),
                   **{k: round(float(v), 5) for k, v in stats.items()}}
            emit(row)
            log.info("step %6d | loss %.4f | cell %.3f | grid %.3f | stop %.1f | %.1f it/s",
                     step + 1, row["loss"], row["cell"], row["grid"],
                     stats["mean_stop"], its)
        if (step + 1) % args.eval_every == 0:
            evaluate(step + 1)
            model.train()
        if (step + 1) % args.ckpt_every == 0:
            save("latest.pt")

    step = args.steps
    save("latest.pt")
    evaluate(args.steps)
    log.info("done: %d steps, %d puzzles seen, best solved %.4f", args.steps, seen, best)
    metrics.close()


def _load_csv(path: str):
    """Load a Sudoku-Extreme test CSV into (puzzles, solutions) int64 (n,81)."""
    import csv
    q, s = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            q.append([0 if c == "." else int(c) for c in r["question"]])
            s.append([int(c) for c in r["answer"]])
    return np.array(q, dtype=np.int64), np.array(s, dtype=np.int64)


if __name__ == "__main__":
    main()
