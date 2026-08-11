"""Held-out evaluation: solve rate at a pass budget, optionally by difficulty.

Everything a baseline reports is here:
  * **solved** — fraction of grids whose every blank is filled correctly.
  * **finished** — fraction that met the stop rule within the budget (halted on
    their own rather than running out of passes).
  * **mean passes** — average passes actually run before the answer was taken.

The model is run at a fixed pass ``budget``; because the pass code is an unbounded
sinusoid the budget may exceed the trained ``refine_steps``. Report a checkpoint at
its trained budget for the honest number, and one budget above to read whether it
has learned a stable fixed point (a higher budget helping = yes).

    python3 evaluate.py --ckpt run/best.pt --data data/eval-17clue-64.npz \
        --budgets 32,64
    python3 evaluate.py --ckpt run/best.pt --data data/eval-17clue-64.npz \
        --budgets 32,64 --ratings data/sudoku-extreme-test.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from model import RefinementUT, ModelConfig, target_mask


def load_pool(path: str) -> tuple[np.ndarray, np.ndarray]:
    if path.endswith(".csv"):
        q, s = [], []
        with open(path) as f:
            for r in csv.DictReader(f):
                q.append([0 if c == "." else int(c) for c in r["question"]])
                s.append([int(c) for c in r["answer"]])
        return np.array(q, dtype=np.int64), np.array(s, dtype=np.int64)
    d = np.load(path)
    return d["puzzles"].astype(np.int64), d["solutions"].astype(np.int64)


def load_ratings(path: str) -> np.ndarray:
    """Ratings aligned to the pool order. Accepts an npz with a `ratings` key or
    the Sudoku-Extreme CSV (column `rating`)."""
    if path.endswith(".npz"):
        return np.load(path)["ratings"].astype(np.float64)
    with open(path) as f:
        return np.array([float(r["rating"]) for r in csv.DictReader(f)], dtype=np.float64)


RATING_BANDS = [(0, 0), (1, 10), (11, 25), (26, 50), (51, 100), (101, 10 ** 9)]
GIVEN_BANDS = [(17, 21), (22, 24), (25, 27), (28, 30), (31, 36)]


@torch.no_grad()
def solve(model: RefinementUT, puzzles: np.ndarray, solutions: np.ndarray,
          budget: int, device: str, batch_size: int = 512) -> dict:
    """Run the model at ``budget`` passes and return per-grid arrays."""
    model.eval()
    n = len(puzzles)
    solved = np.zeros(n, dtype=bool)
    finished = np.zeros(n, dtype=bool)
    passes = np.zeros(n, dtype=np.float64)
    cell_acc = np.zeros(n, dtype=np.float64)
    for i in range(0, n, batch_size):
        x = torch.from_numpy(puzzles[i:i + batch_size]).to(device)
        sol = torch.from_numpy(solutions[i:i + batch_size]).to(device)
        clues = (x != 0)
        out = model(x, steps=budget, readouts=False, clues=clues)
        pred = out["logits"].argmax(-1) + 1
        mine = target_mask(x, clues).bool()
        hit = (pred == sol) | ~mine
        solved[i:i + batch_size] = hit.all(dim=1).cpu().numpy()
        finished[i:i + batch_size] = out["finished"].cpu().numpy()
        passes[i:i + batch_size] = (out["stop_pass"].float() + 1).cpu().numpy()
        cell_acc[i:i + batch_size] = ((hit & mine).sum(dim=1).float()
                                      / mine.sum(dim=1).clamp(min=1).float()).cpu().numpy()
    return {"solved": solved, "finished": finished, "passes": passes, "cell_acc": cell_acc}


def _band_table(name: str, bands, key, solved_by_budget: dict, acc_by_budget: dict,
                passes_by_budget: dict, budgets: list[int]) -> str:
    lines = [f"\n  by {name}"]
    head = "  " + f"{'band':>10}" + f"{'n':>7}"
    head += "".join(f"{'b'+str(b)+' solved':>12}{'b'+str(b)+' acc':>10}{'b'+str(b)+' passes':>12}"
                    for b in budgets)
    lines.append(head)
    for lo, hi in bands:
        sel = (key >= lo) & (key <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        label = f"{lo}-{hi}" if hi < 10 ** 8 else f"{lo}+"
        row = f"  {label:>10}{n:>7}"
        for b in budgets:
            row += (f"{solved_by_budget[b][sel].mean():>12.3f}"
                    f"{acc_by_budget[b][sel].mean():>10.3f}"
                    f"{passes_by_budget[b][sel].mean():>12.2f}")
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="npz with {puzzles, solutions}")
    ap.add_argument("--budgets", default="32,64")
    ap.add_argument("--ratings", default=None,
                    help="npz(ratings) or Sudoku-Extreme test CSV, aligned to --data")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N puzzles (default: all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location=a.device)
    model = RefinementUT(ModelConfig(**ck["model_config"])).to(a.device)
    model.load_state_dict(ck["model_state"])

    puzzles, solutions = load_pool(a.data)
    if a.limit:
        puzzles, solutions = puzzles[:a.limit], solutions[:a.limit]
    givens = 81 - (puzzles == 0).sum(1)
    ratings = load_ratings(a.ratings) if a.ratings else None
    if ratings is not None and a.limit:
        ratings = ratings[:a.limit]
    budgets = [int(b) for b in a.budgets.split(",")]

    print(f"{a.ckpt}  step {ck.get('step', '?')}  |  {len(puzzles)} puzzles from {a.data}")
    solved_by_budget: dict[int, np.ndarray] = {}
    acc_by_budget: dict[int, np.ndarray] = {}
    passes_by_budget: dict[int, np.ndarray] = {}
    print(f"\n  {'budget':>7}{'solved':>9}{'cell_acc':>10}{'finished':>10}{'passes':>9}")
    for b in budgets:
        res = solve(model, puzzles, solutions, b, a.device, a.batch_size)
        solved_by_budget[b] = res["solved"]
        acc_by_budget[b] = res["cell_acc"]
        passes_by_budget[b] = res["passes"]
        print(f"  {b:>7}{res['solved'].mean():>9.4f}{res['cell_acc'].mean():>10.4f}"
              f"{res['finished'].mean():>10.2f}{res['passes'].mean():>9.2f}")

    if ratings is not None:
        print(_band_table("rating (backtracking depth)", RATING_BANDS, ratings,
                          solved_by_budget, acc_by_budget, passes_by_budget, budgets))
    print(_band_table("givens", GIVEN_BANDS, givens, solved_by_budget, acc_by_budget,
                      passes_by_budget, budgets))


if __name__ == "__main__":
    main()
