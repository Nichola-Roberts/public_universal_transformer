"""Data pipeline: base pools, symmetry augmentation, and revision boards.

A *base pool* is an npz of ``{puzzles, solutions}`` as ``(n, 81)`` int, 0 = blank,
1..9 = digit — each puzzle carries its own blanks (these are real puzzles, not a
solved grid to be masked). Training draws from the pool with replacement and,
optionally, applies a sudoku-preserving symmetry per draw.

Augmentation (``augment_pair``) is the group of transforms that leave a sudoku a
sudoku: digit relabelling, permuting the three bands and the three rows within
each band (and likewise columns), and optionally transposing. HRM/TRM use the
band+digit group without transpose; ``transpose=False`` matches them. Blanks are
preserved — a blank cell stays blank under every transform.

Revision (``add_guesses``) fabricates the self-revision task: some blanks are
pre-filled with guesses, a fraction of them deliberately wrong, and the model is
trained to overwrite the wrong ones. Clues are never touched. The baselines feed
no prediction back as input, so ``revision_prob=0`` matches them.
"""

from __future__ import annotations

import numpy as np
import torch

N = 9
CELLS = 81


def load_base_pool(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["puzzles"].astype(np.int8), d["solutions"].astype(np.int8)


def _band_order(rng: np.random.Generator) -> np.ndarray:
    """Permute the three bands, and the three lines within each band."""
    order = []
    for b in rng.permutation(3):
        order.extend(b * 3 + rng.permutation(3))
    return np.asarray(order)


def augment_pair(puzzle: np.ndarray, solution: np.ndarray, rng: np.random.Generator,
                 transpose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A random sudoku symmetry applied to a puzzle+solution together, blanks
    preserved. Relabel uses one permutation for both grids; band/line orders are
    shared; transpose (if drawn) applies to both."""
    p = np.asarray(puzzle).reshape(N, N)
    s = np.asarray(solution).reshape(N, N)

    perm = rng.permutation(9).astype(np.int8) + 1
    s = perm[s - 1]
    p = np.where(p > 0, perm[np.clip(p.astype(np.int16) - 1, 0, 8)], 0).astype(np.int8)

    rows, cols = _band_order(rng), _band_order(rng)
    p, s = p[rows, :][:, cols], s[rows, :][:, cols]
    if transpose and rng.random() < 0.5:
        p, s = p.T, s.T
    return np.ascontiguousarray(p), np.ascontiguousarray(s)


def add_guesses(puzzles: np.ndarray, solutions: np.ndarray, rng: np.random.Generator,
                wrong_prob: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Build revision boards: fill a per-grid random fraction of the blanks with
    guesses, each wrong with probability ``wrong_prob`` (a wrong digit is any
    other digit). Returns (boards, clue_mask); clues are the original givens and
    are never filled."""
    B, n = puzzles.shape
    clue_mask = puzzles != 0
    boards = puzzles.copy()

    blanks = ~clue_mask
    frac = rng.uniform(0.0, 1.0, size=(B, 1))
    take = blanks & (rng.random((B, n)) < frac)

    offset = rng.integers(1, 9, size=(B, n))                     # 1..8 → never the true digit
    wrong = ((solutions - 1 + offset) % 9 + 1).astype(np.int8)
    is_wrong = rng.random((B, n)) < wrong_prob
    guesses = np.where(is_wrong, wrong, solutions)

    boards[take] = guesses[take]
    return boards.astype(np.int8), clue_mask


def materialize_pool(puzzles: np.ndarray, solutions: np.ndarray, factor: int,
                     rng: np.random.Generator, transpose: bool = False,
                     log=None) -> tuple[np.ndarray, np.ndarray]:
    """Expand a base pool into a *fixed* set of ``factor`` augmentations per base
    puzzle, built once. This is HRM/TRM's protocol: 1000 base puzzles × ~1000
    materialised augmentations, iterated over during training — as opposed to
    resampling a fresh symmetry every draw (see ``Batcher(augment=True)``)."""
    n = len(puzzles)
    out_p = np.empty((n * factor, CELLS), dtype=np.int8)
    out_s = np.empty((n * factor, CELLS), dtype=np.int8)
    k = 0
    for i in range(n):
        for _ in range(factor):
            p, s = augment_pair(puzzles[i], solutions[i], rng, transpose=transpose)
            out_p[k] = p.reshape(CELLS)
            out_s[k] = s.reshape(CELLS)
            k += 1
        if log is not None and (i + 1) % 200 == 0:
            log.info("  materialising %d/%d base puzzles", i + 1, n)
    return out_p, out_s


class Batcher:
    """Draws training batches from a base pool. One RNG, seeded and reproducible."""

    def __init__(self, puzzles: np.ndarray, solutions: np.ndarray, device: str,
                 seed: int = 0, augment: bool = True, transpose: bool = False,
                 revision_prob: float = 0.0, wrong_prob: float = 0.25):
        self.puzzles = puzzles
        self.solutions = solutions
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.transpose = transpose
        self.revision_prob = revision_prob
        self.wrong_prob = wrong_prob

    def _draw(self, bs: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, len(self.puzzles), size=bs)
        if self.augment:
            pairs = [augment_pair(self.puzzles[i], self.solutions[i], self.rng,
                                  transpose=self.transpose) for i in idx]
            puz = np.stack([p.reshape(CELLS) for p, _ in pairs])
            sol = np.stack([s.reshape(CELLS) for _, s in pairs])
        else:
            puz = self.puzzles[idx].reshape(bs, CELLS)
            sol = self.solutions[idx].reshape(bs, CELLS)
        return puz.astype(np.int8), sol.astype(np.int8)

    def batch(self, bs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (board, sol, clues): board/sol long (B,81), clues bool (B,81).
        With revision, the first round(bs*revision_prob) rows carry guesses; the
        rest are clean puzzles."""
        puz, sol = self._draw(bs)
        if self.revision_prob > 0:
            n_rev = int(round(bs * self.revision_prob))
            board, clues = add_guesses(puz[:n_rev], sol[:n_rev], self.rng,
                                       wrong_prob=self.wrong_prob)
            boards = np.concatenate([board, puz[n_rev:]])
            clue_mask = np.concatenate([clues, puz[n_rev:] != 0])
        else:
            boards, clue_mask = puz, puz != 0

        b = torch.from_numpy(boards).long().to(self.device)
        s = torch.from_numpy(sol).long().to(self.device)
        c = torch.from_numpy(clue_mask).to(self.device)
        return b, s, c
