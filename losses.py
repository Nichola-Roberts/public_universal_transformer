"""The training loss — the two-term floor, plus two optional terms.

Extracted from the research code's ``basic`` loss (the "v4 line": a loss rebuilt
from a two-term floor after v3/v4/v5 each grew by stacking terms nothing had
ablated). The one-at-a-time experimental terms that were never adopted
(wrong-cell integral, worst-cell soft-max, final-pass anchors) are dropped.

Two core terms, both over the cells the model is responsible for (non-clue),
meaned over every pass the forward ran:

  1. **error**       — cross-entropy of each pass's readout against the solution.
  2. **settledness** — BCE of the settledness head against whether that cell's
     argmax is currently correct (label detached). This *defines* ``w`` as "am I
     right", which is exactly what the stop rule reads, so it is the term that
     makes halting mean something.

Two optional terms (0 = off):

  * **pass penalty** ``pass_weight`` — ``sum(1 - w)`` over the passes each grid
    ran, normalised by cells and budget: pays for stopping sooner.
  * **min settledness** ``min_settle_weight`` — ``-log`` of the least-settled
    non-clue cell, per pass: drives the cell that gates the stop up into the tail.

Optional **clue grading** (``clue_share`` > 0): the clue cells carry that share
of each grid's error weight *as a group* (a copy task). With it off, nothing in
the loss can be reduced by reading the input — every graded cell needs a digit
deduced — which is why a non-revision run needs it (or revision) to avoid the
constant-predictor collapse. ``clue_denom_blanks`` normalises by the blank count
so the clue term is additive and self-annealing (vanishes once copying is
learned) rather than a permanent rescaling of the blank gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model import N_CELLS, N_DIGITS, target_mask


def compute_loss(
    out: dict,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    clues: torch.Tensor | None = None,
    budget: int = 32,
    settle_weight: float = 1.0,
    pass_weight: float = 0.0,
    min_settle_weight: float = 0.0,
    grade_clues: bool = False,
    clue_share: float = 0.0,
    clue_denom_blanks: bool = False,
) -> tuple[torch.Tensor, dict]:
    target = (solutions.long() - 1).clamp(min=0)          # 0..8
    mask = target_mask(puzzles, clues)                    # (B, 81) non-clue cells
    grid_denom = mask.sum(1).clamp(min=1.0)               # (B,)

    # optional clue grading: the clue group carries `clue_share` of the error
    # weight; the per-cell weight that delivers it caps at 1 (uniform grading).
    if grade_clues and clue_share > 0:
        nb = grid_denom[:, None]                          # blanks per grid
        nc = (float(N_CELLS) - nb).clamp(min=1.0)         # clues per grid
        s_ = min(float(clue_share), 1.0)
        cw = (s_ * nb / (max(1.0 - s_, 1e-6) * nc)).clamp(max=1.0)
    else:
        cw = torch.zeros_like(mask[:, :1])
    err_w = mask + cw * (1.0 - mask)                      # blanks 1, clues cw
    err_denom = (grid_denom if clue_denom_blanks else err_w.sum(1)).clamp(min=1.0)

    logits = torch.stack(out["step_logits"], dim=1)       # (B, P, 81, 9)
    halts = torch.stack(out["halt_logits"], dim=1)        # (B, P, 81)
    B, P = logits.shape[0], logits.shape[1]
    dev = logits.device

    # 1. error — flat per-pass CE on the logits
    ce = F.cross_entropy(
        logits.reshape(-1, N_DIGITS).float(),
        target[:, None, :].expand(-1, P, -1).reshape(-1),
        reduction="none",
    ).view(B, P, N_CELLS)
    l_err = ((ce * err_w[:, None]).mean(1).sum(1) / err_denom).mean()

    # 2. settledness — w predicts whether this cell's argmax is right
    correct = (logits.detach().argmax(-1) + 1 == solutions[:, None, :]).float()
    bce = F.binary_cross_entropy_with_logits(halts.float(), correct, reduction="none")
    l_settle = ((bce * mask[:, None]).sum((1, 2)) / (grid_denom * P)).mean()

    loss = l_err + settle_weight * l_settle

    # 3. optional pass penalty — sum(1 - w) over the passes each grid ran
    w = torch.sigmoid(halts.float())
    idx = torch.arange(P, device=dev)
    ran = (idx[None] <= out["stop_pass"][:, None]).float()            # (B, P)
    l_pass = ((((1.0 - w) * mask[:, None]).sum(-1) * ran).sum(1)
              / (grid_denom * max(budget, 1))).mean()
    if pass_weight > 0:
        loss = loss + pass_weight * l_pass

    # 4. optional min settledness — -log of the least-settled non-clue cell
    l_min = logits.new_zeros(())
    if min_settle_weight > 0:
        w_min = w.masked_fill(~mask[:, None].bool(), 1.0).amin(dim=2)  # (B, P)
        l_min = (-torch.log(w_min.clamp(min=1e-6))).mean()
        loss = loss + min_settle_weight * l_min

    stats = {
        "l_err": l_err.detach(),
        "l_settle": l_settle.detach(),
        "l_pass": l_pass.detach(),
        "l_min": l_min.detach(),
        "mean_stop": (out["stop_pass"].float() + 1).mean().detach(),
        "finished_frac": out["finished"].float().mean().detach(),
        "clue_share": min(float(clue_share), 1.0) if grade_clues else 0.0,
    }
    return loss, stats
