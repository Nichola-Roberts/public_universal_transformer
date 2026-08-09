"""A Universal Transformer for sudoku — the clean, minimal arch.

This is the ``refine_v3`` three-stream Universal Transformer, extracted from the
research code with every dead branch removed. One recurrent block is applied for
a budget of ``T`` passes; each pass reads three streams:

  * **loop** ``h``   — the working state, carried pass to pass, never decoded.
  * **board** ``X``  — the current guessed grid, re-embedded and consulted by
    cross-attention. Rebuilt every pass from the model's own latest guesses
    (clues pinned), so later passes reason over a partially-filled board.
  * **record** ``V`` — the committed answer. Rebuilt each pass by a *judge*
    attention that lets each cell draw from the loop's new output; the readout
    and the settledness head both read the record.

All three attentions carry the sudoku **relation bias**: a learned per-head
scalar keyed on how two cells relate (self / same row / col / box / unrelated),
i.e. the constraint graph handed to attention as an additive prior.

A per-cell **settledness** head on the record predicts "is this cell's argmax
correct". The stop rule reads it: a grid halts once every non-clue cell is over
``settle_threshold``. Because the pass code is an unbounded sinusoid, inference
may run more passes than training used.

Simplifications vs the research code, all behaviour-preserving for the trained
recipe: only the FFN readout head (the linear head was never active), only
``hard`` guess feedback (argmax→embedding; the differentiable soft/ste variants
were unused), no ACT, no plain-UT arch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CELLS = 81
N_DIGITS = 9
VOCAB = 10       # 0 = blank, 1..9 = digit
N_RELATIONS = 5  # self, same-row, same-col, same-box, unrelated

# what a cell *is*, distinct from what digit it holds
KIND_BLANK = 0   # still empty
KIND_CLUE = 1    # part of the puzzle — fixed, never revised
KIND_GUESS = 2   # the model's own earlier answer — revisable
N_KINDS = 3


@dataclass
class ModelConfig:
    d_model: int = 96
    n_heads: int = 4
    d_ff: int = 384
    dropout: float = 0.05
    refine_steps: int = 32        # pass budget T used at train time
    settle_threshold: float = 0.995
    halt_bias_init: float = -3.0  # born reluctant to call a cell settled
    rel_bias: bool = True         # constraint-graph attention bias
    structured_pos: bool = True   # add row/col/box embeddings
    kind_embed: bool = True       # tell clues apart from the model's own guesses
    settle_feedback: bool = True  # feed per-cell settledness into the board

    def to_dict(self) -> dict:
        return asdict(self)


def _relation_matrix() -> torch.Tensor:
    """(81, 81) int tensor: how cell i relates to cell j.

    Precedence self > row > col > box, so a pair sharing a row *and* a box is
    labelled row: the box label only tags the 4 peers a box adds beyond its row
    and column. Per cell: 8 row peers, 8 col peers, 4 box-only, self, 60 unrelated.
    """
    idx = torch.arange(N_CELLS)
    r, c = idx // 9, idx % 9
    b = (r // 3) * 3 + (c // 3)
    same_r = r[:, None] == r[None, :]
    same_c = c[:, None] == c[None, :]
    same_b = b[:, None] == b[None, :]
    eye = torch.eye(N_CELLS, dtype=torch.bool)
    rel = torch.full((N_CELLS, N_CELLS), 4, dtype=torch.long)
    rel[same_b] = 3
    rel[same_c] = 2
    rel[same_r] = 1
    rel[eye] = 0
    return rel


def _cell_kinds(x: torch.Tensor, clues: torch.Tensor | None) -> torch.Tensor:
    """Blank / clue / guess for every cell. Without a clue mask every filled
    cell is a clue (a plain one-shot forward)."""
    filled = x != 0
    if clues is None:
        return filled.long()  # 0 = blank, 1 = clue
    return filled.long() + (filled & ~clues.bool()).long()


class Attention(nn.Module):
    """Multi-head self-attention with the additive relation bias."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        assert self.dh * cfg.n_heads == cfg.d_model, "d_model must divide by n_heads"
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout
        self.rel_bias = cfg.rel_bias
        if cfg.rel_bias:
            self.register_buffer("rel", _relation_matrix(), persistent=False)
            self.rel_emb = nn.Embedding(N_RELATIONS, cfg.n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        bias = None
        if self.rel_bias:
            bias = self.rel_emb(self.rel).permute(2, 0, 1).unsqueeze(0).to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class CrossAttention(nn.Module):
    """Cross-attention with the relation bias. Queries from the consulting
    stream; keys/values from the board being consulted. ``project_kv`` projects
    a board once so it can be reused across passes."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        assert self.dh * cfg.n_heads == cfg.d_model, "d_model must divide by n_heads"
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout
        self.rel_bias = cfg.rel_bias
        if cfg.rel_bias:
            self.register_buffer("rel", _relation_matrix(), persistent=False)
            self.rel_emb = nn.Embedding(N_RELATIONS, cfg.n_heads)

    def project_kv(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = src.shape
        k, v = self.kv(src).view(B, L, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, xq: torch.Tensor,
                kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, L, D = xq.shape
        k, v = kv
        q = self.q(xq).view(B, L, self.h, self.dh).transpose(1, 2)
        bias = None
        if self.rel_bias:
            bias = self.rel_emb(self.rel).permute(2, 0, 1).unsqueeze(0).to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class JudgeAttention(nn.Module):
    """The commit gate. The old record provides the queries, the loop's new
    output the keys and values; the attention output becomes the new record.
    The old record is never offered as content — it shapes the new record only
    through its queries. A learned ``accept_bias`` (+12) on the diagonal, with
    value/output init to the identity, makes the first commits ≈ the loop's
    output until training learns to keep older content."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout
        self.rel_bias = cfg.rel_bias
        if cfg.rel_bias:
            self.register_buffer("rel", _relation_matrix(), persistent=False)
            self.rel_emb = nn.Embedding(N_RELATIONS, cfg.n_heads)
        self.register_buffer("eye", torch.eye(N_CELLS), persistent=False)
        self.accept_bias = nn.Parameter(torch.tensor(12.0))

    def forward(self, vq: torch.Tensor, h_new: torch.Tensor) -> torch.Tensor:
        B, L, D = vq.shape
        q = self.q(vq).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(h_new).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(h_new).view(B, L, self.h, self.dh).transpose(1, 2)
        if self.rel_bias:
            bias = self.rel_emb(self.rel).permute(2, 0, 1)  # (H, L, L)
        else:
            bias = vq.new_zeros(self.h, L, L)
        bias = (bias + self.accept_bias * self.eye).unsqueeze(0).to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class RefineBlock(nn.Module):
    """Pre-LN block: self-attention, then a cross-attention read of the board,
    then feed-forward. Weight-tied across passes."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln_x = nn.LayerNorm(cfg.d_model)
        self.cross = CrossAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor,
                x_kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h = h + self.drop(self.attn(self.ln1(h)))
        h = h + self.drop(self.cross(self.ln_x(h), kv=x_kv))
        h = h + self.drop(self.ff(self.ln2(h)))
        return h


def target_mask(puzzles: torch.Tensor, clues: torch.Tensor | None) -> torch.Tensor:
    """Cells the model is responsible for: everything that is not a clue. With a
    clue mask that also covers the model's own guesses (which teaches revision)."""
    if clues is None:
        return (puzzles == 0).float()
    return (~clues.bool()).float()


class RefinementUT(nn.Module):
    """The three-stream refinement Universal Transformer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # input / board embedding
        self.tok = nn.Embedding(VOCAB, cfg.d_model)
        self.pos = nn.Embedding(N_CELLS, cfg.d_model)
        if cfg.kind_embed:
            self.kind = nn.Embedding(N_KINDS, cfg.d_model)
        if cfg.structured_pos:
            self.row_emb = nn.Embedding(9, cfg.d_model)
            self.col_emb = nn.Embedding(9, cfg.d_model)
            self.box_emb = nn.Embedding(9, cfg.d_model)

        # unbounded sinusoidal pass code (any t valid → inference can overrun T)
        half = cfg.d_model // 2
        inv_freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half, 1)
        )
        self.register_buffer("time_inv_freq", inv_freq, persistent=False)
        self.time_scale = nn.Parameter(torch.tensor(0.03))

        # settledness fed back into the board (continuous → a projection, zero-init)
        if cfg.settle_feedback:
            self.settle_emb = nn.Linear(1, cfg.d_model)
            nn.init.zeros_(self.settle_emb.weight)
            nn.init.zeros_(self.settle_emb.bias)
        else:
            self.settle_emb = None

        self.block = RefineBlock(cfg)

        # record stream: judge (commit), receipt (record→loop channel), settle head
        self.ln_v = nn.LayerNorm(cfg.d_model)
        self.judge = JudgeAttention(cfg)
        self.ln_r = nn.LayerNorm(cfg.d_model)
        self.receipt = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_r = nn.Linear(cfg.d_model, cfg.d_model)  # injects r into the loop
        self.halt_ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        # readout (FFN head) on the record
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, N_DIGITS),
        )

        idx = torch.arange(N_CELLS)
        self.register_buffer("cell_idx", idx, persistent=False)
        self.register_buffer("row_idx", idx // 9, persistent=False)
        self.register_buffer("col_idx", idx % 9, persistent=False)
        self.register_buffer("box_idx", (idx // 9 // 3) * 3 + (idx % 9) // 3, persistent=False)

        self.apply(self._init)
        # deliberate inits — do not "fix" these:
        nn.init.zeros_(self.w_r.weight)          # receipt channel starts silent
        nn.init.zeros_(self.w_r.bias)
        nn.init.eye_(self.judge.v.weight)        # judge starts as pure acceptance
        nn.init.eye_(self.judge.proj.weight)
        nn.init.constant_(self.halt_ffn[-1].bias, cfg.halt_bias_init)
        if cfg.rel_bias:
            for emb in (self.block.attn.rel_emb, self.block.cross.rel_emb,
                        self.judge.rel_emb):
                nn.init.zeros_(emb.weight)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # -- embedding helpers ------------------------------------------------
    def _embed(self, x: torch.Tensor, clues: torch.Tensor | None) -> torch.Tensor:
        h = self.tok(x.long()) * math.sqrt(self.cfg.d_model) + self.pos(self.cell_idx)[None]
        if self.cfg.structured_pos:
            h = h + (self.row_emb(self.row_idx) + self.col_emb(self.col_idx)
                     + self.box_emb(self.box_idx))[None]
        if self.cfg.kind_embed:
            h = h + self.kind(_cell_kinds(x, clues))
        return h

    def _board_embed(self, board: torch.Tensor, w: torch.Tensor,
                     clue_mask: torch.Tensor, clues: torch.Tensor | None) -> torch.Tensor:
        """The board embedding plus a per-cell settledness term (clues pinned to
        full confidence). ``w`` is detached upstream: a conditioning signal, not
        a gradient path into the halt head."""
        emb = self._embed(board, clues)
        if self.settle_emb is not None:
            w_in = torch.where(clue_mask, torch.ones_like(w), w).unsqueeze(-1)
            emb = emb + self.settle_emb(w_in.to(emb.dtype))
        return emb

    def _board_kv(self, x: torch.Tensor, logits: torch.Tensor, w: torch.Tensor,
                  clue_mask: torch.Tensor, clues: torch.Tensor | None):
        """Fold the current guesses into the board and re-project the cross K/V.
        Clues keep their given digit; every other cell takes its latest argmax
        guess (hard feedback)."""
        guess = logits.argmax(-1) + 1                    # (B, 81) in 1..9
        board = torch.where(clue_mask, x.long(), guess)  # clues pinned
        return self.block.cross.project_kv(
            self._board_embed(board, w, clue_mask, clues))

    def _time(self, t: int, device) -> torch.Tensor:
        ang = float(t) * self.time_inv_freq.to(device)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)])
        if emb.numel() < self.cfg.d_model:  # odd d_model: pad the last slot
            emb = torch.cat([emb, emb.new_zeros(self.cfg.d_model - emb.numel())])
        return self.time_scale * emb

    def _readout(self, v: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_out(v))

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor, steps: int | None = None, readouts: bool = True,
                clues: torch.Tensor | None = None,
                force_stop: torch.Tensor | None = None,
                run_full: bool = False) -> dict:
        """x: (B, 81) int of 0..9. ``clues`` marks the given cells; the rest are
        the model's revisable guesses.

        Returns:
          logits      (B, 81, 9)  the answer, snapshotted at each grid's stop pass
          step_logits list of (B, 81, 9), one per executed pass
          halt_logits list of (B, 81) settledness logits per executed pass
          stop_pass   (B,) long  the pass each grid's answer comes from
          finished    (B,) bool  met the stop rule within the budget
          ponder / n_updates  (B, 81) passes actually run

        ``force_stop`` (B,) long: a pass index to force-stop a grid (training
        audit; -1 = normal rule). ``run_full`` keeps looping to the full budget
        even after every grid has stopped, so the per-pass error term keeps
        getting supervision (answers are still snapshotted at each grid's stop).
        """
        T = steps or self.cfg.refine_steps
        B = x.shape[0]
        dev = x.device
        thr = self.cfg.settle_threshold
        mb = target_mask(x, clues).bool()                # clue cells: no vote in the stop rule
        clue_mask = clues.bool() if clues is not None else (x != 0)

        X = self._embed(x, clues)                        # loop init (no settle term)
        w0 = clue_mask.float()                           # pass 0 board: clues settled, blanks not
        x_kv = self.block.cross.project_kv(self._board_embed(x, w0, clue_mask, clues))
        h = X
        V = torch.zeros_like(X)                          # record starts empty
        r = torch.zeros_like(X)

        done = torch.zeros(B, dtype=torch.bool, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        stop_pass = torch.zeros(B, dtype=torch.long, device=dev)
        final_logits = x.new_zeros((B, N_CELLS, N_DIGITS), dtype=torch.float32)
        halt_logits: list[torch.Tensor] = []
        step_logits: list[torch.Tensor] = []

        for t in range(T):
            h = self.block(h + self._time(t, dev) + self.w_r(r), x_kv)
            V = self.judge(self.ln_v(V), h)              # commit: rebuild the record

            r = self.receipt(self.ln_r(V))
            ell = self.halt_ffn(r).squeeze(-1)
            halt_logits.append(ell)
            w = torch.sigmoid(ell.float())
            settled = ((w >= thr) | ~mb).all(-1)
            stop_now = (~done) & settled & (t >= 1)      # minimum two passes
            if force_stop is not None:
                stop_now = stop_now | ((~done) & (force_stop == t))
            last = t == T - 1
            take = stop_now | (last & ~done)
            finished = finished | stop_now
            stop_pass = torch.where(take, torch.full_like(stop_pass, t), stop_pass)
            done = done | take

            exiting = last or (bool(done.all()) and not run_full)

            logits_t = self._readout(V)                  # every pass: supervise + become next board
            if readouts or exiting:
                step_logits.append(logits_t)
            if bool(take.any()):
                final_logits = torch.where(
                    take.view(B, 1, 1), logits_t.float(), final_logits)
            if exiting:
                break

            # fold guesses back into the board, tagged with settledness (detached)
            x_kv = self._board_kv(x, logits_t, w.detach(), clue_mask, clues)

        passes = (stop_pass + 1).float()
        return {
            "logits": final_logits.to(step_logits[-1].dtype),
            "step_logits": step_logits,
            "ponder": passes[:, None].expand(-1, N_CELLS),
            "n_updates": passes[:, None].expand(-1, N_CELLS),
            "halt_logits": halt_logits,
            "stop_pass": stop_pass,
            "finished": finished,
        }


@torch.no_grad()
def accuracy(out: dict, puzzles: torch.Tensor, solutions: torch.Tensor,
             clues: torch.Tensor | None = None) -> dict:
    """Cell accuracy over the responsible cells, and the fraction of grids
    solved completely."""
    pred = out["logits"].argmax(-1) + 1
    mine = target_mask(puzzles, clues).bool()
    filled = torch.where(mine, pred, puzzles.long())
    hit = (filled == solutions.long()) | ~mine
    cell_acc = (hit & mine).sum().float() / mine.sum().clamp(min=1).float()
    grid_acc = hit.all(dim=1).float().mean()
    return {"cell_acc": cell_acc, "grid_acc": grid_acc}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
