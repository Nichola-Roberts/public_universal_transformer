# Universal Transformer for Sudoku — clean

A minimal, self-contained implementation of the `refine_v3` recurrent
Universal Transformer that solves sudoku, with training recipes matched as
closely as possible to the published baselines (Palm et al. RRN; HRM/TRM on
Sudoku-Extreme). Five files, no dead code:

| file | what |
|---|---|
| `model.py` | the three-stream refinement UT: relation-bias self/cross/judge attention, a settledness (halting) head, an unbounded sinusoidal pass code |
| `losses.py` | the two-term training loss (per-pass error + settledness), plus optional pass-penalty, min-settledness, and clue-grading terms |
| `data.py` | base-pool loading, sudoku symmetry augmentation, and revision-board construction |
| `train.py` | headless trainer with named presets |
| `evaluate.py` | held-out solve rate over a pass-budget sweep, by rating and by givens |

It depends only on `torch` and `numpy` (`numpy<2` against a torch built on the
numpy-1 ABI). Data pools live in `data/`.

## The architecture

One block is applied recurrently for a budget of `T` passes. Each pass reads
three streams:

- **loop** `h` — the working state, carried pass to pass, never decoded.
- **board** `X` — the current guessed grid, consulted by cross-attention and
  **rebuilt every pass** from the model's own latest guesses (clues pinned), so
  later passes reason over a partially-filled board.
- **record** `V` — the committed answer. A *judge* attention rebuilds it each
  pass, letting every cell draw from the loop's new output; the readout and the
  settledness head both read the record.

All three attentions carry the **relation bias**: a learned per-head scalar
keyed on how two cells relate — self / same row / column / box / unrelated —
i.e. the sudoku constraint graph handed to attention as an additive prior. It is
5×`n_heads` parameters and does not scale with width, so it survives shrinking
the model.

A per-cell **settledness** head predicts "is this cell's argmax correct". The
stop rule reads it: a grid halts once every non-clue cell clears
`settle_threshold`. Because the pass code is an unbounded sinusoid, **inference
can run more passes than training used** — evaluate at the trained budget for the
honest number, and one budget above to see whether a stable fixed point was
learned (a higher budget helping = yes; hurting = the model drifts past its
training horizon).

## Training

```bash
# Sudoku-Extreme, HRM/TRM's 1000-base-puzzle protocol (the default preset)
python3 train.py --preset extreme --steps 30000 --run-name extreme-clean

# Palm et al.'s 216k materialised set (uniform 17..34 givens)
python3 train.py --preset palm --steps 300000 --run-name palm-clean

# resume
python3 train.py --resume logs/extreme-clean/best.pt --steps 60000
```

### Presets

| preset | data | augmentation | revision | loss | LR |
|---|---|---|---|---|---|
| `extreme` *(default)* | 1000 base Sudoku-Extreme | **1000× materialised** (band+digit, no transpose) | off | 2-term + clue grading | cycles |
| `extreme-ours` | 1000 base Sudoku-Extreme | per-draw band+digit+**transpose** | **0.5** | 2-term + clue grading | cycles |
| `palm` | 216k materialised (17..34 givens) | none (pre-materialised) | off | 2-term + clue grading | cosine |
| `palm-ours` | 216k materialised | none | **0.5** | 2-term + clue grading | cycles |
| `palm-repro` | 216k materialised | none | off | **flat per-pass CE only** | cosine 2e-4 |

Any preset field is overridable: `--d-model --heads --d-ff --budget --lr
--revision-prob --transpose --materialize --pool --eval-data`.

**Large-model stability / bootstrap** (defaults reproduce the original recipe;
these target wider models that stall at the uniform-predictor floor or diverge to
NaN): `--amp-dtype bf16` (default; removes the fp16 softmax-overflow NaN),
`--init-width-scale 1` (scales Linear init by `sqrt(96/d_model)` so the LR
transfers across widths), `--warmup` / `--lr-half-life` (hold LR high longer so a
slow bootstrap can escape), `--grad-clip 1.0` (the deep weight-tied recurrence
wants a tight clip), and `--halt-warmup N` (ramp the halting-loss terms in over
`N` steps so they don't compete with the error gradient during the bootstrap).

The `extreme` default **materialises** a fixed set of 1000 augmentations per base
puzzle once at startup (~2 min for the full 1M; the count is `--materialize`),
exactly HRM/TRM's protocol. `extreme-ours` instead resamples a fresh symmetry
every draw. Palm's 216k pool is already a materialised set, so it is used as-is.

### Matching the baselines — and where we still differ

The `extreme`/`palm` defaults are the *faithful match*: our architecture on the
baselines' data protocol, with the features the baselines don't have turned off.
Two honest caveats remain, both documented so a reported number carries them:

1. **Augmentation matches by default; transpose and per-draw are opt-in.** The
   `extreme` default materialises 1000 augmentations per base from the same
   symmetry group HRM/TRM use (band + digit permutations, no transpose) — their
   protocol. `extreme-ours` adds transpose and resamples per draw (strictly more
   augmentation, though modest in practice — 1000 already defeats surface
   memorisation), which is the only sense in which the `-ours` data differs.
2. **Clue grading is on by default.** `refine_v3` with hard guess-feedback needs
   an input-readable gradient or the readout collapses to a constant predictor;
   grading the givens (a copy task that self-anneals to zero once learned)
   supplies it without revision. The baselines don't need this because they
   aren't built this way. `palm-repro` turns it off to reproduce Palm's exact
   flat-CE loss (and will exhibit the collapse the design is built to avoid —
   it exists as a loss baseline, not a training recipe).

Not modelled here (out of scope for these two pool-based protocols): the
random-blank **curriculum** used to train from scratch on generated puzzles. The
palm and extreme pools are fixed-difficulty, so there is nothing to promote
through.

### Baseline numbers to beat

| | task | reference |
|---|---|---|
| Palm et al. (RRN) | 17-given, 32 / 64 steps | 94.1% / 96.6% |
| HRM | Sudoku-Extreme, 1000 base | 55.0% |
| TRM | Sudoku-Extreme, 1000 base | 87.4% |

## Evaluation

```bash
# budget sweep on the Palm held-out 17-clue set
python3 evaluate.py --ckpt logs/palm-clean/best.pt \
    --data data/eval-17clue-64.npz --budgets 32,64,128

# Sudoku-Extreme test split, by rating and by givens
python3 evaluate.py --ckpt logs/extreme-clean/best.pt \
    --data data/sudoku-extreme-test.csv \
    --ratings data/sudoku-extreme-test.csv --budgets 32,64
```

`solved` is the fraction of grids with every blank correct; `finished` is the
fraction that halted on their own within the budget; `passes` is the mean number
actually run. The Sudoku-Extreme test CSV is the standard release (not shipped
here); the Palm and mixed eval npz files are in `data/`.

## Provenance

Distilled from a private research codebase into a self-contained package: the
`refine_v3` architecture and its two-term loss are ported verbatim — verified
numerically identical, bit-for-bit, against the original forward pass and loss.
The clean package drops ACT, the plain-UT arch, the unused linear readout head,
the soft/ste guess-feedback variants, the v3/v4/v5/summed loss families and their
un-ablated terms, the from-scratch curriculum machinery, and the research
scripts.

The Sudoku-Extreme test split (`data/sudoku-extreme-test.csv`) used by the
by-rating evaluation is the standard public release and is not bundled here.
Same for the `extreme-full` preset's training pool (`data/extreme-full-train.npz`,
3.8M puzzles) — both are gitignored (too large for git) and must be rebuilt from
[`sapientinc/sudoku-extreme`](https://huggingface.co/datasets/sapientinc/sudoku-extreme)
on Hugging Face before an `extreme-full` run:

```bash
curl -L -o /tmp/train.csv https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/train.csv
curl -L -o data/sudoku-extreme-test.csv https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv
python3 -c "
import csv, numpy as np
puzzles, solutions = [], []
with open('/tmp/train.csv') as f:
    for r in csv.DictReader(f):
        puzzles.append([0 if c == '.' else int(c) for c in r['question']])
        solutions.append([int(c) for c in r['answer']])
np.savez('data/extreme-full-train.npz',
         puzzles=np.array(puzzles, dtype=np.int8),
         solutions=np.array(solutions, dtype=np.int8))
"
```

On an ephemeral pod (see `CLAUDE.md`) this is lost whenever the box is recycled
and needs re-running before the next `extreme-full` launch — it isn't backed up
by pushes to the repo the way `logs/` is.
