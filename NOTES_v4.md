# v4 notes — rebuilding the loss from the floor up

The engineering log for the v4 line of work. `NOTES.md` holds everything up to
and including v5 of the loss; this file starts from the observation that the
whole v3→v4→v5 sequence was never measured against a *simple* loss, and works
forward from that floor. Same rules as NOTES.md: what was tried, what broke, and
what the numbers actually were.

## The `basic` loss: the two-term floor nobody measured — 2026-08-07

v3, v4 and v5 each grew by adding a term to the one before it, so there has
never been a run to compare them *against*. Every claim of the form "the ramp
buys X" or "the shares hold the balance" is measured against another loss that
already has five terms. `compute_basic_loss` is the floor those numbers were
missing: **one error term on the logits, one settledness term on the settledness
head**, fixed weights, nothing else.

    l_err     flat per-pass CE over non-clue cells, mean over passes and cells
    l_settle  BCE of the settledness head against "is this cell's argmax right",
              same cells, every pass
    loss      l_err + settle_weight * l_settle  [+ pass_weight * l_pass]

What is deliberately *absent*, each one a claim a richer loss makes:

- **No share rescaling.** Fixed weights. If the balance drifts as error's own
  scale swings across the curriculum — v4's stated reason for the share
  machinery — that drift is the finding, not a bug to patch mid-run.
- **No pass-weight shape** (ramp / peak-at-stop). Every pass counts the same.
- **No answer anchoring.** Flat CE over all passes, not v5's snapshot at the
  stop pass.
- **No worst-cell term.** Per-cell mean, so "solved is a MIN over cells" goes
  unpriced.
- **No clue grading, no clue ramp.** Only the cells the model is responsible for.

The interesting property of the two-term version is that *nothing explicitly
rewards stopping sooner*, and it may not need to: the settle term defines w as
"am I correct", and the stop rule fires when every non-clue cell is over
threshold, so w rises as cells come right and stopping falls out of correctness
by itself. Whether a grid ever learns to stop early without being paid to is
exactly what the first run asks. `pass_weight` adds the payment — `Σ(1-w)` over
the passes each grid ran, normalised by cells and budget, the same quantity v5
prices — and the pair isolates what it buys. `l_pass` is logged either way, so
the unpriced run still reports the quantity the priced run charges for.

### Random per-batch pass budget

`--rand-budget lo,hi` samples the budget per *batch* instead of holding it
fixed. Unlike `--budget-ladder`, which moves the budget in stages and re-climbs
the blank curriculum at each rung, this varies it every step, so the model never
learns which pass is the last one and has to be willing to answer at any depth.
Loss terms that normalise by the budget get the budget the batch actually ran
(`Trainer._loss` takes a `budget` override; the four loss versions all thread it).
**Eval stays at the fixed budget** so eval curves remain comparable across runs.

### The four runs (2026-08-07, A40, all concurrent)

Same house baseline as the v4 comparisons — d96/h4/ff384, ffn head,
settle_threshold 0.995, lr 1e-3 cycles + patience 150, batch 256, grad_clip 10,
promote_at 0.98, unique pools from `data/`, no head swap — differing only in the
loss and the budget. **`--refine-steps 12`, not the 36 every recent run used.**

| run | pass_weight | budget |
|---|---|---|
| `logs/v4/basic` | 0 | fixed 12 |
| `logs/v4/basic-pass` | 0.1 | fixed 12 |
| `logs/v4/basic-rand` | 0 | U(1,12) per batch |
| `logs/v4/basic-pass-rand` | 0.1 | U(1,12) per batch |

20k steps each. **Careful reading the rand rows against the fixed ones:** a
random budget averages 6.5 passes against 12, so those runs are cheaper per step
and get further per unit wall-clock. Compare at equal *steps* for the loss
question and note the wall-clock separately; they are two different questions.

### Logs reorganised

`logs/` had 56 run dirs at the top level. Everything before today moved to
`logs/old/`, this batch goes in `logs/v4/`. `app.py` discovered runs with a
one-level glob, which would have hidden every archived run from the Logs tab and
every old checkpoint from the picker, so both globs are now recursive: a run is
any directory holding a `config.json`, at any depth.

### The pass-penalty ladder — 2026-08-07

Measured on the running pair, the penalty at 0.1 is worth only **2.5% of the
loss** (`l_pass` ~ 0.08 against err 0.18 + settle 0.13), which is why the first
comparison looked inert. The ladder walks it up from there, everything else held
at the house baseline and a fixed 12-pass budget:

| run | pass_weight | share of loss | note |
|---|---|---|---|
| `logs/old/basic` | 0.0 | 0% | stopped, 46 blanks, step 6325 |
| `logs/v4/basic-pass` | 0.1 | 2.5% | **the survivor — through the 48-blank wall** |
| `logs/old/basic-pass10` | **1.0** | ~21% | equal to `settle_weight`; stopped, 46 blanks |
| `logs/old/basic-pw2` | 2.0 | ~34% | learned fine (grid 0.984 @ step 1025); stopped early |
| `logs/old/basic-pw5` | 5.0 | ~54% | **collapsed, never learned** |
| `logs/old/basic-pw10` | **10.0** | ~70% | **collapsed, never learned** |

**Naming hazard: `basic-pass10` is pass_weight 1.0**, not 10 — the "10" meant ten
times the original 0.1. `basic-pw10` is the weight-10 run. Read the
`pass_weight` field in the metrics rows, not the directory name.

`settle_weight` stays at 1.0 throughout, so weights above 1.0 put the pass term
*above* the only term anchoring `w` to actual correctness. The predicted failure
is grids inflating `w` and committing while wrong — high `finished_frac` with
falling `grid_acc`. At weight 10 the pass term is worth ~9.5 at cold start
against an error term of ~2.2, so the first few hundred steps tell the model to
settle far more loudly than to be right.

What the 0.1-vs-1.0 comparison showed before 5 and 10 were launched, matched at
step 2075 (same 36 blanks, same 16 promotions):

    pw 0.1   grid 0.979  passes 8.82  fin 0.81  l_pass 0.0739
    pw 1.0   grid 0.973  passes 8.62  fin 0.81  l_pass 0.0544

A 10x weight increase lowers the term it prices by 26% but buys only 0.20
passes. **Returns diminish sharply**: 0.0 -> 0.1 bought 0.77 passes (9.02 ->
8.25, matched at step 5150), 0.1 -> 1.0 bought 0.20. Where 1.0 clearly differs is
the *onset* — it is at fin 0.77 by step 1500 where 0.1 takes until ~step 2000 —
not the endpoint, which is ~8 passes either way.

**Hypothesis the ladder tests: the ~8-pass floor is the stop rule, not the loss.**
Every run so far bottoms out near 8 regardless of weight. Stopping requires every
one of ~46 non-clue cells over `settle_threshold` 0.995, and that conjunction may
be what costs 8 passes. If weight 10 still floors at ~8, the knob to turn is the
threshold, not the penalty.

### The 48-blank wall is where this gets decided

`basic-pass` reached 48 blanks at step 6525 and went backwards: grid 0.971 ->
0.912, passes back *up* 8.25 -> 9.46. That is the same boundary where
`v3-d96-answer` needed 3575 -> 7275 -> 16675 steps for its last three levels.
Nothing below 46 blanks separates these losses by much; the region above it is
the experiment.

### pass_weight 5 and 10 do not train at all — the saturation trap, 2026-08-07

Both collapsed inside ~150 steps and never recovered. Killed at step ~750.

    run       l_err            l_settle      l_pass          grad_norm
    pw 1.0    0.158 falling    0.155         0.026           4.8-31
    pw 5.0    2.197 = ln(9)    1.56 RISING   0.177           0.42
    pw 10.0   2.197 = ln(9)    2.13 RISING   0.091           0.92

`l_err` frozen at exactly ln(9) means the readout is still uniform — the model
learned nothing about digits in 750 steps, while pw 1.0 was at cell 0.956 by
step 400.

**The mechanism.** The pass term is minimised far more cheaply by inflating `w`
than by solving the grid. Within 150 steps `l_pass` crashes 0.95 -> 0.18/0.09
while `l_settle` *climbs* 0.4 -> 1.6/2.1: the model claims settledness on cells
it has not solved. Then it traps itself. `w` lands near 0.91, where
`dw/d(logit)` is small, so the settle BCE can no longer push `w` back — the same
saturation that satisfies the penalty kills the gradient that would correct it.
Total grad norm falls to 0.1-0.9 (against 5-30 for pw 1.0) and everything stops
moving, the digit readout included.

The trap's cruellest detail: `w ~ 0.91` is not enough to *stop*, because the
threshold is 0.995. So these runs never stop early, never pay a real pass
penalty, and never learn. A dead zone, not a trade-off.

This is the trap NOTES.md flagged for v5's share mechanism — "(1-w) can be driven
to exactly zero, so an uncapped share blows the weight up right as the gradient
w(1-w) dies" — reached from the other direction. v5 caps the multiplier
(`pass_k_max`); a large *fixed* weight walks straight in.

**Gradient clipping is not the cause.** `grad_clip` is 10.0 and during the dead
phase the norms were 0.1-0.9, so the clip never engaged. It did fire in the first
~100 steps (pw 5 hit 14.05, pw 10 hit `inf`, which the AMP scaler skips), but
clipping rescales every component by the same factor and cannot change the
error-vs-pass balance. That balance is the weight ratio alone.

**So pass_weight has a usable ceiling between 1 and 5.** `logs/v4/basic-pw2`
(weight 2.0) bisects it. No `best.pt` kept for pw5/pw10 — the weights are a
chance-level model; the metrics are the finding.

If the high-weight region is ever wanted, the fix is not a bigger clip but to
stop the penalty being payable by saturation: ramp `pass_weight` from 0 once the
model can already solve, or floor the settle gradient the way `err_settle_floor`
does for the v3 gate.

### Verdict on the pass penalty: 0.1 wins, and the ladder is closed

Matched at step 4525, both at 46 blanks with 21 promotions:

    pw 0.1   grid 0.959  cell 0.992  passes 8.78  fin 0.90  l_pass 0.1191
    pw 1.0   grid 0.938  cell 0.988  passes 8.96  fin 0.80  l_pass 0.0766

**pw 1.0 loses on the very thing it prices.** It drives `l_pass` down 36% (it must
— that is the term in the loss) while producing *more* real passes and fewer
finished grids. It optimises the surrogate and regresses on the objective: the
mild form of what killed pw 5 and pw 10, where `l_pass` falls because `w` is
inflated rather than because grids finish sooner. Steps-to-level are a dead heat
to 44 blanks, so the cost is quality at a level, not curriculum speed.

The whole ladder, best figure for each:

    pw 0.0   8.77 passes  grid 0.972  @46 blanks   (stopped)
    pw 0.1   8.63 passes  grid 0.959  @48 blanks  fin 0.93   <- only run past 48
    pw 1.0   8.81 passes  grid 0.961  @46 blanks   (stopped)
    pw 2.0   learned fine, stopped at 8 blanks / step 1025 -- so the collapse
             boundary is between 2 and 5, not between 1 and 5
    pw 5.0   dead      pw 10.0  dead

So the usable window is narrow and the *useful* weight is one worth ~2.5% of the
loss. Above that it degrades monotonically until it collapses. Not "tune it
higher" — the term is payable by saturation, so a bigger price buys a better lie.

### The settle:logits balance holds without any share machinery — 2026-08-07

Measured on `basic-pass` over 9550 steps, 8 -> 48 blanks, 20 promotions:

    settle : logits  =  0.736 : 1     (logits 56.1% of loss, settle 41.3%, pass 2.6%)

and the ratio over the whole run: **0.73-0.76 : 1**, logit share **55.3-56.3%**.
Flat. Not a trend — noise in the third decimal.

This contradicts the premise behind v4/v5's share rescaling, which NOTES.md
justifies as necessary because "a fixed multiplier cannot hold a balance while
the error's own share swings ~33x across the curriculum". The swing is real —
`l_err` moves **16x** here (0.133 .. 2.197) — but the balance survives it, because
**both terms are per-cell means over the same cells driven by the same fact** (is
this cell right yet). When error rises on harder puzzles the settledness BCE rises
with it. The share machinery solves a problem that fixed weights do not have,
at least not between these two terms.

The single exception is cold start: at step 10 the ratio is 0.21:1 (logits 79.6%)
because `l_err` begins at ln(9)=2.197 while the settle BCE begins near 0.47. It
converges to 0.74 inside ~500 steps and never moves again — and the transient
leans the right way, digits first, calibration after.

**Open, and the obvious next axis:** nobody chose 41%. It fell out of
`settle_weight = 1.0`. Whether the stopping behaviour needs that much gradient,
or whether it is better spent on the digits, is one `--basic-settle-weight` sweep
(0.5 / 2.0) away. The pass axis is exhausted; this one is untouched.

### Where the runs live

`logs/v4/` holds only what is **still running**; a stopped run moves to
`logs/old/`. The v4 batch therefore spans both directories, and the split is
live vs finished, not new vs old. `compare_v4.py` globs `logs/v4/*` by default,
so name paths explicitly to include archived runs:

    python3 compare_v4.py logs/v4/* logs/old/basic-2026* logs/old/basic-p*

`best.pt` force-added for every archived run that learned anything: `basic`,
`basic-rand`, `basic-pass-rand`, `basic-pass10`, `basic-pw2`. Nothing kept for
`basic-pw5`/`basic-pw10` — chance-level weights, the metrics are the finding.
Still live: `basic-pass` (pw 0.1).

## Which loss terms actually work — the term-by-term breakdown, 2026-08-07

Ten years of terms across v3/v4/v5/basic, and what the evidence says about each.
Written down because almost every term was introduced *alongside* others, so most
of them have never been measured on their own.

| term | prices | evidence | verdict |
|---|---|---|---|
| per-pass CE, flat (deep supervision) | be right at every pass | 4.2k vs 8.0k equivalents to 32 blanks (1.9x); settle 1.00 vs 0.58-0.62 | **load-bearing** |
| calibration BCE (w vs correctness) | w must mean "am I right" | alone it produces stopping: 12 -> 8.5 passes, fin 0 -> 0.87 at pass_weight 0 | **load-bearing** |
| pass penalty `sum(1-w)` | stop sooner | helps at 2.5% of loss, hurts at 21%, collapses at >=54% | **narrow window** -- payable by inflating w |
| share rescaling (v4/v5) | hold balance as scales swing | settle:logits flat at 0.73-0.76 across a 16x swing in l_err | **unnecessary** for these two terms |
| clue grading at uniform weight | don't move the clues | ~3000-step cold-start stall; clue_ramp 0.001 escapes at 625 vs 3750; 0.05 == off | **harmful without the ramp** |
| answer term (CE at the returned grid) | grade what is returned | v3-d96-answer is the best from-scratch v3 run (54 blanks), never A/B'd | **suggestive only** |
| err_settle_gate | price errors by the claim behind them | dead heat | **null so far** |
| pass-weight shapes (ramp / peak-at-stop) | when being wrong costs most | v3-peak's first 4400 steps provably say nothing; v4 dropped ramp un-ablated | **unresolved** |
| min_settle `-log(w_min)` | per-grid worst settledness | v4 deleted it, no ablation | **now under test** |
| worst-cell soft-max (v5) | solved is a MIN over cells | drafted, smoke-tested, never trained at scale | **now under test** |
| wrong-cell integral (new) | be right, fast, stop only when right | see below | **now under test** |

Two terms do all the work. One has a narrow usable window. One piece of
machinery is redundant. Four things had never been measured.

### The wrong-cell integral

    n_wrong(g,t) = sum_cells (1 - p_correct)          # expected wrong cells
    l_wrong = [ sum_{t<=stop} n_wrong(t) + (T - stop - 1) * n_wrong(stop) ] / T

It lives on the **logits**, which is the whole point: the pass penalty is
measured on `w` and can be paid off by inflating it (that is how weights >= 5
collapsed), whereas a wrong-cell count can only be lowered by fixing cells.

`(1-p)` rather than `-log p` matters: CE's per-cell gradient explodes as p -> 0,
so hopeless cells dominate, while a bounded count hands proportionally more
gradient to the last few stubborn cells once the rest are right.

**The hold is load-bearing.** Summing only over passes run has a hole -- a grid
stopping early *while wrong* truncates its own sum and pays LESS. Verified on a
forced stop at pass 1 with the model wrong: hold charges **25.56**, leaky charges
**11.03**, and the hold's figure matches never stopping (25.94). So stopping
early is no longer a discount.

**Caveat measured on the seed checkpoint:** `l_wrong` is 7.77 while the argmax is
wrong on only **0.38 cells**. ~95% of the "wrong-cell mass" sits on cells that
are already correct but not confident, so this term mostly pushes *confidence*.
That may be exactly what closes the last cells (more confident -> higher w ->
stops sooner) or mostly wasted. The run decides.

### The four-run term ablation — all resumed from one checkpoint

Seed: `logs/v4/basic-pass-20260807-193146/best.pt`, step 12175 at 48 blanks
(force-added; four runs depend on it). Resume restores step, blanks, optimiser,
scaler and RNG, so all four start from **byte-identical state** and differ by
exactly one term. Core throughout: flat per-pass CE + settle BCE + pass 0.1.

| run | added term | weight | realised share of loss |
|---|---|---|---|
| `logs/v4/term-none` | — (control) | — | 0% |
| `logs/v4/term-wrong` | wrong-cell integral | 0.011 | 14.1% |
| `logs/v4/term-worst` | v5 worst-cell soft-max | 2.01 | 17.0% |
| `logs/v4/term-min` | v3 `-log(w_min)` | 0.158 | 7.4% |

Weights were **measured, not guessed**: each was set so its term takes ~15% of
the loss at the seed checkpoint, since the pass-penalty ladder showed that a
term's share is what decides whether it helps or destroys the run. min_settle
realises 7.4% rather than 15% because `-log(w_min)` falls as soon as w rises.

`basic-pass` was stopped at 48 blanks: it never cleared the 0.98 gate there, and
the deliberate 6144-grid eval put it at 0.960 at budget 12 against 0.9922 at
budget 24. Note what that does and does not say -- three budget-12 runs
(`deep-refine` and its two continuations, all 891,692 params) reached 50, 52 and
54 blanks, so 12 passes is not itself the barrier; this model has 230k params and
the depth is compensating for capacity it does not have.

## The term ablation, measured — 2026-08-07

Four arms from one seed (`basic-pass` step 12175, 48 blanks), identical but for
one added term, all with `--fresh-lr-cycle` at a 1e-3 peak. 1375 steps of
divergence. `grid8` = mean of the last 8 evals; the sd of such a mean is
**0.0032**, so anything under ~0.01 is noise.

    run         grid8   cell8   passes   fin   grad_norm
    t2-wrong    0.963   0.9942   8.41   0.94    0.29
    t2-none     0.960   0.9932   8.33   0.94    0.31
    t2-min      0.953   0.9923   8.31   0.93    0.26
    t2-worst    0.926   0.9888   9.26   0.85    1.58

### v5's worst-cell soft-max is harmful

The term v5's entire design rests on -- "solved is a MIN over cells" -- drafted
2026-08-07, never trained until now. It costs **3.4 points of grid accuracy, a
full extra pass, and 9 points of finishing rate**, at 5x the gradient norm of
every other arm at an identical LR.

It is not slow recovery from the shock. The gap closed and then stopped:

    step    field   worst    gap
    12425   0.947   0.881   -0.066
    12925   0.953   0.914   -0.038
    13425   0.956   0.918   -0.039     <- flat for 500 steps

And the passes/fin damage was never a shock artifact: splitting each arm's
post-shock evals in half, the other three converge on 8.3-8.5 passes at fin
0.93 while worst sits at 9.31 / 0.85.

**Mechanism, and it is legible in the columns.** The term loads CE pressure onto
the *stop pass* specifically, so committing becomes more expensive, the model
delays stopping, and the delay buys nothing. A term meant to close the last cell
instead taught the model to keep deliberating.

### The other two new terms are null

`t2-wrong` (+0.003, ~1 sd) -- the total-incorrect-cells integral neither helped
nor hurt. Consistent with the measurement taken before launching it: `l_wrong`
was 7.77 while the argmax was wrong on only 0.38 cells, so ~95% of its mass sits
on cells that are already right, and the term mostly pushed confidence the model
did not need.

`t2-min` (-0.007, ~2 sd, still not clearly outside noise) -- v3's `-log(w_min)`.
**v4 was right to delete it**, though nobody had checked. Its one visible effect
was a worse `l_settle` (0.197 against 0.180), which is the mechanism: it pushes
`w` up on the worst cell whether or not that cell is right, degrading exactly the
calibration the BCE maintains.

### Where the term question now stands

Everything this project has tried, graded:

* **Earn their place (2):** flat per-pass CE (deep supervision), calibration BCE.
* **Narrowly conditional (1):** pass penalty -- only at ~2.5% of the loss.
* **Null (3):** min-settle `-log(w_min)`, wrong-cell integral, `err_settle_gate`.
* **Harmful (2):** v5 worst-cell soft-max, clue grading without a ramp.
* **Redundant (1):** share rescaling -- settle:logits holds at 0.73-0.76 across a
  16x swing in `l_err` without it.

Two terms do the work. The two-term `basic` core is, so far, the whole loss.

**Retracted: "no arm broke the 48-blank gate".** That was written at step 13550,
1375 steps after the shock, and it was wrong. Every arm broke it -- the shock just
needed ~4500 steps, not ~1400. All four reached 50 blanks. The claim that followed
from it, that the next axis had to be depth or width rather than a loss term,
does not hold either: an LR shock at a 12-pass budget cleared a gate that a
6144-grid eval had put at 0.960 against a 0.98 bar. Recorded because the wrong
conclusion was drawn from too short a window, twice in one session.

### The blank-count ladder was the cost, and it bought nothing — 2026-08-08

`ours-on-palm` trains this repo's recipe directly on Palm et al.'s pool
(`mixed-givens-17-34`, 47–64 blanks from step 0, no curriculum). Against the
two furthest curriculum runs, at the level where both of those lines died:

| run | 56 blanks ≥0.98 | budget |
|---|---|---|
| `t2-min-b48` | step 28420 | 48 |
| `t2-min-p36` | step 29430 | 36 |
| `ours-on-palm` | **step 4000** (0.988) | **32** |

Roughly 7x fewer steps at a smaller pass budget. Both curriculum runs set
`promote_at` 0.98 — an earlier note in this session said 0.9, which was read off
the wrong config.

**The comparison is fair, though it nearly wasn't.** `sudoku.make_batch` digs
blanks at random with no uniqueness check, and at 56 blanks 0/128 of those
puzzles are uniquely solvable (5% at 48 blanks, 1% at 54) — a 1.000 score on
them would mean nothing, since the eval scores against the one solution the
grid was dug from. But those runs ran with `unique_pool_dir=data`, so every
level drew from `unique-<N>-10000.npz`, and `_evaluate` calls
`_batch(revision=False)`, so no guesses are pre-filled. Both sides are
uniquely-solvable puzzles with nothing handed over. Worth re-checking on any
run that leaves `unique_pool_dir` unset: the random-dig path is silently
unscorable above ~48 blanks.

**The ladder is where the steps went, not the difficulty.** Both curriculum
runs cleared 0.98 at 56 blanks within 10–20 steps of *arriving* there — b48
arrived at 28410 and passed at 28420. The 28k steps were spent climbing 4→54
two blanks at a time, each rung demanding its own 0.98 with patience. So this
is not "the architecture learns hard puzzles faster". It is: the model could
have been training on 56-blank puzzles the whole time, and the ladder was
33k steps of arriving late. It also explains why b48 and p36 both stalled at
58 — that is simply where they ran out of session, not where they ran out of
capability.

### Sudoku-Extreme, zero-shot: the model learned propagation, not search — 2026-08-08

`ours-on-palm` at step 31000, run against HRM's Sudoku-Extreme test set
(422,786 puzzles, `sapientinc/sudoku-extreme` on HF) with no fine-tuning. The
set is format-compatible and 99.5% of it sits inside the 17-34 given training
range (17-36 givens, mean 25.2), so this is a fair transfer test even though it
is NOT a leaderboard entry: HRM (55.0%), TRM (87.4%) and EqR (86.4%) all train
on Sudoku-Extreme's own training split.

Overall zero-shot: 0.228 @32, 0.243 @64, 0.245 @128 on a 4096-puzzle sample.
The breakdown is the whole story. By Sudoku-Extreme's rating (how much
backtracking a solver needs):

    rating      n     b32     b64    b128
       0        603   0.975   0.987   0.987
     1-10      1027   0.333   0.378   0.384
    11-25      1069   0.000   0.007   0.008
    26-50       970   0.002   0.006   0.007
    51-100      376   0.000   0.000   0.000
     101+        51   0.000   0.000   0.000

**A cliff, not a slope.** Puzzles solvable by pure constraint propagation are
solved essentially perfectly; anything needing real search is solved never.

**Clue count is not difficulty.** Sorted by this repo's own axis the ordering
inverts: 17-21 givens scores 0.791 while 22-24 givens scores 0.104, because
Sudoku-Extreme's 22-24-given puzzles are curated from the hardest-known forum
collections while its 17-given ones are ordinary minimal puzzles. Every
curriculum, gate and ladder in this project has been built on blank count, and
blank count is close to orthogonal to the difficulty that matters. That is the
most useful thing this run has produced, and it did not need the run at all.

**The halt signal is a sound certificate.** At budget 64 on 4096 out-of-
distribution puzzles: 990 halts, 990 correct, **zero false commits**; 6 correct
answers arrived without halting. P(correct | halted) = 1.000 (95% upper bound on
the false-commit rate is ~0.3% given 0/990), P(halted | correct) = 0.994. The
model does not know how to search, but it knows exactly when it has not solved
something -- and it says so by refusing to stop rather than by committing to a
wrong grid. Worth keeping in view: the settle/BCE calibration term is what this
is measuring, and it survives a distribution it was never trained on.

**Confirmed on the full test set.** The zero-shot budget-32 figure above was
re-measured against all 422,786 Sudoku-Extreme test puzzles rather than the
4096-puzzle sample: **0.2269 +/- 0.0018 at n=211,456**, stable to four decimals
from n=21,504 onward. The run was stopped deliberately at the halfway mark --
the interval was already an order of magnitude tighter than anything the
conclusion depends on, and the GPU is shared with training. The 4096-sample
figure was 0.2275, so the original sampling was sound; the by-rating and
by-givens breakdowns above are still the 4096-puzzle ones (rating-0 cell
n=603, rating-101+ cell n=51).

### Measured their way: fixed-pass readout, and the stop rule is free — 2026-08-08

RRN and Recurrent Transformer have no halting: every puzzle gets 32 (train) or
64 (eval) steps and the answer is the readout at the last one. `eval_fixed_step.py`
runs our loop with `run_full=True` and reads `step_logits` at a fixed pass, which
is exactly their protocol. At step 34000 -- 11% of Palm's 300k-update budget:

| | ours | RRN | RT |
|---|---|---|---|
| 17-clue held out, fixed@32 | 0.9160 | 0.941 | -- |
| 17-clue held out, fixed@64 | 0.9600 | 0.966 | 0.967 |
| 17-34 mix, fixed@64 | **0.9946** | 0.989 | 0.995 |

Above RRN on the mix and level with RT, at a ninth of the training.

**halt@64 == fixed@64 at every clue level** (0.9946 vs 0.9946 on the mix, 0.9600
vs 0.9600 on 17-clue). Halting costs nothing in accuracy while using 6.84 passes
at 34 givens and 25.44 at 17.

**And it beats equal-compute fixed reading.** At 19 givens: fixed@32 is 0.959,
halt is 0.984 at a mean of 20.31 passes -- better accuracy *and* fewer passes
than reading everything at 32. Halting is per-puzzle, so a board that settles at
pass 12 is read at pass 12 while one that needs 60 keeps going; a fixed readout
interrupts the hard ones and lets the easy ones drift past their best moment.
The stop rule is not a compute optimisation with an accuracy cost attached, it
is picking a better moment to commit than "the end" is.

Caveat on the compute claim: a pass here is ~3 attention ops (self, cross,
commit) against RT's one block per recurrence, so 13.2 mean passes vs their
fixed 64 is ~1.6x fewer attention ops, not 4.9x. And the loop only exits early
when every example in the batch has halted, so per-puzzle savings need batch
size 1 or difficulty-bucketed batches to show up in wall clock.

### Comparability caveat: revision training is ours alone — 2026-08-08

Neither baseline uses anything like `revision_prob`. RRN computes node features
once from the puzzle digit and evolves only the LSTM hidden state across its 32
steps; Recurrent Transformer encodes the puzzle once into `H^(0,L)` and carries
`H^(r,0) = H^(r-1,L)`. Neither ever feeds a prediction back as an input digit,
and neither trains on boards containing wrong digits.

refine_v3 must: the readout becomes the next pass's input board, so from pass 1
onward the model always looks at its own possibly-wrong guesses, and training
only on clean puzzles would leave every pass after the first out of
distribution. The augmentation is a consequence of board feedback, not an
advantage bolted on.

Measured cost of the difference, palm ckpt step 37000 on 512 Extreme puzzles at
budget 32: plain grid 0.225, revision grid 0.861 (25% of guesses wrong). A
revision puzzle is ~4x more likely to be solved, which is the entire train/eval
gap in the logs -- and the reason the train column must be ignored on every run
in this repo.

**Where this bites.** Test time is clean (`_evaluate` passes `revision=False`),
so the headline numbers are measured on their task. But training differs twice
over: revision spawns many corrupted variants per solution, and `augment_pair`
re-permutes digits and bands on every draw where RRN permutes once at dataset
build. So "near their accuracy at 11% of their update budget" is architecture
PLUS training regime. Attributing it needs a `--revision-prob 0` ablation, which
has never been run.

**Correction to the entry above.** "Architecture PLUS training regime" overstated
it, and the framing implied revision is an advantage the baselines lack. It is
not. At inference, pass n of refine_v3 reads a board full of its own guesses,
some wrong -- that IS the input distribution the architecture visits, so
revision training covers the visited states rather than adding information.
Training only on clean boards would leave every pass after the first out of
distribution and make the model worse.

RRN and RT get the same coverage implicitly: their step-15 hidden state is also
their own intermediate guess, and backprop through 32 recurrences teaches them
to handle it. refine_v3 cannot rely on that alone because its feedback leaves
through a digit readout and re-enters as a board, so the coverage must be
supplied explicitly. Same mechanism, different feedback channel.

The 0.861 vs 0.225 figure measures how much easier the revision *task* is --
which is why the train column is uninterpretable -- not how much the model gains.

What actually survives as a regime difference is narrower: `augment_pair`
re-permutes digits and bands on every draw, where RRN permutes once at dataset
build, so their training set is 180,000 fixed items and ours is unbounded
variants of 11,952 grids. That cuts both ways -- their 38,151 seeds carry more
underlying grid diversity than our 11,952 -- and has nothing to do with board
feedback. Roughly a wash. The efficiency claim stands as stated.

**Precision on the above.** Two mechanisms, often conflated. `augment_pair` is
symmetry only -- relabel, band/stack permute, transpose; bijections of the
constraint graph, no wrong digits, and its purpose is amortising the uniqueness
check across draws. `add_guesses` is the one that inserts errors, and they are
synthetic: the true solution, a random fraction of blanks filled, 25% of those
replaced by the true digit shifted a random offset. Clues untouched.

Those synthetic errors only ever set the loop's *entry* state. From pass 1 on,
the input board is the model's own readout -- its real errors -- in training and
inference alike. Revision exists so pass 0 can look like an arbitrary mid-solve
state instead of always an empty board, and so there is dense signal early when
the model's own guesses are still noise.

So "covers the states the architecture visits" was loose: add_guesses corrupts
by a uniform random offset at random cells, while real errors are structured
(near-misses, clustered in the hard region). It is a hand-designed proxy for the
visited distribution, not the distribution.

**Retracted: "revision is required or passes >=1 go out of distribution".**
Wrong. The v3 loop feeds `logits_t` back into `x_kv` every pass during training,
and `deep_supervision` is on, so the model is already trained on boards full of
its own guesses at passes 1..T whether or not revision is enabled. The feedback
is `w.detach()`'d, so each pass is an independent supervised map board ->
solution; self-generated boards are squarely inside the training distribution.

What revision actually changes is only: (1) the pass-0 entry state, which
`add_guesses` fills to a uniform-random fraction so the loop can start from an
arbitrary mid-solve board; (2) the error character, synthetic uniform offsets vs
the model's own structured near-misses; and (3) early-training signal -- at step
500 the model's own guesses are noise, so self-feedback teaches nothing about
correcting a *plausible* mistake because none exist yet.

(3) is likely the whole point, which makes revision a bootstrapping aid rather
than a coverage requirement. The prediction that a `--revision-prob 0` ablation
would "degrade badly" has no basis; the shape to expect from a curriculum aid is
a slow start converging to a similar place. Still never measured.

### revision_prob 0 NaNs under AMP — 2026-08-08

The first attempt at the revision ablation (`palm-norev`, identical to
`ours-on-palm` but `--revision-prob 0`) never learned anything: `l_err` sat at
exactly ln(9) = 2.1972 from step 10, logits stayed uniform (std 0.0036 against
the baseline's 4.99), and grid accuracy never left 0.000.

It was not a hard-task slow start. The gradient norm decayed 0.66 (step 80) ->
0.033 (120) -> 0.026 (160) and the training forward went **NaN at step 200**.
With AMP the scaler then skips every step, so the model froze; the constant
`l_err`/`l_pass`/`l_min` in the logs after that are a static model being
re-measured, and the small `l_settle` wobble is batch noise.

Diagnostic that localises it: the **eval** forward stayed finite (2.1972) while
the **training** forward was NaN. Eval and train build identical batches when
revision is off, so the difference is autocast plus dropout -- fp16 overflow, not
the loss. `compute_basic_loss` is defensive anyway (`grid_denom` clamped,
`w_min` clamped at 1e-6).

`--no-amp` added to train_cli.py; there was no AMP toggle before, `amp: bool =
True` was simply unreachable from the CLI. Re-running as `palm-norev-fp32`.

**Consequence beyond this experiment:** an all-blank board drives activations
outside fp16 range on this architecture. Every run in this repo has had revision
at 0.5, which keeps boards mostly filled, so the instability has never been hit.
Any future run that stalls with grad_norm collapsing toward zero and then going
NaN should try `--no-amp` first.

### The revision ablation: refine_v3 collapses to a constant without it — 2026-08-08

`palm-norev-fp32` (identical to `ours-on-palm` but `--revision-prob 0 --no-amp`)
at step 4000, against the baseline at 39000:

| | norev | baseline |
|---|---|---|
| cell accuracy on blanks | 0.1102 (chance 0.1111) | 0.9988 |
| logit variance across cells | 0.00000 | 5.167 |
| \|logit(puzzle A) - logit(puzzle B)\| | 0.00000 | 0.376 |
| settledness w | 0.2511, sd 0.0000 | 0.9994 |
| predicted digits | all 14,189 cells = "7" | ~1575 of each |

It is a **constant function**: digit 7 everywhere, identical confidence
everywhere, output independent of the input including the clue digits it is
handed directly.

**Every logged scalar follows from that.** Near-uniform logits give
`l_err` = ln(9) = 2.19722, which it has held for 4000 steps (lowest value ever
recorded was step 160, never beaten). The halt head learned the one thing still
learnable -- the optimal constant: BCE pulls w toward the base rate of being
right (~1/9), `-log(w_min)` at 0.158 and the pass penalty at 0.1 push it up, and
the equilibrium is 0.2511. -log(0.2511) = 1.3820, which is exactly the `l_min`
the logs have been pinned at.

**Mechanism.** At init the readout is uniform, so `logits.argmax(-1)` is an
arbitrary constant, so the board folded back by `_board_kv` is constant, so
cross-attention reads a constant, so nothing makes the output input-dependent.
Self-reinforcing, and the feedback is a hard argmax that is `detach()`ed, so no
gradient can push back through the loop to break it. RRN and Recurrent
Transformer cannot fail this way: they carry continuous hidden state, and an
uninformative hidden vector still has gradient structure where an uninformative
digit has none.

**Conclusion, and it is not the one predicted.** Revision is not a head start or
a curriculum aid. Without it this architecture does not train at all. The
"the task is just hard to learn cold" reading is also ruled out: RRN learns this
exact distribution from cold with no revision and reaches 96.6%. The failure is
specific to the board-feedback design.

Revision is therefore load-bearing, and everything the v4/v5 line has measured
sits on top of a mechanism nobody had ablated. It buys the ability to bootstrap;
what it costs, if anything, is still unknown.

### The pool has no shallow end — 2026-08-08

Measured over `mixed-givens-17-34-216000-heldout`, counting rows/cols/boxes
with 8 of 9 cells filled (the last cell forced -- the one deduction requiring no
reasoning):

| givens | blanks | fullest unit | units with 8/9 | puzzles with a forced cell |
|---|---|---|---|---|
| 34 | 47 | 6.32 | 0.038 | 3.7% |
| 31 | 50 | 5.93 | 0.013 | 1.3% |
| 28 | 53 | 5.48 | 0.002 | 0.2% |
| 25 | 56 | 5.01 | 0.000 | 0.0% |
| 17 | 64 | 3.19 | 0.000 | 0.0% |

At the EASIEST end of the training mix the fullest unit averages 6.3 of 9, and
96.3% of puzzles contain no forced cell at all. Below 25 givens, none do.

Cause: puzzles are built by adding *randomly chosen* solution digits to a
17-given seed, so clues scatter evenly and no unit ever fills. A newspaper
"easy" 34-given puzzle is arranged so naked singles exist; a randomly-34-given
puzzle has none. Inherited from Palm's construction, which this repo copied.

**Corrects an earlier claim in this session** that the mix supplies "34-given
puzzles the model can solve on day one". It does not. There is no trivially
learnable subtask anywhere in this data -- the first deduction on the easiest
puzzle already needs constraints intersected across units.

This is why the bootstrapping problem is so sharp, and why clue grading is a
fix: copying a given is the only genuinely trivial signal available, and it has
to be inserted deliberately because the data contains none. It does not explain
the collapse to a *constant*, which stays specific to argmax feedback -- RRN
learns this same dataset from cold with neither revision nor clue grading.

### Straight-through feedback does NOT break the collapse — 2026-08-08

`--feedback {hard,soft,ste}` added to refine_v3. `ste` keeps the hard argmax in
the forward pass and passes the soft-mixture gradient backward, so the loop is
differentiable while the consulted board stays a definite grid. Verified
forward- and loss-identical to `hard` (3.17683744 both, logits allclose) with
differing gradients, and `test_app.py model` 18/18.

`norev-ste` (no revision, no clue grading, ste feedback, fp32) sat at
l_err 2.19722 with cell accuracy at chance for all 480 steps before being
stopped -- indistinguishable from plain `hard`.

**Why, and it was visible before launching:** at init, straight-through changed
the total gradient norm from 0.544819 to 0.544814 -- about 1 part in 100,000.
Making the fixed point non-flat is not enough when it is still nearly flat. The
loop needs a gradient of usable magnitude, not merely a nonzero one.

State of the fixes:

| fix | breaks the collapse |
|---|---|
| revision training | yes -- the original, undocumented dependency |
| clue grading (`--basic-clue-share`) | yes, below ln(9) by step 20 |
| straight-through (`--feedback ste`) | **no**, 480 steps unmoved |
| soft feedback (`--feedback soft`) | untried |

`soft` is the remaining candidate and is a much stronger signal, since the
fed-back board itself then moves continuously with the model's beliefs -- at the
cost of the board no longer being a definite grid, which is a real departure
from the design's premise rather than a free fix.

### The model switched off its own pass code — 2026-08-09

`residual_profile.py` instruments the refinement loop and reports, per pass,
`||h||`, `||V||`, their relative change, argmax churn, and how loud the two
signals injected alongside `h` are relative to it. On
`clueloss-blankdenom` at step 68000, 512 held-out 17-clue puzzles:

| pass | \|\|h\|\| | \|\|V\|\| | d h % | churn | time% | recpt% | grid |
|---|---|---|---|---|---|---|---|
| 1 | 0.257 | 0.285 | — | — | 0.02 | 6.59 | 0.0000 |
| 16 | 0.358 | 0.356 | 20.5 | 1.75 | 0.01 | 25.59 | 0.6641 |
| 32 | 0.883 | 0.743 | 4.4 | 0.15 | 0.00 | 13.50 | 0.9688 |
| 64 | 1.432 | 1.117 | 1.1 | 0.10 | 0.00 | 7.13 | 0.9824 |

**`time_scale` is trained to zero, in every run.** v3 replaced v2's learned
per-pass table with an unbounded sinusoid times one learned loudness knob,
init 0.03 to match v2's magnitude. Measured: 5.6e-05 at step 68000, i.e.
1/537 of init, putting the pass code at **0.02% of `||h||` at pass 1 and 0.00%
by pass 32**. It is not fading with depth — it is silent from the first pass.
And it is not this run: clueloss-blankdenom 0.018x init, ours-on-palm 0.0043x,
extreme-clueloss **-0.0089x** (sign flipped). It moved 10x between steps 68000
and 72000 of one run. That is a random walk near zero — an unused parameter.

**Why this is the explanation for the extrapolation, not a curiosity.** The
model has no explicit representation of which pass it is on, so it cannot
behave differently at pass 32 than at pass 33. That is exactly why halt@64 ==
fixed@64, why accuracy climbs smoothly past the trained budget to 0.9824 at 64,
and why running deeper than trained works at all: there is no deadline in the
model because there is no clock in the model.

**And it predicts the final-pass anchors cannot do what they were designed to
do.** `--basic-final-weight` / `--basic-final-settle-weight` ask for particular
behaviour *at a particular pass index*. With the clock off there is no clean
lever to condition on, so the gradient can only make the model uniformly faster
to converge, not specially good at 32. Caveat against overclaiming: depth is
still available *implicitly* -- `||h||` grows monotonically 5.57x, so the state
itself is a de facto clock even though consumers see `ln(h)`. The claim is that
the explicit signal is off, not that no depth information exists.

Cheap test on the running experiment: **if an anchor starts working by making
the model pass-aware, `time_scale` must grow away from zero.** If it stays a
random walk while `fixed@32` improves, the anchor worked as a generic
convergence pressure and the pass index was never the mechanism.

**Two secondary findings.**

*Pre-LN growth is real but mostly benign.* `||h||` grows 5.57x over 64 passes
and has not plateaued; `||V||` grows 3.92x. (A prior guess that `V` would stay
bounded because `commit` *rebuilds* it rather than accumulating was wrong --
`V`'s queries come from `V` and its content from a growing `h`, so it tracks
`h`.) Every consumer sees a LayerNorm (`ln1`/`ln_x`/`ln2` in the block,
`ln_v`/`ln_r`/`ln_out` on the record), so absolute scale is normalised away at
every use. What is *not* normalised is `h + _time(t) + w_r(r)`, which is how a
growing `||h||` silences the injections: the receipt peaks at 45% of `||h||` at
pass 2 and decays to 7% by pass 64.

*The answer settles long before the state does.* At pass 32 churn is 0.148
cells changed per grid -- one cell per ~7 grids -- while `h` is still moving
4.4% per pass. The state keeps drifting after the output has converged, and
given the norm growth much of that drift is inflation rather than computation.

### Which embeddings the model actually uses — 2026-08-09

`embed_contributions.py` decomposes the embedding sum and reports each term's
RMS, its share of the assembled sum, and its variation across the axis it
distinguishes (shares do not add to 100% -- the terms are not orthogonal, so a
share is "how loud", not "how much it owns"). Same checkpoint, step 68000:

**Which board you embed changes the answer**, so both are reported. At pass 0
the board is the raw puzzle, where at 17 givens 64 of 81 cells hold the *blank*
token -- `tok` there is mostly measuring "nothing here". From pass 1 on,
`_board_kv` pins the clues and fills every other cell with the latest guess, so
`tok` carries real digits. The steady-state column is the one that describes all
but the first pass.

| term | RMS @ pass 0 | share | RMS @ pass 32 | share | var/RMS |
|---|---|---|---|---|---|
| `tok*sqrt(d)` | 0.2468 | 1.02 | **0.3370** | 0.98 | 0.93 |
| `pos` | 0.0219 | 0.09 | 0.0219 | 0.06 | 0.51 |
| `row` | 0.0151 | 0.06 | 0.0151 | 0.04 | 0.56 |
| `col` | 0.0157 | 0.06 | 0.0157 | 0.05 | 0.56 |
| `box` | 0.0162 | 0.07 | 0.0162 | 0.05 | 0.63 |
| `kind` | 0.0334 | 0.14 | 0.0591 | 0.17 | 0.92 |
| **`settle_emb`** | 0.0916 | 0.38 | **0.1859** | **0.54** | — |
| `time` | 0.00004 | 0.0002 | 0.00004 | 0.0001 | — |

`tok` is 37% louder in the steady state because those 64 blanks stop being the
`0` token; `kind` nearly doubles for the same reason, as those cells flip from
kind `blank` to kind `guess`. (`tok`'s pass-0 share exceeds 1 because the other
terms partially cancel it -- the sum's RMS, 0.2424, is slightly below `tok`
alone.)

**Settledness is the second-loudest input in the model.** `settle_emb` is
zero-initialised by design -- "starts silent, speaks once training makes w mean
something" -- and it has grown to 0.54 of the embedding stack in the steady
state, three times `kind` and ten times any positional term. So the
per-cell "how sure am I" signal is not a minor conditioning hint; it is nearly
as loud as the digit itself. Two consequences. It is strong evidence that
`--settle-feedback` earns its place. And it means
`--basic-final-settle-weight` has a much bigger lever than the error anchor
does: changing w's calibration changes a signal at 0.54 of the embedding
magnitude, feeding the consulted board on every subsequent pass, where
`--basic-final-weight` only moves a readout. Expect the settle anchor to be
the more powerful *and* the more dangerous of the two.

**The positional embeddings are nearly silent, and that is probably correct.**
`pos`/`row`/`col`/`box` sit at 0.06-0.09 share, ~11x quieter than `tok`, though
they do vary (var/RMS ~0.5-0.6, so they are not constants). `rel_bias` is on in
every run in this line, which puts the grid geometry in the attention bias
instead -- so the additive positional terms are largely redundant rather than
broken. Untested prediction: `--structured-pos` off with `rel_bias` on should
cost little. That would be a cheap parameter saving, not a finding about
whether geometry matters.

**`kind` is quiet but sharply discriminative.** 0.14 share, yet its pairwise
distances relative to its own RMS are blank/clue 1.34, blank/guess 1.15, and
**clue/guess 2.14** -- the largest of the three. The distinction the fed-back
board depends on (a given digit vs one the model wrote) is the one the kind
embedding separates most, at low amplitude. Loudness and informativeness are
different questions, which is why the table reports both.

### The final-pass anchors are harmful — 2026-08-09

`--basic-final-weight` and `--basic-final-settle-weight` add extra CE / settle
BCE at the budget's last pass, the index RRN and RT are scored at. Both were run
at weight 10 (24% of the total loss) from a step-68000 `clueloss-blankdenom`
checkpoint, `--grad-clip 30`, 8000 steps, with the control's own trajectory from
the same seed as the baseline. **Result: worse on every axis, including the one
the terms were built to improve.**

Fixed-pass eval, held-out 17-clue, 2048 puzzles:

| | anchored @76k | control @73k | delta |
|---|---|---|---|
| **fixed@32** | **0.9590** | **0.9761** | **-0.0171** |
| fixed@64 | 0.9839 | 0.9888 | -0.0049 |
| halt@64 | 0.9839 | 0.9878 | -0.0039 |
| mean passes | 21.43 | 18.14 | +3.29 |
| 17-34 mix @64 | 0.9937 | 0.9970 | -0.0033 |

**Confirmed at a matched step.** The table above compares against a control
frozen at 73000, which leaves open whether the control would have drifted too. It
would not have: resumed, it reached held-out @32 0.9795 and 16.85 mean passes at
step 75000 -- near the top of its own band and *better* on passes than when it
paused. Against the anchored run at the same step 75000 (0.9463 @32, 0.9746 @64,
20.48 passes) the anchors cost **-0.0332 at budget 32** and -0.0137 at 64, nearly
double the frozen-checkpoint figure, because the control went on improving while
the anchored run declined.

`fixed@32` is what `--basic-final-weight` exists to raise, and it is the number
that fell furthest. Held-out halting accuracy over the run: 0.9395 at step
69000 (an immediate 4-point drop), recovering to a peak of 0.9609 by 71000, then
declining and settling in a 0.946-0.957 band -- the seed was 0.9756.

**The stop rule is not what broke.** `halt@64 == fixed@64` exactly (0.9839 vs
0.9839), so the model still commits at a good moment; it has a worse answer to
commit to, and needs 3.3 more passes to reach it. Nor were the anchored
quantities ever satisfied: `l_final` was 0.00808 at the start and 0.00778 8000
steps later, flat, while `l_err` and `l_settle` sat 13% above the control the
whole time. The model paid the cost and got nothing back.

**Why, and it was predictable from the pass-code finding.** With the clock off
the model cannot tell pass 32 from pass 33, so "be finished by 32" is not
learnable; the only reachable policy is uniform caution. Caution costs most when
the budget is tight and nothing when there is room, which is exactly the shape
of the damage (-0.017 at 32, -0.005 at 64). Notably `time_scale` did grow 48x,
from 5.6e-05 to 2.7e-03, well outside the control's random walk -- so the anchors
*did* push the model toward pass-awareness, and what it used the clock for was
stalling, not finishing.

**Methodological failure, recorded so it is not repeated.** Both anchors were
switched on in the same run, which is exactly what `compute_basic_loss`'s
docstring forbids -- the optional terms are to be enabled ONE AT A TIME so a run
isolates one thing. So this result condemns the *pair* at weight 10; it does not
apportion blame. The settle anchor is the likelier culprit, since `settle_emb`
is 0.54 of the embedding stack (see the embedding-contributions entry) and so
retuning w perturbs a loud input on every later pass, where the error anchor only
moves a readout. That remains inference, not measurement. The isolation runs --
error anchor alone, settle anchor alone -- were not run.

Artifact: `logs/v4/final32-both-w10-clip30-20260809-100530/latest.pt` (step
76000, force-added). Two earlier attempts are also on disk: `final32-w02`
(weight 0.2, ~1% of the loss, abandoned as a no-op) and
`final32-both-w10-clip5` (clip 5 tightened rather than loosened by mistake).

### Cross-domain transfer is asymmetric — 2026-08-09

Each model evaluated on the other's distribution. `clueloss-blankdenom` at step
75000 (trained on `mixed-givens-17-34`, `revision_prob 0`) and
`extreme-clueloss` at step 63000 (trained on `extreme-train-full`,
`revision_prob 0.5`).

| model | held-out 17-clue @64 | Sudoku-Extreme test @64 |
|---|---|---|
| Palm-pool | **0.9883** | **0.2954** |
| Extreme | **0.9321** | **0.5847** |

**Transfer runs one way.** The Extreme model gives up 5.6 points moving to the
Palm set (0.9321 vs 0.9883); the Palm model gives up 29 points moving to Extreme
(0.2954 vs 0.5847), i.e. half the specialist's score. Search-heavy training
transfers down to clue-poor puzzles; clue-poor training does not transfer up.

**The cliff is in search depth, not clue count.** Palm-pool model on Extreme, by
rating: 0.998 at rating 0 (n=603), 0.457 at 1-10, then **0.066 at 11-25**, 0.057
at 26-50, 0.029 at 51-100. It is near-perfect with no backtracking and near-zero
with any, and the fall is a cliff between 1-10 and 11-25 rather than a slope. By
givens over the same puzzles it scores 1.000 at 31-36 and 0.776 at 17-21 but
0.108 at 22-24 -- where the hard-rated puzzles sit. This is the sharpest version
yet of the propagation-not-search finding: the two difficulty axes are close to
independent, and this repo has only ever trained on one of them.

**The Extreme model pays in passes.** On held-out 17-clue it needs 28.85 mean
passes against the Palm model's ~17. Hence `fixed@32` 0.8281 against `fixed@64`
0.9321 -- at budget 32 it is truncated mid-computation rather than answering
wrong. `halt@64 == fixed@64` exactly (0.9321), so its stop rule is free too, and
the 17-34 mix at 64 is 0.9898.

Not a recipe comparison: different steps (75000 vs 63000) and different
`revision_prob` (0 vs 0.5). It measures transfer between two trained models.

### Rating the easy set, and the extreme model's divergence past its budget — 2026-08-09

`rate_heldout.py` gives Palm's held-out 17-clue set the rating axis it never had:
propagate naked + hidden singles to a fixpoint, then count wrong guesses.
Validated against Sudoku-Extreme's own `rating` on 600 of their test puzzles --
**92.2% agreement on the rating-0 boundary**, Spearman 0.651, and median guesses
rising monotonically across their bands (0, 5, 91, 98, 116, 273). Units are not
theirs, so bands are not interchangeable; rank order is.

**Two earlier metrics were useless and are recorded so they are not retried.**
Counting MRV *branch points* gave a median of ~1900 on this set; counting MRV
*dead ends* gave a median of ~10071 with a max of 2.77M, three orders of
magnitude above Extreme's scale and with 90% of puzzles in two bands. Both
failed for the same reason: bare MRV prices propagation as expensive, and the
model is a propagation machine. The axis must make propagation free.

The set itself: median 0 guesses, mean 3.7, max 357, and **48.2% solved by
propagation alone**.

Extreme model (step 66000) on the held-out set, by band:

| band | n | halt | fail rate | share of failures |
|---|---|---|---|---|
| 0 | 1273 | 0.962 | 3.8% | 33% |
| 1-9 | 624 | 0.899 | 10.1% | 43% |
| 10-49 | 126 | 0.778 | 22.2% | 19% |
| 50-99 | 12 | 0.917 | 8.3% | 1% |
| 100-499 | 13 | 0.615 | 38.5% | 3% |
| all | 2048 | 0.929 | 7.1% | 100% |

The failure *rate* climbs monotonically with search depth, the same cliff as on
Extreme but much shallower. Yet a third of its failures need no search at all and
76% are in bands 0-9: the tail is weakest per puzzle, the bulk of the loss is on
easy puzzles.

The Palm-pool model (step 79000) on the same puzzles is nearly flat -- 0.995,
0.978, 0.976, 1.000, 1.000 -- solving **all 25 of the hardest-rated**. Its 1.1%
failures are rating-independent here. Not a contradiction of the Extreme cliff:
this set tops out at 357 guesses with 62% at zero, so it barely probes the axis.

**The real find: the extreme model now diverges past its trained budget.**

| extreme model | fixed@32 | fixed@64 | halt@64 | mix @64 |
|---|---|---|---|---|
| step 61000 (pre-pause ckpt) | 0.8115 | **0.9082** | 0.9067 | **0.9852** |
| ~step 62000 | 0.8281 | **0.9321** | 0.9321 | -- |
| step 66000 | 0.8232 | **0.8022** | 0.9292 | **0.4592** |

**Not a resume artifact, checked three ways.** The configs differ only by the two
new anchor fields, both 0.0 and therefore inert. There is no discontinuity at the
seam: lr 1.0e-04 both sides (no `--fresh-lr-cycle`, so the cycle carried),
blanks 64 both sides, loss 0.672/0.798/0.755 before against 0.717/0.711/0.765
after, and the optimizer state is restored from the checkpoint's `opt` key. The
pre-pause checkpoint at 61000 does NOT diverge -- depth still helped it, 0.8115
-> 0.9082, mix 0.9852. So the collapse developed over the ~5000 steps after the
resume, peaking around 62-63k first. Unseparable confound, stated for honesty:
the resume lost 830 steps to the checkpoint interval and re-trained them on
different draws, so this is not the exact counterfactual trajectory.

Note the model is not simply degrading -- its *committed* answer improved over
the same interval (halt 0.9067 -> 0.9292) while its depth behaviour collapsed.

Running to 64 passes went from helping to actively destroying the answer -- on
the mix, 0.97 at 32 against 0.46 at 64 -- inside 4000 steps, while its
in-distribution Extreme score improved 0.5487 -> 0.5632. It is trading
depth-extrapolation for specialisation. **This also breaks the transfer-table
figure above**: the 0.9321 recorded there is a snapshot that no longer holds, and
the gloss "pays for its generality in passes" was wrong -- it no longer uses the
extra passes productively, it is harmed by them.

**And this is the first checkpoint where the stop rule is not merely free.**
`halt@64` 0.9292 against `fixed@64` 0.8022 is **+12.7 points** for halting: it
commits before the divergence sets in. Everywhere else in this repo halt@64 ==
fixed@64 to four decimals. Worth watching whether the Palm-pool line ever
develops the same thing, since it is an argument for halting that none of the
earlier measurements could make.

### The cross-attention is a constraint graph, not a residual — 2026-08-09

`attention_maps.py` recomputes the cross-attention weights (the module uses
`F.scaled_dot_product_attention`, which returns none) and asserts the
reconstruction reproduces the module's own output before reporting. The question
was whether cross-attention had collapsed to near-diagonal, in which case it is
imitating a residual connection and could be replaced by one.

**It has not. It is a constraint graph.** `clueloss-blankdenom` at step 80000,
mass averaged over heads and cells, against what a uniform map would give:

| pass | self | same-row | same-col | same-box | unrelated | diag |
|---|---|---|---|---|---|---|
| uniform | 0.012 | 0.099 | 0.099 | 0.049 | 0.741 | -- |
| 8 | 0.041 | 0.383 | 0.383 | 0.170 | **0.023** | 0.060 |
| 32 | 0.029 | 0.386 | 0.385 | 0.181 | **0.019** | 0.041 |

Peer mass is 0.90-0.97 per head against a 0.247 baseline, and mass on unrelated
cells is 0.019 against a chance 0.741 -- nearly all suppressed. Diagonal mass is
0.02-0.07, so no head is a residual. **No simplification is available here**, and
the mechanism is genuinely unlike HRM or Recurrent Transformer: neither feeds a
prediction back as an input digit, so neither has a pass in which cells read
their row/column/box peers' *current guessed values*.

**Row and column are learned as interchangeable.** `rel_emb` is symmetric to
three decimals -- h0 1.712/1.709, h1 1.523/1.527, h2 1.473/1.472, h3
1.707/1.705 -- and realised mass follows (0.386/0.385).

**Box is where heads specialise.** Same-box bias h0 2.212, h1 1.466, **h2
0.188**, h3 2.140: two heads weight box above row/col, and h2 ignores boxes
entirely, making it a pure row/column head.

**Routing is static in geometry and dynamic in confidence, and there is a trust
schedule.** Per peer cell the mass is essentially uniform -- row 0.386/8 = 0.048,
col 0.385/8 = 0.048, box 0.181/4 = 0.045 -- i.e. a graph convolution over a fixed
constraint graph. What moves is *which kind* of peer it reads. Attention mass on
clue cells, as a share of peer attention (clues are 0.210 of peers):

| pass | clue mass | ratio to share | recency, blanks only | changed |
|---|---|---|---|---|
| 2 | 0.661 | **3.15x** | 0.97 | 31.5 |
| 4 | 0.503 | 2.40x | 0.79 | 16.3 |
| 8 | 0.400 | 1.90x | 0.59 | 8.0 |
| 12 | 0.346 | 1.65x | 0.53 | 3.6 |
| 20 | 0.324 | **1.54x** | 0.05 | 0.3 |

The model leans on the givens 3.15x at pass 2 and relaxes to 1.54x by pass 20,
shifting onto its own guesses as they settle. The mechanism is the keys: they
carry `kind` and `settledness`, so the distribution moves along the
given-vs-inferred axis while the relation bias holds the topology fixed.

**It still does not chase the frontier**, but the first version of this measurement
overstated the case. Raw recency looked strongly below chance (0.40-0.48 early),
which was mostly an artifact: changed cells are never clues, and the model prefers
clues. Restricted to non-clue peers, recency at pass 2 is **0.97 -- exactly
chance**. It never exceeds 1.0 at any pass, so there is no frontier-following, but
the real signal is the mild avoidance that develops mid-solve (0.53-0.64),
consistent with churning cells being unreliable to read. Recency was simply the
wrong probe: confidence is the axis this model routes on, and it is the better
proxy for reliability than recency is.

**The extreme model is structurally different: two of four heads are at chance.**
At step 67000 its per-head peer mass is 0.245, 0.868, 0.291, 0.880 -- h0 is at
0.245 against the 0.247 baseline, i.e. exactly uniform -- and overall unrelated
mass is 0.397 against the Palm model's 0.019. Its relation bias is much weaker
(unrelated -1.3 to -2.7 vs -3.2 to -4.3). Training on search-heavy puzzles has
repurposed half its heads into global pooling rather than local propagation.

Two loose ends this closes and opens. It **confirms the embedding finding from
the other side**: geometry lives in `rel_emb` (a ~5.8 logit gap between row/col
and unrelated), which is why the additive positional embeddings measure near
silent at 0.06 share. And it **rules out attention collapse as the cause of the
extreme model's divergence** past pass 32 -- its maps are stable at passes 32,
40, 48 and 64 (peer mass 0.245 throughout), so the divergence lives downstream in
the record or the readout.

### Is the settledness signal honest? Yes — 100% precision when it commits — 2026-08-09

The worry: `l_settle` trains `w` against a detached correctness target, but `w`
comes from `_receipt(V)` and `V` from the same trunk the readout uses, so its
gradient reshapes the shared representation. Two further terms push `w` up
*regardless* of correctness -- `l_pass` = sum(1-w) at 0.1 and
`basic_min_settle_weight` = -log(w_min) at 0.158 -- against `l_settle` at 1.0 as
the only anchor to truth. That is a recipe for manufactured confidence.

Measured instead, `clueloss-blankdenom` at step 86000, 2048 held-out 17-clue
puzzles at budget 32, per-cell `w` at each grid's own stop pass:

| w bin | cells | P(correct) | gap |
|---|---|---|---|
| 0.5-0.9 | 1356 | 0.5929 | -0.125 |
| 0.9-0.99 | 398 | 0.9397 | -0.017 |
| 0.99-0.995 | 165 | 0.9818 | -0.011 |
| 0.995-0.999 | 1822 | 0.9989 | +0.001 |
| 0.999+ | 127331 | **1.0000** | 0.000 |

**P(correct | w >= 0.995) = 0.99998** over 129,153 cells. Of 581 wrong cells at
the stop pass, **2** were above threshold. Mean `w` is 0.718 on wrong cells
against 0.9981 on right ones. Calibration error (mean |w - correct|) is 0.0051.

At grid level: **97.80% of grids halt, and of those 100.00% are fully correct**
(2003/2003). The grids that run out of budget are 2.20% and only 13.3% right.
Overall 0.9810 = 0.978 x 1.000 + 0.022 x 0.133. The entire error rate is
non-commitment, not false confidence. If anything the model is *under*-confident:
the only real miscalibration is the 0.5-0.9 bin, far below the 0.995 threshold,
where it never touches the stop decision.

**Three consequences.**

*Accuracy is budget-limited, not confidence-limited.* The lever on the headline
number is more passes for the hard 2.2%, which is what the budget-64 row already
shows (0.9932 at step 85000). Sharpening `w` buys nothing.

*This is the retrospective explanation for the final-pass anchor failure.*
`--basic-final-settle-weight` targets committing-while-wrong, a failure mode worth
2 cells in 2048 grids. Pushing on a quantity already at 0.99998 leaves only one
direction to move -- more reluctance to commit -- which is exactly what was
measured (passes 17.4 -> 21.96, accuracy down 3.3 points). The term was aimed at
a non-problem.

*And it corrects the 26-given note above*: that single wrong cell was one of the
~2 confident errors in the entire set, not a representative calibration miss.

Practical: `w` is a usable abstention signal -- answer only when halted for
**100% precision at 97.8% coverage**.

### Does it prefer the obvious change over the forced one? — 2026-08-09

A sharper version of the settledness worry, and not the one the calibration entry
above answers. `w` is rewarded for confidence and for matching correctness, so a
change that leaves the model more certain may be worth more than the deduction the
puzzle actually needs. That cannot be seen in `w`'s calibration -- wherever the
obvious move *is* the correct move the two objectives agree -- so it has to be
looked for where they come apart. `obvious_bias.py`, 512 held-out puzzles,
`clueloss-blankdenom` at step 87000.

**It does fill obvious-first.** Labelling each blank by the propagation round that
determines it (round 1 = naked/hidden single from the clues alone), the pass at
which the model settles a cell tracks that round almost exactly:

| prop round | cells | share | settle pass | accuracy |
|---|---|---|---|---|
| 1 | 1878 | 0.057 | 1.25 | 1.0000 |
| 2 | 2058 | 0.063 | 1.83 | 1.0000 |
| 3 | 1918 | 0.059 | 2.27 | 1.0000 |
| 4 | 1806 | 0.055 | 2.85 | 1.0000 |
| 5 | 1802 | 0.055 | 3.18 | 1.0000 |
| 6+ | 12800 | 0.391 | 6.56 | 0.9998 |
| **needs search** | 10506 | 0.321 | **8.53** | **0.9861** |

**But the ordering is dependency-forced, not incentive-driven.** A round-3 cell
cannot be resolved before its round-2 prerequisites are placed, so obvious-first
is what any correct propagation does. The incentive version of the worry predicts
*avoidance* of hard cells, and there is none: the model fills all 32.1% of
search-needing cells and gets 98.61% of them right.

**Its errors are not plausible-looking fills either.** A model optimising local
legibility would produce conflict-free wrong grids. Measured: 149 wrong cells
across 11 wrong grids, **49% of them actively conflict with a peer's digit**, and
**0 of 11** wrong grids have all errors locally consistent. Mean 13.5 wrong cells
per wrong grid -- these are grids abandoned mid-solve with a large unresolved
region, not single bad deductions, which agrees with the calibration entry's
finding that the whole error rate is non-commitment.

**Where the worry does land: competence, not choice.** Accuracy is 0.9999 on
propagation-determined cells against 0.9861 where search is needed -- a 100x
higher error rate on exactly the cells where obviousness and truth diverge, and
all of the error mass sits there. This is the Sudoku-Extreme rating cliff measured
from the inside, on the easy distribution.

Caveat on the labels: only naked and hidden singles are implemented, so "needs
search" at 32.1% is an upper bound -- many of those cells fall to naked pairs or
box-line reduction, which the model may implement. It also explains how the model
solves ~98% of puzzles when only 50.6% are fully solvable by these rules.

### Matching HRM/TRM's protocol: 1000 base puzzles — 2026-08-09

`extreme-clueloss` trains on the full 3,831,994-row Sudoku-Extreme train split,
which the entry above already flags as an easier setting than the published
numbers. HRM (55.0%) and TRM (87.4%) train on **1000 base puzzles**. `v4/extreme-1k`
(`logs/v4/extreme-1k-20260809-143216`) is that protocol: identical recipe to
`extreme-clueloss` -- basic loss, `revision_prob 0.5`, clue_share 0.5, budget 32,
batch 256 -- with `--mixed-pool data/extreme-train-1000.npz`, which is a uniform
1000-row sample of the same split (mean 25.1 givens against the full split's 25.2).

**From scratch, deliberately.** Resuming from an `extreme-clueloss` checkpoint
would leak 3.8M puzzles into a run whose entire claim is that it saw 1000.

**They augment, and so do we -- more, and now checked against the sources rather
than assumed.** Read the table before quoting any margin over a baseline.

| | base puzzles | augmentation | applied |
|---|---|---|---|
| RRN (Palm) | Royle's 49,151 17-clue grids -> 180,000 puzzles | **digit relabel only** (9! = 362,880) | once, at generation |
| HRM | 1000 (`--subsample-size 1000`) | band + digit permutations, x1000 (`--num-aug 1000`) | once, materialised |
| ours | 216,000 from the same Royle set | digit relabel x row band+line (1296) x col band+line (1296) x transpose (2) = **~1.2e12** | **fresh every draw** |

HRM's flags confirm 1000 bases x 1000 augmentations (~1M examples), so the
base-puzzle count is the thing `extreme-1k` matches. Their paper describes the
transforms as band and digit permutations -- no transpose.

**RRN's generator is ours minus the augmentation, and that matters more.**
`generate_hard.py` takes the same Gordon Royle 17-clue set, applies `permute()`
(digit relabel), then `add()` -- filling k random cells from the solution for
k = 0..17 to reach 17-34 givens. That is exactly `gen_from_17.py`'s construction,
from the same source, over the same givens range. Then `data.py` reads the CSVs
and augments nothing per batch. So our data source and construction match theirs
closely while our symmetry group is ~3.4 million times larger (the band/line
permutations and transpose they never apply), resampled per draw rather than baked
in.

**What that does to the headline.** "3.6x fewer failures than RRN at budget 32"
is measured on their task, at their protocol, on solution-disjoint held-out
puzzles, so it is not contaminated -- but part of the margin plausibly comes from
stronger augmentation rather than from the architecture, and the claim should be
stated with that attached. An augmentation-matched run (digit relabel only, once
at pool build) is the experiment that would separate the two, and has not been run.

Corrects an earlier note in this file which said RRN "permutes once at dataset
build": true, but it is digits *only* -- no band, line or transpose permutation.

So matching the base-puzzle count closes the big gap but **two differences remain,
both in our favour**, and a number from this run should be reported as "1000 base
puzzles, our augmentation and revision training" rather than as a head-to-head:

1. unbounded per-draw augmentation against a fixed ~1000x materialisation. Modest
   in practice -- 1000 already defeats surface memorisation -- but strictly more.
2. `revision_prob 0.5`: we train on boards carrying the model's own wrong digits.
   Neither baseline feeds a prediction back as an input digit at all.

A third thing to watch that the full-split run could not show: with 1000 base
grids, overfitting is finally a live risk. Eval is on the separate test split, so
the watcher's numbers stay honest, and the train/test gap is now informative
rather than the noise the revision augmentation usually makes of it.

Note `trainer.py:424` only overrides `blanks` when the npz carries a `blanks` key,
which these pools do not -- so the "1000 puzzles at 64 blanks" in the log is
cosmetic. The pool is used whole; nothing filters it to 64 blanks.

### Rating is not monotone in difficulty for this architecture — 2026-08-09

Correcting a claim made in passing during the sweep setup, that models here score
zero from rating 11-25 upward. That is true of exactly two things and neither
generalises: the Palm-pool model evaluated **zero-shot** on Extreme (0.013 at
11-25, 0.003 at 51-100 -- the propagation-not-search finding), and `extreme-1k`
at step 4000, which is simply early.

A model trained on the split handles the search-heavy bands. `extreme-clueloss`
at step 80000, 8192 test puzzles:

| rating | n | b32 | b64 |
|---|---|---|---|
| 0 | 1211 | 0.987 | 0.993 |
| 1-10 | 2061 | 0.594 | 0.684 |
| 11-25 | 2080 | **0.448** | 0.509 |
| 26-50 | 2006 | 0.509 | 0.577 |
| 51-100 | 757 | 0.567 | 0.655 |
| 101+ | 77 | **0.610** | 0.714 |

**The profile dips and then rises.** The worst band is 11-25, not 101+, and the
hardest-rated band is its second-best after rating 0. So Sudoku-Extreme's rating
is not a difficulty ordering for this architecture, and "harder rating = lower
score" is wrong here. The by-givens table from the same eval points at why: 0.667
at 17-21 givens, **0.407 at 22-24**, 0.668 at 25-27, 0.892 at 28-30, 1.000 at
31-36. The trough sits at 22-24 givens on both axes, so what the rating partly
tracks is clue count, and the high-rating puzzles are not clue-poor.

Consequence for the sweep: the hard bands are not an untouched frontier, and the
question `logs/extreme/` asks is not "can it do search at all" but "how much of
the full split's 0.45-0.61 survives training on 1000 base puzzles". The sweep
dashboard now draws that run's per-band scores as a dashed rule on each band
graph, read from its log at build time.

### The width sweep on 1000 base puzzles: five sizes — 2026-08-09

`logs/extreme/` now holds five runs of the same recipe, differing only in width,
all trained on `extreme-train-1000.npz` (HRM/TRM's 1000 base puzzles) with
augmentation matched to their transform set (`--no-aug-transpose`). Held-out
numbers are `eval_sudoku_extreme.py` on 4096 test puzzles, seed 0.

**At matched step 8000 — every run has this rung.**

| run | params | b32 | b64 | rating 0 | 1-10 | 11-25 | GPU-h |
|---|---|---|---|---|---|---|---|
| h2·d24 | 17,511 | 0.0210 | 0.0017 | 0.012 | 0.000 | 0.000 | 1.67 |
| h4·d48 | 61,785 | 0.0625 | 0.0476 | 0.260 | 0.037 | 0.000 | 1.69 |
| h4·d96 | 230,625 | 0.1821 | 0.2129 | 0.882 | 0.294 | 0.022 | 2.39 |
| h6·d144 | 506,631 | 0.3521 | 0.4177 | 0.922 | 0.425 | 0.276 | 2.71 |
| h8·d192 | 889,773 | **0.4382** | **0.5176** | 0.955 | 0.509 | 0.394 | 3.27 |

(band columns at budget 64.) Monotone in size in every column, with no
exceptions — the advantage is not a difficulty-mix effect.

**Width buys search, not propagation.** Rating 0 is nearly saturated from
230k params up (0.88-0.96) and the sizes separate on the search bands: 11-25
reads 0.000 / 0.000 / 0.022 / 0.276 / 0.394 across the five. The two narrow runs
have never solved a puzzle above rating 0. h4·d96 shows the acquisition is also
a matter of training time, not size alone — it sat at exactly 0.000 for three
rungs, broke at step 8000, and reached 0.250 by step 14000.

**Extra passes help the wide runs and hurt the narrow ones.** b64 against b32:
h2·d24 0.0210 -> 0.0017, h4·d48 0.0625 -> 0.0476, but h4·d96 0.1821 -> 0.2129
and both larger runs likewise gain. The crossover coincides exactly with halting
starting to work — finished fraction at the same rung is 0.00 / 0.03 / 0.29 /
0.42 / 0.45. A run that never halts keeps refining past its training horizon and
overwrites correct cells; the halt head is what protects the answer, so more
budget is only safe once it fires.

**Budget 96 buys nothing over 64**, measured on all three larger runs at one
checkpoint each: +0.0007 / +0.0043 / +0.0144, all within noise, with the
finished fraction barely moving (h8 0.52 -> 0.54). The grids still unfinished at
64 are stuck, not mid-solve. 64 is the evaluation ceiling for this line;
`logs/extreme/budget96/` holds the raw output.

**Where the sizes stand now** (different steps, so read the matched table above
for the comparison): h8·d192 0.5413 at step 16000 / 5.99 GPU-h, peaking 0.5510
at step 14000; h6·d144 0.5015 at 12000; h4·d96 0.4082 at 14000. HRM's 0.550 on
the same base count is a landmark, not a like-for-like target — our augmentation
resamples per draw where theirs materialises 1000 fixed images per puzzle, and
we also train on revision boards. The full-split reference is 0.6566 at b64, so
the cost of 1000 bases rather than 3.83M is roughly 11 points at this width.

**A 17,511-parameter model solves real Sudoku-Extreme puzzles.** h2·d24 is at
86 of 4096 at budget 32 by step 8000, all of them rating 0. On the trainer's own
clean-board eval it went from zero to solving at step 2500. It has never halted
on any grid, at any budget.

`show_solved.py`, `solve_order.py` and `cell_certainty.py` inspect a single
puzzle: the grid it solved, the pass at which each cell locked in, and the halt
head's per-cell settledness. On one solved board h2·d24 had all 47 blanks
correct by pass 9 and ran to 32 without stopping — not from confusion but
because settledness spanned 0.971-0.999 and only 31 of 47 cleared the 0.995
threshold. One cell at 0.971 holds a finished grid open.
