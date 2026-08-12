"""SVD-based weight inspector.

Loads a checkpoint and computes the singular-value spectrum of every weight
matrix in the model, then reports the spectral shape of each: how much of the
matrix's action lives in a few directions vs. spread across many. Useful for
spotting low-rank collapse, dead layers, blown-up spectral norms, and which
projections the model actually leans on.

Per matrix ``W`` (shape m x n, singular values sigma_1 >= ... >= sigma_r):

  * **spec_norm**  sigma_1                       largest gain of the map
  * **fro_norm**   sqrt(sum sigma_i^2)           total energy
  * **stable_rank** ||W||_F^2 / sigma_1^2        soft rank, robust to noise
  * **eff_rank**   exp(-sum p_i ln p_i), p_i = sigma_i / sum sigma
                                                 entropy rank (Roy & Vetterli)
  * **cond**       sigma_1 / sigma_min           conditioning
  * **r90 / r99**  #components for 90% / 99% of the energy (sum sigma^2)

Fused projections (``qkv``, ``kv``) are split into their q/k/v parts by default,
since those blocks play different roles and their spectra are worth seeing apart.

Usage:
    python3 inspect_weights.py                       # default d96h4 latest.pt
    python3 inspect_weights.py --ckpt logs/runs/<run>/best.pt
    python3 inspect_weights.py --top 8               # print top-8 sigmas too
    python3 inspect_weights.py --csv logs/spectra.csv
    python3 inspect_weights.py --plot logs/spectra   # per-matrix spectrum PNGs
    python3 inspect_weights.py --no-split-fused      # keep qkv/kv whole
    python3 inspect_weights.py --sort stable_rank    # order the table
"""

from __future__ import annotations

import argparse
import csv as csvmod
import math
import os

import torch

DEFAULT_CKPT = "logs/runs/extreme-full-3.8M-d96h4/latest.pt"


def split_fused(name: str, w: torch.Tensor):
    """Yield (subname, submatrix) for fused projections, else the matrix whole.

    ``qkv.weight`` is (3*d, d) stacked query/key/value row-blocks; ``kv.weight``
    is (2*d, d) stacked key/value. Everything else passes through untouched.
    """
    out, inp = w.shape
    if name.endswith("qkv.weight") and out == 3 * inp:
        q, k, v = w.chunk(3, dim=0)
        base = name[: -len("qkv.weight")]
        return [(base + "qkv.q", q), (base + "qkv.k", k), (base + "qkv.v", v)]
    if name.endswith("kv.weight") and out == 2 * inp:
        k, v = w.chunk(2, dim=0)
        base = name[: -len("kv.weight")]
        return [(base + "kv.k", k), (base + "kv.v", v)]
    return [(name, w)]


def spectrum(w: torch.Tensor) -> torch.Tensor:
    """Singular values of a 2D matrix, descending, as float64 on CPU."""
    return torch.linalg.svdvals(w.detach().to(torch.float64).cpu())


def metrics(sigma: torch.Tensor) -> dict:
    s = sigma.clamp_min(0)
    energy = s.pow(2)
    total_e = energy.sum()
    smax = s[0].item() if s.numel() else 0.0
    smin = s[s > 0].min().item() if (s > 0).any() else 0.0
    fro = math.sqrt(total_e.item())
    stable = (total_e / (s[0] ** 2)).item() if smax > 0 else 0.0
    # entropy (effective) rank on the sigma distribution
    p = s / s.sum() if s.sum() > 0 else s
    nz = p[p > 0]
    eff = math.exp(float(-(nz * nz.log()).sum())) if nz.numel() else 0.0
    # components needed for 90% / 99% of the energy
    cum = torch.cumsum(energy, 0) / total_e.clamp_min(1e-30)
    r90 = int((cum < 0.90).sum().item()) + 1
    r99 = int((cum < 0.99).sum().item()) + 1
    return {
        "n": s.numel(),
        "spec_norm": smax,
        "fro_norm": fro,
        "stable_rank": stable,
        "eff_rank": eff,
        "cond": (smax / smin) if smin > 0 else float("inf"),
        "r90": r90,
        "r99": r99,
    }


def iter_matrices(state_dict: dict, split: bool):
    """Yield (name, 2D-weight) for every matrix worth decomposing."""
    for name, w in state_dict.items():
        if not torch.is_tensor(w) or w.ndim != 2:
            continue  # skip biases, LayerNorm gains, scalars, buffers
        for sub, mat in (split_fused(name, w) if split else [(name, w)]):
            yield sub, mat


def plot_spectra(rows, outdir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping plots")
        return
    os.makedirs(outdir, exist_ok=True)
    for name, sigma, _m in rows:
        s = sigma.numpy()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.semilogy(range(1, len(s) + 1), s, marker=".", ms=3, lw=1)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("index")
        ax.set_ylabel("singular value (log)")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fname = name.replace("/", "_").replace(".", "_") + ".png"
        fig.savefig(os.path.join(outdir, fname), dpi=110)
        plt.close(fig)
    print(f"[plot] wrote {len(rows)} spectrum plots to {outdir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint .pt to inspect")
    ap.add_argument("--top", type=int, default=0,
                    help="also print the top-N singular values per matrix")
    ap.add_argument("--csv", default=None, help="write full spectra to this CSV")
    ap.add_argument("--plot", default=None, help="write per-matrix spectrum PNGs to this dir")
    ap.add_argument("--no-split-fused", dest="split", action="store_false",
                    help="keep fused qkv/kv projections as single matrices")
    ap.add_argument("--sort", default="name",
                    choices=["name", "stable_rank", "eff_rank", "spec_norm", "cond", "r99"],
                    help="order the report by this column")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    cfg = ck.get("model_config", {}) if isinstance(ck, dict) else {}
    step = ck.get("step") if isinstance(ck, dict) else None

    print(f"checkpoint : {args.ckpt}")
    if step is not None:
        print(f"step       : {step}")
    if cfg:
        print(f"config     : d_model={cfg.get('d_model')} n_heads={cfg.get('n_heads')} "
              f"d_ff={cfg.get('d_ff')} refine_steps={cfg.get('refine_steps')}")
    print()

    rows = []  # (name, sigma, metrics)
    for name, mat in iter_matrices(sd, args.split):
        sigma = spectrum(mat)
        rows.append((name, sigma, metrics(sigma)))

    keymap = {
        "name": lambda r: r[0],
        "stable_rank": lambda r: -r[2]["stable_rank"],
        "eff_rank": lambda r: -r[2]["eff_rank"],
        "spec_norm": lambda r: -r[2]["spec_norm"],
        "cond": lambda r: -r[2]["cond"],
        "r99": lambda r: -r[2]["r99"],
    }
    rows.sort(key=keymap[args.sort])

    hdr = f"{'matrix':44s} {'shape':>11s} {'spec':>8s} {'fro':>8s} " \
          f"{'stable':>7s} {'eff':>7s} {'cond':>9s} {'r90':>4s} {'r99':>4s}"
    print(hdr)
    print("-" * len(hdr))
    for name, sigma, m in rows:
        # sigma length is min(m, n): the largest possible rank of the matrix
        print(f"{name:44s} {('r<=' + str(m['n'])):>11s} "
              f"{m['spec_norm']:8.3f} {m['fro_norm']:8.3f} "
              f"{m['stable_rank']:7.2f} {m['eff_rank']:7.2f} "
              f"{m['cond']:9.2f} {m['r90']:4d} {m['r99']:4d}")
        if args.top:
            k = min(args.top, sigma.numel())
            vals = "  ".join(f"{v:.3f}" for v in sigma[:k].tolist())
            print(f"    top{k}: {vals}")

    # roll-ups
    print()
    print("legend: spec=sigma_max  fro=Frobenius  stable=||F||^2/sigma_max^2  "
          "eff=entropy rank  cond=sigma_max/sigma_min  r90/r99=components for 90%/99% energy")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            wr = csvmod.writer(f)
            wr.writerow(["matrix", "index", "singular_value"])
            for name, sigma, _ in rows:
                for i, v in enumerate(sigma.tolist()):
                    wr.writerow([name, i, v])
        print(f"[csv] wrote full spectra ({sum(len(r[1]) for r in rows)} rows) to {args.csv}")

    if args.plot:
        plot_spectra(rows, args.plot)


if __name__ == "__main__":
    main()
