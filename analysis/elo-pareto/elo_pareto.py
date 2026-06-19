# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "pandas", "matplotlib", "seaborn"]
# ///
"""Fit Elo from the versus tournament's pairwise matrix and plot Elo vs compute.

Inputs (written by `cargo run -p tetr-research --example elo_pareto`):
  configs.csv : label,width,depth,compute_ms,nodes   (the compute x-axis)
  pairs.csv   : a,b,a_wins,b_wins,draws,games         (the pairwise outcomes)

Model: Bradley-Terry. Config i has strength theta_i; P(i beats j) = sigmoid(theta_i-theta_j).
A draw counts as half a win to each side. A small L2 ridge regularizes perfectly-separated
edges (cross-strength matchups that go 24-0) and fixes the additive gauge. Elo = theta *
400/ln(10) (a 400-point gap == 10:1 odds), anchored so the weakest config sits at 0.
95% CIs come from a multinomial bootstrap over each edge's games.

Run:  uv run analysis/elo-pareto/elo_pareto.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import expit, log_expit

HERE = pathlib.Path(__file__).resolve().parent
ELO_SCALE = 400.0 / np.log(10.0)
RIDGE = 1e-3


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = pd.read_csv(HERE / "configs.csv")
    pairs = pd.read_csv(HERE / "pairs.csv")
    # keep only pairs whose endpoints both have a compute row
    known = set(cfg.label)
    pairs = pairs[pairs.a.isin(known) & pairs.b.isin(known)].reset_index(drop=True)
    return cfg, pairs


def fit_bt(labels: list[str], pairs: pd.DataFrame) -> np.ndarray:
    """Maximum-likelihood Bradley-Terry strengths (in nats), gauge-fixed by the ridge."""
    idx = {lab: k for k, lab in enumerate(labels)}
    i = pairs.a.map(idx).to_numpy()
    j = pairs.b.map(idx).to_numpy()
    si = pairs.a_wins.to_numpy(float) + 0.5 * pairs.draws.to_numpy(float)  # eff. wins for i
    sj = pairs.b_wins.to_numpy(float) + 0.5 * pairs.draws.to_numpy(float)  # eff. wins for j
    n = len(labels)

    def nll_grad(theta: np.ndarray):
        d = theta[i] - theta[j]
        nll = -(si * log_expit(d) + sj * log_expit(-d)).sum() + RIDGE * theta @ theta
        p = expit(d)
        g_edge = (si + sj) * p - si  # d nll / d d
        grad = np.zeros(n)
        np.add.at(grad, i, g_edge)
        np.add.at(grad, j, -g_edge)
        grad += 2.0 * RIDGE * theta
        return nll, grad

    res = minimize(nll_grad, np.zeros(n), jac=True, method="L-BFGS-B")
    return res.x - res.x.mean()


def bootstrap(labels, pairs, reps=400, seed=0) -> np.ndarray:
    """Multinomial-resample each edge's games; refit. Returns (reps, n_configs) Elo draws."""
    rng = np.random.default_rng(seed)
    out = np.empty((reps, len(labels)))
    a_w = pairs.a_wins.to_numpy()
    b_w = pairs.b_wins.to_numpy()
    dr = pairs.draws.to_numpy()
    tot = (a_w + b_w + dr).clip(min=1)
    p = np.stack([a_w / tot, b_w / tot, dr / tot], axis=1)
    for r in range(reps):
        res = np.array([rng.multinomial(t, pr) for t, pr in zip(tot, p)])
        bp = pairs.copy()
        bp.a_wins, bp.b_wins, bp.draws = res[:, 0], res[:, 1], res[:, 2]
        out[r] = fit_bt(labels, bp) * ELO_SCALE
    return out


def pareto_front(compute: np.ndarray, elo: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points: no other point is cheaper AND stronger."""
    order = np.argsort(compute)
    front, best = [], -np.inf
    for k in order:
        if elo[k] > best + 1e-9:
            front.append(k)
            best = elo[k]
    return np.array(front)


def elbow(compute: np.ndarray, elo: np.ndarray) -> int:
    """Kneedle elbow on a concave frontier: the point farthest above the chord joining the
    cheapest and most-expensive frontier points, in normalized (log-compute, Elo) space.
    This is the diminishing-returns knee — beyond it, strength costs disproportionate compute."""
    x = np.log10(compute)
    x = (x - x.min()) / max(np.ptp(x), 1e-9)
    y = (elo - elo.min()) / max(np.ptp(elo), 1e-9)
    chord = y[0] + (y[-1] - y[0]) / (x[-1] - x[0]) * (x - x[0])
    return int(np.argmax(y - chord))


def main() -> int:
    cfg, pairs = load()
    if pairs.empty:
        print("pairs.csv is empty — run the tournament first", file=sys.stderr)
        return 1

    # connectivity check (BT needs the comparison graph connected)
    labels = list(cfg.label)
    import scipy.sparse.csgraph as cg
    from scipy.sparse import coo_matrix

    idx = {lab: k for k, lab in enumerate(labels)}
    ii = pairs.a.map(idx).to_numpy()
    jj = pairs.b.map(idx).to_numpy()
    adj = coo_matrix((np.ones(len(ii) * 2), (np.r_[ii, jj], np.r_[jj, ii])), shape=(len(labels),) * 2)
    n_comp, comp = cg.connected_components(adj, directed=False)
    main_comp = np.bincount(comp).argmax()
    keep = comp == main_comp
    if not keep.all():
        print(f"warning: {(~keep).sum()} configs not yet connected — fitting the main component "
              f"({keep.sum()}/{len(labels)})", file=sys.stderr)
    cfg = cfg[keep].reset_index(drop=True)
    labels = list(cfg.label)
    known = set(labels)
    pairs = pairs[pairs.a.isin(known) & pairs.b.isin(known)].reset_index(drop=True)

    theta = fit_bt(labels, pairs)
    elo = theta * ELO_SCALE
    elo = elo - elo.min()  # anchor weakest config at 0
    boot = bootstrap(labels, pairs)
    boot = boot - boot.min(axis=1, keepdims=True)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    cfg = cfg.assign(elo=elo, elo_lo=lo, elo_hi=hi)
    games = int(pairs.games.sum())
    front = pareto_front(cfg.compute_ms.to_numpy(), cfg.elo.to_numpy())
    cfg["on_front"] = False
    cfg.loc[front, "on_front"] = True
    cfg.to_csv(HERE / "elo.csv", index=False)  # the fitted strengths, for scaling_analysis.py

    # ---- summary ----
    print(f"\nfit on {len(cfg)} configs, {len(pairs)} matchups, {games} games\n")
    fc = cfg.iloc[front].sort_values("compute_ms")
    print("Pareto frontier (compute-efficient configs):")
    print(fc[["label", "width", "depth", "compute_ms", "nodes", "elo"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    champ = cfg.loc[cfg.compute_ms.idxmax()]
    # the knee = the diminishing-returns elbow of the frontier
    fc_sorted = fc.sort_values("compute_ms").reset_index(drop=True)
    knee = fc_sorted.iloc[elbow(fc_sorted.compute_ms.to_numpy(), fc_sorted.elo.to_numpy())]
    print(f"\nknee (elbow): {knee.label} = {knee.elo:.0f} Elo "
          f"({knee.elo - champ.elo:+.0f} vs champion {champ.label}) at {knee.compute_ms:.1f} ms "
          f"= {champ.compute_ms / knee.compute_ms:.1f}x less compute than the champion.")

    # ---- plot ----
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 9))
    depths = sorted(cfg.depth.unique())
    palette = dict(zip(depths, sns.color_palette("viridis", len(depths))))

    # CI whiskers (thin)
    ax.vlines(cfg.compute_ms, cfg.elo_lo, cfg.elo_hi, color="0.7", lw=1.0, zorder=1)
    # all configs: color = depth, size = width
    for _, r in cfg.iterrows():
        ax.scatter(r.compute_ms, r.elo, s=30 + r.width * 1.6, color=palette[r.depth],
                   edgecolor="white", lw=0.6, zorder=3,
                   alpha=0.95 if r.on_front else 0.55)
    # frontier line + emphasis
    ax.plot(fc.compute_ms, fc.elo, color="crimson", lw=2.2, zorder=2, alpha=0.9,
            label="Pareto frontier")
    ax.scatter(fc.compute_ms, fc.elo, s=70 + fc.width * 1.6, facecolor="none",
               edgecolor="crimson", lw=2.0, zorder=4)

    # label frontier configs + the champion
    for n, (_, r) in enumerate(fc.sort_values("compute_ms").iterrows()):
        ax.annotate(r.label, (r.compute_ms, r.elo), textcoords="offset points",
                    xytext=(5, 7 if n % 2 == 0 else -15), fontsize=8.5,
                    color="crimson", weight="bold")
    ax.annotate(f"champion\n{champ.label}", (champ.compute_ms, champ.elo),
                textcoords="offset points", xytext=(-10, -38), fontsize=11,
                ha="center", color="black",
                arrowprops=dict(arrowstyle="->", color="0.4"))

    # champion-strength reference + the diminishing-returns elbow
    ax.axhline(champ.elo, ls="--", lw=1.2, color="0.6", zorder=0)
    ax.text(cfg.compute_ms.min(), champ.elo + 8, f"champion strength ({champ.elo:.0f} Elo)",
            fontsize=10, color="0.45")
    ax.scatter([knee.compute_ms], [knee.elo], marker="*", s=720, color="gold",
               edgecolor="black", lw=1.3, zorder=6)
    ax.annotate(
        f"knee: {knee.label}\n{knee.elo:.0f} Elo  ({knee.elo - champ.elo:+.0f} vs champion)\n"
        f"{champ.compute_ms / knee.compute_ms:.1f}x less compute",
        (knee.compute_ms, knee.elo), textcoords="offset points", xytext=(14, -72),
        fontsize=11, weight="bold", color="darkgoldenrod",
        arrowprops=dict(arrowstyle="->", color="darkgoldenrod"))

    ax.set_xscale("log")
    ax.set_xlabel("compute  (ms / decision, native release — log scale)")
    ax.set_ylabel("Elo  (Bradley-Terry, vs weakest config)")
    ax.set_title(f"Beam search-shape Pareto frontier: Elo vs compute\n"
                 f"{len(cfg)} (width, depth) configs · {games} rain-decisive games · "
                 f"TP-beam / attack-tuned CC2", fontsize=15)

    # depth legend (color) + width legend (size)
    from matplotlib.lines import Line2D
    dlg = [Line2D([0], [0], marker="o", ls="", mfc=palette[d], mec="white",
                  ms=11, label=f"depth {d}") for d in depths]
    wlg = [Line2D([0], [0], marker="o", ls="", mfc="0.5", mec="white",
                  ms=np.sqrt(30 + w * 1.6) / 1.4, label=f"w{w}")
           for w in [4, 16, 64, 128]]
    leg1 = ax.legend(handles=dlg, title="depth (color)", loc="lower right", fontsize=11)
    ax.add_artist(leg1)
    ax.legend(handles=wlg + [Line2D([0], [0], color="crimson", lw=2.2, label="Pareto frontier")],
              title="width (size)", loc="upper left", fontsize=11)

    fig.tight_layout()
    out = HERE / "elo_pareto.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
