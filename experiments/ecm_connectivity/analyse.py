#!/usr/bin/env python3
"""
analyse.py -- what happens to the ECM around the growing organoid, for each bond probability p.

Reads runs/ecm_p<p>_s<seed>/ (made by run_sweep.sh) and writes experiments/ecm_connectivity/results/:
  timeseries.png   cells, organoid radius, fiber alignment, fiber rotation, fiber movement vs time
  versus_growth.png  fiber alignment, rotation and movement vs organoid size (number of cells)
  profiles.png     the same quantities vs distance from the organoid surface, at the last time
  rotation_growth_only.png   fiber rotation counted from t = 10 (after the linkers have attached)
  summary.csv      one row per run: the numbers at the last time
  feedback_long.png  the runs continued to t = 130 (runs/long_ecm_p1.0_s<seed> vs runs/long_ecm_p0.7_s<seed>,
                   made with run_growth.py --start-from): p = 1.0 minus p = 0.7 in the same gel, over time

Quantities (see growthviz.ecm_structure; "near" = within 2 of the organoid surface, "far" = 4-8 away):
  order   S = (3 cos^2(theta) - 1) / 2, theta = angle between a fiber segment and the line from the
          organoid centre. +1: all segments point at the organoid (radial); 0: random;
          -0.5: all segments wrap around it (tangential).
  rot     how far each segment has turned since the reference time (degrees, 0-90)
  disp    how far each segment's midpoint has moved since the reference time
Every curve is the mean over the seeds; the band is +-1 standard deviation between seeds.

    python3 experiments/ecm_connectivity/analyse.py
"""
import csv
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
import growthviz as gv  # noqa: E402

OUT = ROOT / "experiments/ecm_connectivity/results"
P_VALUES = ["1.0", "0.9", "0.8", "0.7"]
COLORS = {"1.0": "#1b4f9c", "0.9": "#3d8fd1", "0.8": "#e08a2c", "0.7": "#c0392b"}


def load_runs():
    runs = {}
    for p in P_VALUES:
        for d in sorted(glob.glob(str(ROOT / f"runs/ecm_p{p}_s*"))):
            if os.path.isdir(d) and os.path.exists(f"{d}/frames.pkl"):
                with open(f"{d}/frames.pkl", "rb") as f:
                    runs.setdefault(p, []).append((Path(d), pickle.load(f)))
    return runs


def bond_info(p, seed):
    log = ROOT / f"experiments/ecm_connectivity/inputs/p{p}_seed{seed}/ECM_generation.log"
    z = eff = np.nan
    for line in open(log):
        if line.startswith("connectivity:"):
            z = float(line.split()[1])
        if line.startswith("p:"):
            eff = float(line.split()[1])
    return z, eff


def stack(frames_list, key):
    """seed x time array of one quantity (runs may differ in length: cut to the shortest)."""
    n = min(len(fr) for fr in frames_list)
    t = np.array([frames_list[0][i]["t"] for i in range(n)])
    y = np.array([[fr[i].get(key, np.nan) for i in range(n)] for fr in frames_list], float)
    return t, y


def band(ax, t, y, color, label):
    m, sd = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
    ax.plot(t, m, color=color, lw=2, label=label)
    if y.shape[0] > 1:
        ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.18, lw=0)


def timeseries(runs):
    keys = [("n_cells", "number of cells"), ("surface_radius", "organoid radius"),
            ("n_links", "cell-ECM linkers"), ("mean_abs_fiber_strain_near", "mean |fiber stretch| near the organoid"),
            ("order_near", "fiber alignment S, within 2 of surface\n(+1 radial, 0 random, -0.5 wrapping)"),
            ("rot_near", "fiber rotation since t = 0 (deg), within 2"),
            ("disp_near", "fiber movement since t = 0, within 2"),
            ("order_far", "fiber alignment S, 4-8 from surface (control)")]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), dpi=130)
    for ax, (key, title) in zip(axes.flat, keys):
        for p in P_VALUES:
            if p in runs:
                t, y = stack([fr for _, fr in runs[p]], key)
                band(ax, t, y, COLORS[p], f"p = {p}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Growing organoid in gels of different connectivity p (mean of 3 gels, band = +-1 SD)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "timeseries.png")
    plt.close(fig)


def versus_growth(runs):
    """Fiber alignment, rotation and movement against organoid size instead of time."""
    keys = [("order_near", "fiber alignment S within 2 of surface"), ("rot_near", "fiber rotation since t = 0 (deg)"),
            ("disp_near", "fiber movement since t = 0")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=130)
    for ax, (key, title) in zip(axes, keys):
        for p in P_VALUES:
            if p not in runs:
                continue
            for k, (_, fr) in enumerate(runs[p]):
                x = [f["n_cells"] for f in fr]
                y = [f.get(key, np.nan) for f in fr]
                ax.plot(x, y, color=COLORS[p], lw=1.4, alpha=0.85, label=f"p = {p}" if k == 0 else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("number of cells (organoid size)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("The same quantities against organoid size (each line = one gel)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "versus_growth.png")
    plt.close(fig)


def profiles(runs):
    mids = 0.5 * (gv.SHELL_EDGES[1:] + gv.SHELL_EDGES[:-1])
    keys = [("order", "fiber alignment S"), ("rot", "rotation since t = 0 (deg)"),
            ("disp", "movement since t = 0"), ("density", "fiber segments per unit volume")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3), dpi=130)
    for ax, (key, title) in zip(axes, keys):
        for p in P_VALUES:
            if p not in runs:
                continue
            ys = []
            for _, fr in runs[p]:
                last = fr[-1]
                prof = last["ecm_profile"]
                if key == "density":
                    R = last["surface_radius"]
                    r_in, r_out = R + gv.SHELL_EDGES[:-1], R + gv.SHELL_EDGES[1:]
                    vol = 4.0 / 3.0 * np.pi * (np.clip(r_out, 0, None) ** 3 - np.clip(r_in, 0, None) ** 3)
                    ys.append(prof["n"] / vol)
                else:
                    ys.append(prof[key])
            ys = np.array(ys, float)
            m, sd = np.nanmean(ys, axis=0), np.nanstd(ys, axis=0)
            ax.errorbar(mids[1:], m[1:], yerr=sd[1:], color=COLORS[p], marker="o", ms=4, capsize=2, label=f"p = {p}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("distance outside the organoid surface")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Fibers around the organoid at t = {runs[P_VALUES[0]][0][1][-1]['t']:g} (mean of 3 gels)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "profiles.png")
    plt.close(fig)


def rotation_since(run_dir, frames, t_ref=10):
    """Fiber rotation and movement counted from t_ref, using the kept ECM snapshots."""
    ref_file = run_dir / f"seg_{t_ref:07d}/{t_ref:07d}.ECM.topo"
    if not ref_file.exists():
        return None
    ref = gv.reference_nodes(str(ref_file))
    by_t = {int(round(fr["t"])): fr for fr in frames}
    out = []
    for f in sorted(run_dir.glob("seg_*/0*.ECM.topo")):
        t = int(f.name[:7])
        if t <= t_ref or t not in by_t:
            continue
        nodes, _, fibers = gv.read_ecm(str(f))
        fr = by_t[t]
        _, summ = gv.ecm_structure(nodes, fibers, fr["center"], fr["surface_radius"], ref)
        out.append((t, summ["rot_near"], summ["rot_far"], summ["disp_near"]))
    return np.array(out)


def growth_only(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=130)
    table = {}
    for p in P_VALUES:
        if p not in runs:
            continue
        curves = [rotation_since(d, fr) for d, fr in runs[p]]
        curves = [c for c in curves if c is not None and len(c)]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        t = curves[0][:n, 0]
        for ax, col, ls in ((axes[0], 1, "-"), (axes[0], 2, ":"), (axes[1], 3, "-")):
            y = np.array([c[:n, col] for c in curves])
            m = y.mean(axis=0)
            ax.plot(t, m, ls, color=COLORS[p], lw=2, label=f"p = {p}" if (col != 2) else None)
            if y.shape[0] > 1:
                ax.fill_between(t, m - y.std(axis=0), m + y.std(axis=0), color=COLORS[p], alpha=0.15, lw=0)
        table[p] = [c[n - 1] for c in curves]
    axes[0].set_title("fiber rotation since t = 10 (deg)\nsolid: within 2 of surface, dotted: 4-8 away", fontsize=10)
    axes[1].set_title("fiber movement since t = 10, within 2 of surface", fontsize=10)
    for ax in axes:
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "rotation_growth_only.png")
    plt.close(fig)
    return table


def long_runs():
    """Does the gel feed back on growth? Same gel (seed), p = 1.0 vs p = 0.7, runs continued to t = 130.
    The same seed gives the same starting tissue, nearly nested gels and the same Python random numbers
    (the engine's thermal noise is NOT shared: it is seeded from std::random_device), so the difference within a
    gel isolates the effect of p."""
    pairs = []
    for d in sorted(glob.glob(str(ROOT / "runs/long_ecm_p1.0_s*"))):
        seed = d.rsplit("_s", 1)[-1]
        d07 = ROOT / f"runs/long_ecm_p0.7_s{seed}"
        if not (os.path.exists(f"{d}/frames.pkl") and (d07 / "frames.pkl").exists()):
            continue
        with open(f"{d}/frames.pkl", "rb") as f:
            a = {round(x["t"]): x for x in pickle.load(f)}
        with open(d07 / "frames.pkl", "rb") as f:
            b = {round(x["t"]): x for x in pickle.load(f)}
        pairs.append((seed, a, b))
    if not pairs:
        return
    panels = [("n_cells", "number of cells: p = 1.0 vs p = 0.7 in the same gel (% difference)",
               lambda x, y: 100.0 * (x - y) / y),
              ("mean_P", "mean cell pressure: p = 1.0 minus p = 0.7 in the same gel", lambda x, y: x - y),
              ("mean_abs_fiber_strain_near", "mean |fiber stretch| near the organoid", None)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), dpi=130)
    for ax, (key, title, diff) in zip(axes, panels):
        for k, (seed, a, b) in enumerate(pairs):
            t = sorted(set(a) & set(b))
            if diff is None:
                ax.plot(t, [a[i][key] for i in t], color=COLORS["1.0"], lw=1.4, alpha=0.8,
                        label="p = 1.0" if k == 0 else None)
                ax.plot(t, [b[i][key] for i in t], color=COLORS["0.7"], lw=1.4, alpha=0.8,
                        label="p = 0.7" if k == 0 else None)
            else:
                ax.plot(t, [diff(a[i][key], b[i][key]) for i in t], lw=1.6, label=f"gel {seed}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Runs continued to t = 130: does the better-connected gel push back on the organoid?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "feedback_long.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    long_runs()
    runs = load_runs()
    if not runs:
        sys.exit("no runs found in runs/ecm_p*_s*")
    timeseries(runs)
    versus_growth(runs)
    profiles(runs)
    since10 = growth_only(runs)
    rows = []
    for p in P_VALUES:
        for k, (d, fr) in enumerate(runs.get(p, [])):
            seed = int(d.name.split("_s")[-1])
            z, eff = bond_info(p, seed)
            last, first = fr[-1], fr[0]
            r10 = since10.get(p, [None] * 3)[k] if p in since10 and k < len(since10[p]) else None
            rows.append(dict(p=p, effective_p=eff, bonds_per_node=z, seed=seed, t_end=last["t"],
                             cells=last["n_cells"], organoid_radius=round(last["surface_radius"], 3),
                             order_near_start=round(first["order_near"], 3), order_near_end=round(last["order_near"], 3),
                             order_far_end=round(last["order_far"], 3),
                             rot_near_since0=round(last["rot_near"], 2), rot_far_since0=round(last["rot_far"], 2),
                             disp_near_since0=round(last["disp_near"], 3),
                             rot_near_since10=round(float(r10[1]), 2) if r10 is not None else "",
                             disp_near_since10=round(float(r10[3]), 3) if r10 is not None else "",
                             mean_P_end=round(last["mean_P"], 3), linkers_end=last["n_links"],
                             fiber_points_inside=last["fiber_points_inside"]))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}/timeseries.png, profiles.png, rotation_growth_only.png, summary.csv")


if __name__ == "__main__":
    main()
