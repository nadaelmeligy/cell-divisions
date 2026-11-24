#!/usr/bin/env python3
"""
half_gel.py -- does an organoid in a gel that is well connected on one half and loose on the other move
toward one side? Gels: experiments/ecm_connectivity/inputs/half_<L>left_<R>right_seed<s>/ (ECM64.py --p <R>
--p-left <L>). Each seed is run twice, mirrored: A = p0.7left_p1.0right (stiff half at +x) and
B = p1.0left_p0.7right (stiff half at -x).

Why mirror pairs: every run also drifts for reasons that have nothing to do with the stiff half (the
organoid's own start, engine noise, a box-fixed drift common to all runs). Per seed,
  g = (dx_A - dx_B) / 2    flips with the gel: the gel effect (+ = toward the stiff half)
  s = (dx_A + dx_B) / 2    does not flip: the part that comes from the tissue and the box
The same pair statistic along y and z (where the gel is the same on both sides) is the noise null.

Position = volume-weighted centroid of the real cells ('center_vol' in frames.pkl, or computed from the
segment's topology for older runs). The frames' 'center' counts each vertex once per (cell, face, edge)
and jumps at divisions, so it is not used.

Also per run, averaged over a time window: surface-cell pressure on the stiff vs soft side (one value per
surface cell, side from its surface faces) and linker tension ks (length - rest length) on each side
(rest length from frames.pkl 'link_rest', or from the run's shrink schedule).

Writes experiments/ecm_connectivity/results/half_gel.png and prints the tables.
    python3 experiments/ecm_connectivity/half_gel.py
"""
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/ecm_connectivity/results"
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
import growthviz as gv  # noqa: E402
import run_growth as rg  # noqa: E402

LINK_KS = 10.0
T_STEP = 5   # volume centroids are computed every T_STEP time units for runs without center_vol


def load(run):
    try:
        with open(ROOT / "runs" / run / "frames.pkl", "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, NotADirectoryError, EOFError, pickle.UnpicklingError):
        return None


def topo_file(run, t):
    """The topology the frame at time t was made from (this run's segment, or an ancestor's)."""
    d = ROOT / "runs" / run
    while True:
        f = d / f"seg_{t:07d}" / f"{t:07d}.sample.topo"
        if f.exists():
            return f
        cfg = json.loads((d / "config.json").read_text())
        if not cfg.get("start_from"):
            return None
        d = Path(cfg["start_from"])
        d = d if d.is_absolute() else ROOT / d


def volume_centroid(path):
    V, E, P, C = gv.read_topo(path)
    geo = [gv.cell_geometry(cid, V, E, P, C)[:2] for cid, (_, typ) in C.items() if typ > 0]
    return np.average([g[1] for g in geo], axis=0, weights=[g[0] for g in geo])


def track(run, frames):
    """{t: volume centroid} at t = 1 and every T_STEP."""
    out = {}
    for f in frames:
        t = int(round(f["t"]))
        if t != 1 and t % T_STEP:
            continue
        if "center_vol" in f:
            out[t] = np.asarray(f["center_vol"])
        else:
            path = topo_file(run, t)
            if path is not None:
                out[t] = volume_centroid(path)
    return out


def side_means(run, frames, s, t0, t1):
    """Mean over frames t0 < t <= t1 of: surface-cell P on stiff and soft side (one value per cell),
    linker tension on each side."""
    cfg = json.loads((ROOT / "runs" / run / "config.json").read_text())
    cfg["link"] = tuple(cfg["link"])
    Pst, Pso, Tst, Tso = [], [], [], []
    for f in frames:
        if not (t0 < f["t"] <= t1):
            continue
        c = np.asarray(f.get("center_vol", f["center"]))
        acc = {}
        for cid, pts in f["surf"]:
            acc.setdefault(cid, []).append(pts.mean(axis=0))
        side = {cid: np.sign(np.mean(ps, axis=0)[0] - c[0]) * s for cid, ps in acc.items()}
        Pst.append(np.mean([f["cells"][k]["P"] for k, v in side.items() if v > 0]))
        Pso.append(np.mean([f["cells"][k]["P"] for k, v in side.items() if v < 0]))
        L = np.asarray(f["links"])
        if len(L):
            rest = f.get("link_rest", rg.link_rest_length(cfg, f["t"], end_of_segment=True))
            ten = LINK_KS * (np.linalg.norm(L[:, 1] - L[:, 0], axis=1) - rest)
            st = np.sign(L[:, 0, 0] - c[0]) * s > 0
            Tst.append(ten[st].mean())
            Tso.append(ten[~st].mean())
    return np.mean(Pst) - np.mean(Pso), np.mean(Tst), np.mean(Tso)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted({os.path.basename(d) for d in glob.glob(str(ROOT / "runs/half*_p*left_p*right_s*"))
                   if os.path.isdir(d)})
    tracks, frames_of = {}, {}
    for run in runs:
        fr = load(run)
        if fr:
            frames_of[run] = fr
            tracks[run] = track(run, fr)
    # mirror pairs: same prefix and seed, A = stiff at +x, B = stiff at -x
    pairs = {}
    for run in tracks:
        m = re.match(r"(.*)_p([\d.]+)left_p([\d.]+)right_s(\d+)$", run)
        prefix, left, right, seed = m.group(1), float(m.group(2)), float(m.group(3)), m.group(4)
        pairs.setdefault((prefix, seed), {})["A" if right > left else "B"] = run
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=120)
    rows = []
    for (prefix, seed), ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            continue
        ta, tb = tracks[ab["A"]], tracks[ab["B"]]
        ts = sorted(set(ta) & set(tb))
        d = {ax: (np.array([ta[t][ax] - ta[1][ax] for t in ts]), np.array([tb[t][ax] - tb[1][ax] for t in ts]))
             for ax in range(3)}
        g = {ax: (d[ax][0] - d[ax][1]) / 2 for ax in range(3)}
        ls = "-" if "pull" not in prefix else "--"
        lab = f"{prefix} gel {seed}"
        axes[0].plot(ts, g[0], ls, lw=1.8, label=lab)
        axes[1].plot(ts, (d[0][0] + d[0][1]) / 2, ls, lw=1.4, label=lab)
        for ax in (1, 2):
            axes[2].plot(ts, g[ax], ls, lw=0.8, color="0.6")
        for t in (20, 40, 80, 130):
            if t in ts:
                i = ts.index(t)
                rows.append(dict(prefix=prefix, seed=seed, t=t, g=g[0][i], s=(d[0][0][i] + d[0][1][i]) / 2,
                                 gy=g[1][i], gz=g[2][i]))
    axes[0].set_title("gel effect g = (dx_A - dx_B)/2: drift toward the stiff half", fontsize=10)
    axes[1].set_title("tissue + box part (dx_A + dx_B)/2 (does not flip)", fontsize=10)
    axes[2].set_title("noise null: the same pair statistic along y and z", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Organoid in a gel that is well connected (p = 1.0) on one half and loose (p = 0.7) on the other "
                 "(volume centroid; dashed: cells pull from t = 80)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "half_gel.png")
    plt.close(fig)

    print("prefix | gel | t | g (toward stiff) | s (tissue+box) | g_y | g_z")
    for r in rows:
        print(f"{r['prefix']} | {r['seed']} | {r['t']} | {r['g']:+.3f} | {r['s']:+.3f} | {r['gy']:+.3f} | {r['gz']:+.3f}")
    for t in (20, 40, 80):
        g = np.array([r["g"] for r in rows if r["t"] == t and r["prefix"] == "half"])
        null = np.array([v for r in rows if r["t"] == t and r["prefix"] == "half" for v in (r["gy"], r["gz"])])
        if len(g) >= 2:
            se = g.std(ddof=1) / np.sqrt(len(g))
            tcrit = stats.t.ppf(0.975, len(g) - 1)
            z = g.mean() / (np.sqrt(np.mean(null ** 2)) / np.sqrt(len(g)))
            print(f"t = {t}: {len(g)} pairs, mean g {g.mean():+.3f}, 95% CI [{g.mean() - tcrit * se:+.3f}, "
                  f"{g.mean() + tcrit * se:+.3f}] (t-based); y/z null rms {np.sqrt(np.mean(null ** 2)):.3f} -> z = {z:.1f}")
    print("\nrun | window | surface P stiff - soft (per cell) | linker tension stiff / soft")
    for run, fr in sorted(frames_of.items()):
        s = 1 if re.search(r"p0\.7left_p1\.0right", run) else -1
        for (t0, t1) in ((1, 40), (40, 80), (100, 130)):
            if fr[-1]["t"] >= t1:
                dP, Ts, To = side_means(run, fr, s, t0, t1)
                print(f"{run} | {t0}-{t1} | {dP:+.3f} | {Ts:.3f} / {To:.3f}")


if __name__ == "__main__":
    main()
