#!/usr/bin/env python3
"""
feedback_screen.py -- looks for gel -> tissue feedback in runs that differ only in the gel (p = 1.0 vs 0.7,
same seed) under different cell rules. For every saved time it measures, from runs/<run>/frames.pkl:

  P_surf, P_in     mean pressure P = -2 kv (V - V0) of surface cells (a face on the medium) and inner cells
  grow_surf/in     mean growth factor max(0, 1 - P / P_h) (the growth rate as a fraction of the maximum)
  rough            area-weighted rms of (r / R - 1) over the surface faces (r = face centre to organoid centre)
  sh2, sh3, sh4+   rms shape amplitude of spherical-harmonic degree l = 2 (elongation), 3 (three lobes) and
                   4-8 (bumps), fitted to r / R - 1 over the surface faces. l = 1 (a shift) is left out
  link_tension     mean linker tension ks (length - rest length), with the engine's rest length at that time
                   (it shrinks when cells pull; frames.pkl 'link_strain' of older runs is against 1.5)
  fiber_stretch    mean |fiber stretch| near the organoid (from frames.pkl)

Writes experiments/ecm_connectivity/results/feedback_screen.png (one column per cell rule, p = 1.0 blue vs
p = 0.7 red) and feedback_screen.csv (all times, all runs).

    python3 experiments/ecm_connectivity/feedback_screen.py
"""
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/ecm_connectivity/results"
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
sys.path.insert(0, str(ROOT))
import run_growth as rg  # noqa: E402

try:                                   # scipy >= 1.15
    from scipy.special import sph_harm_y

    def _Y(l, m, theta, phi):          # theta = polar angle, phi = azimuth
        return sph_harm_y(l, m, theta, phi)
except ImportError:                    # older scipy
    from scipy.special import sph_harm

    def _Y(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)

RULES = {   # column title: run name pattern ({p} = 1.0 or 0.7)
    "baseline (P_h = 6)": "long_ecm_p{p}_s7",
    "pressure-sensitive (P_h = 3)": "fb_ph3_p{p}_s7",
    "cells pull (linkers shrink)": "fb_pull_p{p}_s7",
    "P_h = 3 + cells pull": "fb_ph3pull_p{p}_s7",
}
COLORS = {"1.0": "#1b4f9c", "0.7": "#c0392b"}
LMAX = 8


def real_sh_basis(theta, phi, lmax=LMAX):
    """Orthonormal real spherical harmonics, columns ordered l = 0..lmax, m = -l..l."""
    cols, degree = [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y = _Y(l, abs(m), theta, phi)
            cols.append(np.sqrt(2) * (y.imag if m < 0 else y.real) if m != 0 else y.real)
            degree.append(l)
    return np.array(cols).T, np.array(degree)


def polygon_area(pts):
    c = pts.mean(axis=0)
    a = 0.0
    for i in range(len(pts)):
        a += 0.5 * np.linalg.norm(np.cross(pts[i] - c, pts[(i + 1) % len(pts)] - c))
    return a


def shape(fr):
    cen = np.asarray(fr["center"])
    c = np.array([pts.mean(axis=0) for _, pts in fr["surf"]]) - cen
    w = np.array([polygon_area(pts) for _, pts in fr["surf"]])
    r = np.linalg.norm(c, axis=1)
    R = np.sum(w * r) / np.sum(w)
    y = r / R - 1.0
    rough = float(np.sqrt(np.sum(w * y * y) / np.sum(w)))
    theta = np.arccos(np.clip(c[:, 2] / r, -1, 1))
    phi = np.arctan2(c[:, 1], c[:, 0])
    A, deg = real_sh_basis(theta, phi)
    # area-weighted least squares; faces cover the sphere, so sum(w) ~ 4 pi R^2
    Wa = w / np.sum(w) * 4 * np.pi
    coef = np.linalg.lstsq(A * np.sqrt(Wa)[:, None], y * np.sqrt(Wa), rcond=None)[0]
    amp = {l: float(np.sqrt(np.sum(coef[deg == l] ** 2) / (4 * np.pi))) for l in range(LMAX + 1)}
    return rough, amp


def _parent(d, cfg):
    """(parent folder, time this run started from it) for a run made with --start-from, else None."""
    if not cfg.get("start_from"):
        return None
    src = Path(cfg["start_from"])
    src = src if src.is_absolute() else ROOT / src
    return src, json.loads((src / "state.json").read_text())["t"]


def parent_cfg_at(d, cfg, t):
    """The config that made the frame at time t (the run's own, or an ancestor's for inherited history)."""
    par = _parent(d, cfg)
    while par is not None and t <= par[1] + 1e-9:
        d = par[0]
        cfg = json.loads((d / "config.json").read_text())
        cfg["link"] = tuple(cfg["link"])
        par = _parent(d, cfg)
    return cfg


def parent_P_h(d, cfg):
    return lambda t: parent_cfg_at(d, cfg, t).get("P_h", 6.0)


def metrics(run):
    d = ROOT / "runs" / run
    if not (d / "frames.pkl").exists():
        return None
    cfg = json.loads((d / "config.json").read_text())
    cfg["link"] = tuple(cfg["link"])
    # a continued run's frames.pkl starts with its parent's history, which used the parent's P_h
    P_h_of = parent_P_h(d, cfg)
    try:
        with open(d / "frames.pkl", "rb") as f:
            frames = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):   # a running run is rewriting it: skip this time
        return None
    rows = []
    for fr in frames:
        if fr["t"] < 60:
            continue
        surf_ids = {cid for cid, _ in fr["surf"]}
        Ps = np.array([c["P"] for k, c in fr["cells"].items() if k in surf_ids])
        Pi = np.array([c["P"] for k, c in fr["cells"].items() if k not in surf_ids])
        P_h = P_h_of(fr["t"])
        g = lambda P: float(np.mean(np.clip(1 - P / P_h, 0, None))) if len(P) else np.nan
        L = np.asarray(fr["links"])
        rest = fr.get("link_rest", rg.link_rest_length(parent_cfg_at(d, cfg, fr["t"]), fr["t"], end_of_segment=True))
        tension = 10.0 * (np.linalg.norm(L[:, 1] - L[:, 0], axis=1) - rest) if len(L) else np.array([])
        rough, amp = shape(fr)
        rows.append(dict(run=run, t=fr["t"], n_cells=fr["n_cells"], P_h=P_h,
                         P_surf=float(Ps.mean()), P_in=float(Pi.mean()) if len(Pi) else np.nan,
                         grow_surf=g(Ps), grow_in=g(Pi), rough=rough, sh2=amp[2], sh3=amp[3],
                         sh4_8=float(np.sqrt(sum(amp[l] ** 2 for l in range(4, LMAX + 1)))),
                         link_tension=float(tension.mean()) if len(tension) else np.nan,
                         fiber_stretch=fr["mean_abs_fiber_strain_near"], n_links=fr["n_links"]))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for title, pat in RULES.items():
        for p in ("1.0", "0.7"):
            rows = metrics(pat.format(p=p))
            if rows:
                data[(title, p)] = rows
    if not data:
        sys.exit("no runs found")
    keys = [("n_cells", "number of cells"), ("P_surf", "surface-cell pressure"), ("P_in", "inner-cell pressure"),
            ("grow_surf", "surface growth factor 1 - P/P_h"), ("grow_in", "inner growth factor 1 - P/P_h"),
            ("rough", "surface roughness (rms of r/R - 1)"), ("sh2", "shape: elongation (l = 2)"),
            ("sh4_8", "shape: bumps (l = 4-8)"), ("link_tension", "mean linker tension"),
            ("fiber_stretch", "fiber stretch near the organoid")]
    titles = [t for t in RULES if any((t, p) in data for p in ("1.0", "0.7"))]
    fig, axes = plt.subplots(len(keys), len(titles), figsize=(4.2 * len(titles), 2.1 * len(keys)), dpi=110,
                             squeeze=False, sharex=True)
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=10)
        for i, (key, lab) in enumerate(keys):
            ax = axes[i, j]
            for p in ("1.0", "0.7"):
                rows = data.get((title, p))
                if rows:
                    ax.plot([r["t"] for r in rows], [r[key] for r in rows], color=COLORS[p], lw=1.5,
                            label=f"p = {p}")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
        axes[-1, j].set_xlabel("time")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Gel 7: p = 1.0 (blue) vs p = 0.7 (red), from t = 80 the cell rule of each column", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "feedback_screen.png")
    plt.close(fig)
    allrows = [r for rows in data.values() for r in rows]
    with open(OUT / "feedback_screen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    print("rule | p | t | cells | P_surf | P_in | grow_surf | grow_in | rough | sh2 | sh4-8 | link tension")
    for (title, p), rows in data.items():
        r = rows[-1]
        print(f"{title[:28]:28s} | {p} | {r['t']:.0f} | {r['n_cells']} | {r['P_surf']:.2f} | {r['P_in']:.2f} | "
              f"{r['grow_surf']:.2f} | {r['grow_in']:.2f} | {r['rough']:.3f} | {r['sh2']:.3f} | {r['sh4_8']:.3f} | "
              f"{r['link_tension']:.3f}")


if __name__ == "__main__":
    main()
