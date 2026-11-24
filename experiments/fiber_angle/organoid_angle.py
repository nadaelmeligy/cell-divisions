#!/usr/bin/env python3
"""
organoid_angle.py -- in the organoid runs, which fibers push back on the organoid and which hold its linkers
hardest, by fiber angle?

Reads runs/<run>/frames.pkl at t = 90, 100, ..., 130 for long_ecm_p1.0_s7-9 and long_ecm_p0.7_s7-9 (cells
grow, no pulling; contact k = 50, range 0.2; linkers ks = 10, rest 1.5).
Push: the organoid surface is the frame's 'surf' faces (between a real cell and the medium), split into fan
  triangles (face centre + two neighbouring corners, as in the engine), each normal pointing away from its own
  cell's centre. Test points every <= 0.5 along each nearby fiber segment ('fibers'); a point outside the
  organoid and closer than 0.2 to the surface gets the engine's contact force 50 ds (0.2 - d)
  (Energy/CellFiberContact.cpp; there a point whose nearest surface point is on a face edge is placed in or
  out by ray counting, here by the nearest triangle's normal). psi = angle between the segment and that
  triangle's normal (|cos psi| = 1 end-on, 0 lying flat).
Stretch: mean strain (l - l0)/l0 of segments whose midpoint is outside and within 1.5 of the surface, by psi.
Pull: each linker ('links': face centre -> fiber node) has tension 10 (length - rest); its alignment is the
  largest |cos| between the linker and the fiber segments at its node (1 = pulls along a fiber, 0 = pulls it
  sideways).
Output: results/organoid_angle.csv (push and stretch by run, t and |cos| bin), results/organoid_links.csv
(one row per linker and frame) and a summary on stdout.
    python3 experiments/fiber_angle/organoid_angle.py
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = [f"long_ecm_p{p}_s{s}" for p in ("1.0", "0.7") for s in (7, 8, 9)]
TIMES = [90, 100, 110, 120, 130]
RANGE, H = 0.2, 0.5
EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9])


def dot(u, v):
    return np.einsum("ij,ij->i", u, v)


def closest_on_tri(p, a, b, c):
    """Closest point to p on triangle (a, b, c), row by row (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac = b - a, c - a
    d1, d2 = dot(ab, p - a), dot(ac, p - a)
    d3, d4 = dot(ab, p - b), dot(ac, p - b)
    d5, d6 = dot(ab, p - c), dot(ac, p - c)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    q, done = np.empty_like(p), np.zeros(len(p), bool)

    def put(mask, val):
        m = mask & ~done
        q[m] = val[m]
        done[m] = True

    safe = lambda z: np.where(z != 0, z, 1.0)  # noqa: E731
    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / safe(d1 - d3))[:, None] * ab)
    put((d6 >= 0) & (d5 <= d6), c)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / safe(d2 - d6))[:, None] * ac)
    put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
        b + ((d4 - d3) / safe((d4 - d3) + (d5 - d6)))[:, None] * (c - b))
    s = safe(va + vb + vc)
    put(np.ones(len(p), bool), a + (vb / s)[:, None] * ab + (vc / s)[:, None] * ac)
    return q


class Surface:
    def __init__(self, fr):
        A, B, C, N = [], [], [], []
        for cid, pts in fr["surf"]:
            ctr = pts.mean(axis=0)
            cc = np.unique(fr["cell_edges"][cid].reshape(-1, 3), axis=0).mean(axis=0)
            for k in range(len(pts)):
                a, b = pts[k], pts[(k + 1) % len(pts)]
                n = np.cross(a - ctr, b - ctr)
                nn = np.linalg.norm(n)
                if nn < 1e-12:
                    continue
                n = n / nn
                A.append(ctr)
                B.append(a)
                C.append(b)
                N.append(n if n @ (ctr - cc) >= 0 else -n)
        self.A, self.B, self.C, self.N = map(np.array, (A, B, C, N))
        cen = (self.A + self.B + self.C) / 3
        self.rad = float(np.max(np.linalg.norm(np.stack([self.A, self.B, self.C], 1) - cen[:, None], axis=2)))
        self.tree = cKDTree(cen)
        self.area = float(0.5 * np.linalg.norm(np.cross(self.B - self.A, self.C - self.A), axis=1).sum())

    def nearest(self, x, reach):
        """For points x: distance to the surface, nearest triangle and point, and whether x is outside.
        Points with no triangle within `reach` get distance inf."""
        D = np.full(len(x), np.inf)
        T = np.zeros(len(x), int)
        Q = np.zeros_like(x)
        for lo in range(0, len(x), 2000):
            xs = x[lo:lo + 2000]
            cand = self.tree.query_ball_point(xs, self.rad + reach)
            cnt = np.array([len(c) for c in cand])
            if cnt.sum() == 0:
                continue
            pi = np.repeat(np.arange(len(xs)), cnt)
            ti = np.concatenate([np.asarray(c, int) for c in cand if len(c)])
            q = closest_on_tri(xs[pi], self.A[ti], self.B[ti], self.C[ti])
            dist = np.linalg.norm(xs[pi] - q, axis=1)
            order = np.lexsort((dist, pi))
            first = order[np.unique(pi[order], return_index=True)[1]]
            idx = lo + pi[first]
            D[idx], T[idx], Q[idx] = dist[first], ti[first], q[first]
        out = dot(x - Q, self.N[T]) >= 0
        return D, T, Q, out


def frame_rows(fr, run, cfg):
    S = Surface(fr)
    seg = fr["fibers"]
    d = seg[:, 1] - seg[:, 0]
    l = np.linalg.norm(d, axis=1)
    tdir = d / l[:, None]
    ns = np.ceil(l / H).astype(int)
    sid = np.repeat(np.arange(len(seg)), ns)
    w = np.concatenate([(np.arange(n) + 0.5) / n for n in ns])
    ds = (l / ns)[sid]
    x = seg[sid, 0] + w[:, None] * d[sid]
    D, T, _, out = S.nearest(x, 1.05)
    c = np.abs(dot(tdir[sid], S.N[T]))
    b = np.clip(np.digitize(c, EDGES) - 1, 0, 4)
    on = out & (D < RANGE)
    f = np.where(on, cfg["contact"][0] * ds * (RANGE - np.minimum(D, RANGE)), 0.0)
    near = out & (D < 1.0)
    # stretch of segments near the surface, by angle at their midpoint
    Dm, Tm, _, outm = S.nearest(seg.mean(axis=1), 1.55)
    cm = np.abs(dot(tdir, S.N[Tm]))
    bm = np.clip(np.digitize(cm, EDGES) - 1, 0, 4)
    shell = outm & (Dm < 1.5)
    F_tot = float(f.sum())
    rows = []
    for k in range(5):
        sel, ss = b == k, shell & (bm == k)
        rows.append(dict(run=run, p=run.split("_p")[1][:3], t=fr["t"], bin=k, P_contact=F_tot / S.area,
                         F=float(f[sel].sum()), F_share=float(f[sel].sum()) / F_tot if F_tot > 0 else np.nan,
                         contact_len=float(ds[on & sel].sum()), near_len=float(ds[near & sel].sum()),
                         n_inside_close=int((~out & (D < RANGE)).sum()),
                         strain_mean=float(fr["fiber_strain"][ss].mean()) if ss.any() else np.nan,
                         strain_n=int(ss.sum())))
    # linkers
    key = lambda p: tuple(np.round(p, 6))  # noqa: E731
    at = {}
    for k in range(len(seg)):
        for e in (0, 1):
            at.setdefault(key(seg[k, e]), []).append(k)
    rest = float(fr.get("link_rest", cfg["link"][2]))
    links = []
    for a, bb in fr["links"]:
        segs = at.get(key(bb), [])
        L = np.linalg.norm(bb - a)
        if not segs or L < 1e-9:
            continue
        links.append(dict(run=run, p=run.split("_p")[1][:3], t=fr["t"], align=float(np.max(np.abs(tdir[segs] @ ((bb - a) / L)))),
                          tension=cfg["link"][1] * (L - rest), length=L, n_seg=len(segs)))
    return rows, links


def main():
    rows, links = [], []
    for run in RUNS:
        path = os.path.join(ROOT, "runs", run)
        cfg = json.load(open(os.path.join(path, "config.json")))
        frames = {round(f["t"]): f for f in pickle.load(open(os.path.join(path, "frames.pkl"), "rb"))}
        for t in TIMES:
            r, lk = frame_rows(frames[t], run, cfg)
            rows += r
            links += lk
        print(f"{run}: done", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    d, L = pd.DataFrame(rows), pd.DataFrame(links)
    d.to_csv(os.path.join(HERE, "results", "organoid_angle.csv"), index=False)
    L.to_csv(os.path.join(HERE, "results", "organoid_links.csv"), index=False)

    print("\nPUSH (contact), pooled over 3 seeds and t = 90-130. Bins of |cos psi|: 0-.2 .2-.4 .4-.6 .6-.8 .8-1"
          " (left = lying flat, right = end-on)")
    for p, g in d.groupby("p", sort=False):
        s = g.groupby("bin")[["F", "contact_len", "near_len"]].sum()
        print(f"  p={p}: contact pressure {g.groupby(['run', 't'])['P_contact'].first().mean():.4f} "
              f"(mean over frames); points inside the organoid within 0.2: "
              f"{g.groupby(['run', 't'])['n_inside_close'].first().sum()}")
        print("    share of push-back       " + " ".join(f"{q:.2f}" for q in s["F"] / s["F"].sum()))
        print("    share of contact length  " + " ".join(f"{q:.2f}" for q in s["contact_len"] / s["contact_len"].sum()))
        print("    share of fiber within 1  " + " ".join(f"{q:.2f}" for q in s["near_len"] / s["near_len"].sum()))
        print("    push per contact length  " + " ".join(f"{q:.2f}" for q in s["F"] / s["contact_len"].replace(0, np.nan)))
        st = g.groupby("bin").apply(lambda q: np.average(q["strain_mean"].fillna(0), weights=q["strain_n"])
                                    if q["strain_n"].sum() else np.nan)
        print("    mean strain within 1.5   " + " ".join(f"{q:+.4f}" for q in st))
    print("\nPULL (linkers): mean tension by alignment bin 0-.2 .2-.4 .4-.6 .6-.8 .8-1 (right = pulls along a fiber);"
          " Spearman rho(alignment, tension) per run")
    for p, g in L.groupby("p", sort=False):
        bins = np.clip(np.digitize(g["align"], EDGES) - 1, 0, 4)
        m = g.groupby(bins)["tension"].agg(["mean", "size"])
        rho = [spearmanr(q["align"], q["tension"])[0] for _, q in g.groupby("run")]
        print(f"  p={p}: " + " ".join(f"{a:.3f}(n={n})" for a, n in zip(m["mean"], m["size"]))
              + "   rho " + " ".join(f"{r:+.2f}" for r in rho))


if __name__ == "__main__":
    main()
