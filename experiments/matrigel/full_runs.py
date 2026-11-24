#!/usr/bin/env python3
"""
matrigel/full_runs.py -- the full organoid runs (run_growth.py, t = 0-80, settings of Result 1's "all forces" run)
in the Matrigel-like mesh gel (make_full_gel.py) against the collagen-like gel of Result 1.

Runs:  runs/tao_all_on_p1_s7        collagen-like fibers (Result 1)
       runs/mg_mesh_ks10_p1_s7      Matrigel-like mesh, strand stiffness ks = 10 (same strands as the fibers)
       runs/mg_mesh_ks4.5_p1_s7     Matrigel-like mesh, ks = 4.5 (matches the collagen gel's push in the ball test)
For each run: the alignment S of a fixed set of strands, those whose midpoints start (t = 0) in the shell the
organoid reaches at t = 40 (radius 3.15 to 5.15 around the starting centre), as in Result 1; and the number of
cells and the mean pressure over time (metrics.csv). Writes results/full_runs.csv.
"""
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "jen_followup"))
import growthviz as gv            # noqa: E402
import alignment_mechanism as am  # noqa: E402

COLL, MESH = "experiments/ecm_connectivity/inputs/p1.0_seed7", "experiments/matrigel/inputs/mesh_p1.0_seed7"
# CHANGED 2026-11-24: the reruns, all on one engine build (tvm-runtime-v3). The first mesh runs (mg_mesh_ks10,
# mg_mesh_ks4.5) let a mesh node hold only one linker, so the mesh had a third of the binding slots, and ks = 4.5
# did not match the fibers' push on this mesh (ks = 2 does: ball_cohort.py). Old runs: set OLD = True.
OLD = len(sys.argv) > 1 and sys.argv[1] == "old"
RUNS = ({"collagen": ("runs/tao_all_on_p1_s7", COLL), "mesh_ks10": ("runs/mg_mesh_ks10_p1_s7", MESH),
         "mesh_ks4.5": ("runs/mg_mesh_ks4.5_p1_s7", MESH)} if OLD else
        {"collagen": ("runs/v3_collagen_all_on", COLL), "mesh_ks10": ("runs/v3_mesh_ks10_m3", MESH),
         "mesh_ks2": ("runs/v3_mesh_ks2_m3", MESH), "collagen_nolink": ("runs/v3_collagen_nolink", COLL),
         "mesh_ks10_nolink": ("runs/v3_mesh_ks10_nolink", MESH)})
R40 = 3.15
rows = []
for name, (run, init) in RUNS.items():
    run, init = ROOT / run, ROOT / init
    V, E, P, C = gv.read_topo(init / "sample.topo")
    pts = np.array([V[v] for c, (polys, typ) in C.items() if typ == 1 for p in polys for e in P[p] for v in E[e]])
    ids0, a0, b0 = am.segments(init / "ECM.topo")
    ok0, r0, s0 = am.S_of(a0, b0, pts.mean(axis=0))
    cohort = [p for p, k in ids0.items() if ok0[k] and R40 <= r0[k] < R40 + 2]
    frames = {round(f["t"]): f for f in pickle.load(open(run / "frames.pkl", "rb"))}
    metrics = {round(float(m["t"])): m for m in csv.DictReader(open(run / "metrics.csv"))}
    snaps = sorted(int(p.name[4:]) for p in run.glob("seg_*") if (p / f"{p.name[4:]}.ECM.topo").exists())
    rows.append(dict(run=name, t=0, S=float(s0[[ids0[p] for p in cohort]].mean()), n_cohort=len(cohort),
                     cells=int(metrics[1]["n_cells"]), mean_P=float(metrics[1]["mean_P"]), links=int(metrics[1]["n_links"])))
    for t in snaps:
        fr = frames[t]
        ids, a, b = am.segments(run / f"seg_{t:07d}" / f"{t:07d}.ECM.topo")
        ok, r, s = am.S_of(a, b, np.asarray(fr.get("center_vol", fr["center"])))
        k = np.array([ids[p] for p in cohort if p in ids])
        k = k[ok[k]]
        rows.append(dict(run=name, t=t, S=float(s[k].mean()), n_cohort=len(cohort),
                         cells=int(metrics[t]["n_cells"]), mean_P=float(metrics[t]["mean_P"]), links=int(metrics[t]["n_links"])))
    print(name, "fibers followed:", len(cohort), " ".join(f"t{r['t']}:S={r['S']:+.3f},N={r['cells']},P={r['mean_P']:.2f}"
                                                       for r in rows if r["run"] == name and r["t"] in (0, 20, 40, 60, 80)))
with open(HERE / "results" / ("full_runs_old.csv" if OLD else "full_runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
