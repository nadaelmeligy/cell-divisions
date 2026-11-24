#!/usr/bin/env python3
"""
matrigel/make_full_gel.py -- a Matrigel-like gel for the full organoid runs (run_growth.py), on the SAME
lattice sites as our collagen-like gel experiments/ecm_connectivity/inputs/p1.0_seed7/ECM.topo, so the hole
(no node closer than 3.2 to the organoid's centre), the box (64, periodic) and the spacing (2.83) are
unchanged. Only the structure changes: instead of long straight fibers, every nearest-neighbour lattice bond
is its own two-node strand, and strands meeting at a site share one node (a fully connected spring mesh with no
bending; see README.md for why this stands in for Matrigel). The engine's rest length 2.83 is the bond length.
Nodes keep the collagen gel's "linkable" flag (the hole-surface nodes); linkers.py can grab any node anyway.
Writes experiments/matrigel/inputs/mesh_p1.0_seed7/{ECM.topo, sample.topo}.
"""
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import growthviz as gv  # noqa: E402

SRC = os.path.join(ROOT, "experiments", "ecm_connectivity", "inputs", "p1.0_seed7")
OUT = os.path.join(HERE, "inputs", "mesh_p1.0_seed7")
BOX, L0 = 64.0, 2.8284271247461903

nodes, linkable, _ = gv.read_ecm(os.path.join(SRC, "ECM.topo"))
sites, flag = {}, {}
for nid, p in nodes.items():                      # three crosslink nodes share each lattice site (each joins 2 of its 6 fibers)
    key = tuple(np.round(p, 6))
    sites.setdefault(key, len(sites))
    flag[sites[key]] = flag.get(sites[key], 0) or (nid in linkable)
X = np.array(sorted(sites, key=sites.get))
pairs = sorted(cKDTree(np.mod(X + BOX / 2, BOX), boxsize=BOX).query_pairs(L0 * 1.001))
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "ECM.topo"), "w") as f:
    f.write(f"nodes {len(X)}\n")
    for i, (x, y, z) in enumerate(X):
        f.write(f"{i:6d} {x:.5e} {y:.5e} {z:.5e} {int(flag[i])}\n")
    f.write(f"fibers {len(pairs)}\n")
    for k, (i, j) in enumerate(pairs):
        f.write(f"{k} {i} {j}\n")
shutil.copy(os.path.join(SRC, "sample.topo"), os.path.join(OUT, "sample.topo"))
r = np.linalg.norm(X, axis=1)
print(f"{len(nodes)} collagen nodes -> {len(X)} sites, {len(pairs)} strands ({2 * len(pairs) / len(X):.1f} per site), "
      f"nearest site {r.min():.2f}, linkable {sum(flag.values())}")
