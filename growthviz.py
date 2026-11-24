#!/usr/bin/env python3
"""
growthviz.py -- read the engine's restart files and turn a run into an MP4.
No ParaView. Needs numpy, scipy, matplotlib and ffmpeg.

You normally don't call this yourself: run_growth.py calls it at the end.
To re-render a finished run (e.g. after changing colours):

    python3 growthviz.py runs/my_run                      # -> runs/my_run/movie.mp4
    python3 growthviz.py --compare runs/before runs/after --out before_after.mp4

What is drawn
  left   : 3D view of the spheroid surface (cells coloured by pressure),
           ECM fibers near the spheroid (coloured by strain), linkers (yellow).
  middle : mid-plane cross-section, like a confocal slice: each cell filled by its
           pressure, fibers crossing the slice, linkers in the slice.
  right  : time series (cell number, linkers, mean pressure, ECM strain).
"""
import argparse
import os
import pickle
import sys

import numpy as np

# ----------------------------------------------------------------------------------
# 1. Reading the engine's text files
# ----------------------------------------------------------------------------------

def read_topo(path):
    """Read a .topo file -> (V, E, P, C).
    V[id] = (x,y,z); E[id] = (v1, v2); P[id] = [edge ids]; C[id] = ([polygon ids], type)
    type 1 = real cell, type 0 = 'empty' cell (the soft medium around the spheroid)."""
    V, E, P, C = {}, {}, {}, {}
    sec = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0] == "time":
                continue
            if t[0] in ("vertices", "edges", "polygons", "cells"):
                sec = t[0]
                continue
            if sec == "vertices":
                V[int(t[0])] = (float(t[1]), float(t[2]), float(t[3]))
            elif sec == "edges":
                E[int(t[0])] = (int(t[1]), int(t[2]))
            elif sec == "polygons":
                P[int(t[0])] = [int(x) for x in t[1:]]
            elif sec == "cells":
                C[int(t[0])] = ([int(x) for x in t[1:-1]], int(t[-1]))
    return V, E, P, C


def read_ecm(path):
    """Read an ECM .topo file -> (nodes {id: xyz}, linkable set, fibers [[node ids]])."""
    nodes, linkable, fibers = {}, set(), []
    sec = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] in ("nodes", "fibers"):
                sec = t[0]
                continue
            if sec == "nodes":
                nid = int(t[0])
                nodes[nid] = (float(t[1]), float(t[2]), float(t[3]))
                if int(t[-1]) > 0:
                    linkable.add(nid)
            elif sec == "fibers":
                fibers.append([int(x) for x in t[1:]])
    return nodes, linkable, fibers


def read_links(path):
    """Read NNNNNNN.linkInfo.txt -> list of (polygon id, node id)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2 and t[0].lstrip("-").isdigit():
                out.append((int(t[0]), int(t[1])))
    return out


def read_link_rests(path):
    """CHANGED 2026-11-24: passive linkers ('link_rest attach') save their own rest length as a 3rd column of
    the link file. Returns {(polygon id, node id): rest length} for the lines that have one."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 3 and t[0].lstrip("-").isdigit():
                out[(int(t[0]), int(t[1]))] = float(t[2])
    return out


def read_cell_volumes(path):
    """cellVolume.txt -> {time: {cell id: volume}} (only real cells are listed)."""
    blocks, cur = {}, None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "time":
                cur = blocks.setdefault(int(round(float(t[1]))), {})
            elif cur is not None:
                cur[int(t[0])] = float(t[1])
    return blocks


def read_last_log_row(path):
    """Last numeric row of the engine's screen log -> dict of energies / counts."""
    last = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 13 and t[0].replace(".", "", 1).isdigit():
                last = t
    if last is None:
        return {}
    # Time Rte Volume I->H H->I N +formed E_V E_S ECM_str ECM_bend Link_str Energy
    return {"time": float(last[0]), "E_V": float(last[7]), "E_S": float(last[8]),
            "E_ecm_stretch": float(last[9]), "E_ecm_bend": float(last[10]),
            "E_link": float(last[11]), "E_total": float(last[12])}

# ----------------------------------------------------------------------------------
# 2. Geometry helpers
# ----------------------------------------------------------------------------------

def polygon_loop(edge_ids, E):
    """Ordered vertex ids around a polygon given its (unordered) edges. Raises ValueError unless the
    edges form exactly one simple loop: a division plane can cut a bent cell's surface in two places,
    and a face made of two loops crashes the engine (Cell::updateVolume, 'cannot find next polygon')."""
    if len(edge_ids) < 3 or len(set(edge_ids)) != len(edge_ids):
        raise ValueError("polygon has fewer than 3 edges or a repeated edge")
    adj = {}
    for e in edge_ids:
        a, b = E[e]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(nb) != 2 for nb in adj.values()):
        raise ValueError("polygon edges do not form a single loop (a vertex is not on exactly two edges)")
    start = E[edge_ids[0]][0]
    loop, prev, cur = [start], None, start
    while True:
        nb = adj[cur]
        nxt = nb[0] if nb[0] != prev else nb[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(edge_ids):
            raise ValueError("polygon edges do not form a single loop")
    if len(loop) != len(edge_ids):   # the walk closed before using every edge: two or more loops
        raise ValueError("polygon edges form more than one loop")
    return loop


def polygon_center(edge_ids, E, V):
    """Same definition as the C++ (Polygon::updateCenter): edge-length-weighted edge midpoints."""
    s, w = np.zeros(3), 0.0
    for e in edge_ids:
        a, b = np.array(V[E[e][0]]), np.array(V[E[e][1]])
        l = np.linalg.norm(b - a)
        s += l * 0.5 * (a + b)
        w += l
    return s / w


def oriented_face_loops(polys, P, E):
    """{face id: vertex loop} for one cell, every loop walked the same way round the cell, as in the
    engine's Cell::updateVolume: two faces that share an edge walk it in opposite directions."""
    loops = {p: polygon_loop(P[p], E) for p in polys}
    by_edge = {}
    for p, l in loops.items():
        for i in range(len(l)):
            by_edge.setdefault(frozenset((l[i], l[(i + 1) % len(l)])), []).append(p)
    out = {polys[0]: loops[polys[0]]}
    todo = [polys[0]]
    while todo:
        p = todo.pop()
        l = out[p]
        for i in range(len(l)):
            a, b = l[i], l[(i + 1) % len(l)]
            for q in by_edge[frozenset((a, b))]:
                if q not in out:
                    lq = loops[q]
                    k = lq.index(b)
                    out[q] = lq if lq[(k + 1) % len(lq)] == a else lq[::-1]
                    todo.append(q)
    if len(out) != len(polys):
        raise ValueError("the cell's faces do not form one connected surface")
    return out


def cell_geometry(cell_id, V, E, P, C):
    """Volume, volume-weighted centroid and vertex array of one cell (fan triangulation of every
    face from its centre, signed tetrahedra to a reference point, faces oriented consistently as
    in the engine). Before 2026-11-19 each tetrahedron counted with abs(), which overestimated the
    volume of non-convex cells (by up to 0.04 for real cells, 0.18 for medium cells)."""
    polys = C[cell_id][0]
    vids = sorted({v for p in polys for e in P[p] for v in E[e]})
    pts = np.array([V[v] for v in vids])
    r0 = pts.mean(axis=0)
    vol, mom = 0.0, np.zeros(3)
    for p, loop in oriented_face_loops(polys, P, E).items():
        c = polygon_center(P[p], E, V)
        for i in range(len(loop)):
            a, b = np.array(V[loop[i]]), np.array(V[loop[(i + 1) % len(loop)]])
            dv = np.dot(a - r0, np.cross(b - r0, c - r0)) / 6.0
            vol += dv
            mom += dv * (r0 + a + b + c) / 4.0
    if vol < 0:   # the loops came out oriented inward
        vol, mom = -vol, -mom
    return vol, mom / vol, pts


def polygon_cells(C):
    """polygon id -> list of (cell id, cell type)."""
    pc = {}
    for cid, (polys, typ) in C.items():
        for p in polys:
            pc.setdefault(p, []).append((cid, typ))
    return pc

# ----------------------------------------------------------------------------------
# 3. How the fiber network is arranged around the organoid
# ----------------------------------------------------------------------------------

SHELL_EDGES = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])   # distance outside the surface
_REF_CACHE = {}


def ecm_structure(nodes, fibers, center, surface_radius, ref_nodes=None):
    """Orientation and movement of every fiber segment, in shells around the organoid.

    For each segment (two neighbouring nodes of a fiber):
      s     distance of its midpoint outside the organoid surface: |mid - centre| - surface_radius
      cos   |cos theta|, theta = angle between the segment and the line from the organoid centre
      order S = (3 cos^2 - 1) / 2:  +1 all segments point at the organoid (radial),
            0 random, -0.5 all segments wrap around it (tangential)
      rot   rotation since the reference state (degrees, 0-90), if ref_nodes is given
      disp  how far the midpoint moved since the reference state, if ref_nodes is given
    Returns per-shell means (shells = SHELL_EDGES) and a few summary numbers."""
    ids = np.array([(a, b) for f in fibers for a, b in zip(f[:-1], f[1:])])
    pos = {k: np.asarray(v) for k, v in nodes.items()}
    A = np.array([pos[a] for a, _ in ids])
    B = np.array([pos[b] for _, b in ids])
    seg = B - A
    L = np.linalg.norm(seg, axis=1)
    ok = (L > 1e-12) & (L < 10.0)          # skip segments that wrap the periodic box
    A, B, seg, L, ids = A[ok], B[ok], seg[ok], L[ok], ids[ok]
    mid = 0.5 * (A + B)
    rvec = mid - center
    r = np.linalg.norm(rvec, axis=1)
    s = r - surface_radius
    cos = np.abs(np.einsum("ij,ij->i", rvec, seg)) / (r * L + 1e-30)
    order = 1.5 * cos ** 2 - 0.5
    rot = disp = None
    if ref_nodes is not None:
        A0 = np.array([ref_nodes[a] for a, _ in ids])
        B0 = np.array([ref_nodes[b] for _, b in ids])
        seg0 = B0 - A0
        L0 = np.linalg.norm(seg0, axis=1)
        c = np.abs(np.einsum("ij,ij->i", seg, seg0)) / (L * L0 + 1e-30)
        rot = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
        disp = np.linalg.norm(mid - 0.5 * (A0 + B0), axis=1)
    shell = np.digitize(s, SHELL_EDGES) - 1
    nshell = len(SHELL_EDGES) - 1
    prof = {"n": np.zeros(nshell, int), "order": np.full(nshell, np.nan),
            "rot": np.full(nshell, np.nan), "disp": np.full(nshell, np.nan)}
    for k in range(nshell):
        m = shell == k
        prof["n"][k] = m.sum()
        if m.any():
            prof["order"][k] = order[m].mean()
            if rot is not None:
                prof["rot"][k] = rot[m].mean()
                prof["disp"][k] = disp[m].mean()
    near, far = (s >= 0) & (s < 2), (s >= 4) & (s < 8)
    summary = {"order_near": float(order[near].mean()) if near.any() else np.nan,
               "order_far": float(order[far].mean()) if far.any() else np.nan,
               "n_fibers_near": int(near.sum())}
    if rot is not None:
        summary.update(rot_near=float(rot[near].mean()) if near.any() else np.nan,
                       rot_far=float(rot[far].mean()) if far.any() else np.nan,
                       disp_near=float(disp[near].mean()) if near.any() else np.nan,
                       disp_far=float(disp[far].mean()) if far.any() else np.nan)
    return prof, summary


def reference_nodes(path):
    """Fiber node positions of a reference ECM file (read once, then cached)."""
    if path not in _REF_CACHE:
        _REF_CACHE[path] = {k: np.asarray(v) for k, v in read_ecm(path)[0].items()}
    return _REF_CACHE[path]

# ----------------------------------------------------------------------------------
# 4. One movie frame = a small dictionary saved after every segment
# ----------------------------------------------------------------------------------

def make_frame(t, topo_path, ecm_path, link_path, cells, fiber_l0, link_l0, extra, ref_ecm=None):
    """cells: {cell id: dict(V=..., V0=..., P=...)} for every real cell.
    ref_ecm: ECM file of the start of the run; if given, fiber rotation and movement since then
    are measured too (see ecm_structure).
    Returns a compact dict holding only what the movie needs (a few hundred kB)."""
    V, E, P, C = read_topo(topo_path)
    real = [cid for cid, (_, typ) in C.items() if typ > 0]
    pc = polygon_cells(C)

    allpts = np.array([V[v] for cid in real for p in C[cid][0] for e in P[p] for v in E[e]])
    center = allpts.mean(axis=0)
    radius = np.linalg.norm(allpts - center, axis=1).max()
    # 'center' counts each vertex once per (cell, face, edge) it sits on, so it jumps when divisions
    # and neighbour swaps change those counts. For tracking the organoid's motion use center_vol, the
    # volume-weighted centroid of the real cells.
    geo = [cell_geometry(cid, V, E, P, C)[:2] for cid in real]
    center_vol = np.average([g[1] for g in geo], axis=0, weights=[g[0] for g in geo])

    # spheroid surface: faces between a real cell and a non-real cell (or nothing)
    surf = []
    for p, owners in pc.items():
        reals = [cid for cid, typ in owners if typ > 0]
        if len(reals) == 1 and (len(owners) == 1 or any(typ <= 0 for _, typ in owners)):
            loop = polygon_loop(P[p], E)
            surf.append((reals[0], np.array([V[v] for v in loop])))

    # every real cell's edges (for the cross-section)
    cell_edges = {}
    for cid in real:
        es = {e for p in C[cid][0] for e in P[p]}
        cell_edges[cid] = np.array([[V[E[e][0]], V[E[e][1]]] for e in es])

    # ECM fibers near the spheroid, with strain (l - l0)/l0, the same strain the engine writes
    # into NNNNNNN.ECM.vtk (Run::dumpConfigurationVtk)
    nodes, linkable, fibers = read_ecm(ecm_path)
    seg, strain = [], []
    rmax = radius + 5.0
    for fib in fibers:
        for a, b in zip(fib[:-1], fib[1:]):
            pa, pb = np.array(nodes[a]), np.array(nodes[b])
            d = pb - pa
            l = np.linalg.norm(d)
            if l > 10.0:  # crosses the periodic box: not near the spheroid anyway
                continue
            if np.linalg.norm(0.5 * (pa + pb) - center) < rmax:
                seg.append((pa, pb))
                strain.append((l - fiber_l0) / fiber_l0)

    # linkers: centre of the linked face -> ECM node. link_strain = (length - link_l0) / link_l0, so
    # pass the engine's CURRENT rest length as link_l0 (it shrinks when cells pull; run_growth.py
    # passes link_rest_length(cfg, t)); then tension = ks * link_l0 * link_strain
    rests = read_link_rests(link_path)   # CHANGED 2026-11-24: passive linkers carry their own rest length
    links, lstrain, lrest = [], [], []
    for pid, nid in read_links(link_path):
        if pid in P and nid in nodes:
            a, b = polygon_center(P[pid], E, V), np.array(nodes[nid])
            r0 = rests.get((pid, nid), link_l0)
            links.append((a, b))
            lstrain.append((np.linalg.norm(b - a) - r0) / r0)
            lrest.append(r0)

    seg = np.array(seg).reshape(-1, 2, 3)
    # fiber points (at 0, 1/4, 1/2, 3/4 of each nearby segment) that lie inside a cell. Each cell is
    # close to convex, so "inside a cell" = inside the convex hull of that cell's corners. (The hull
    # of the whole organoid would also count fiber points in dents between cells.)
    from scipy.spatial import Delaunay
    pts = np.concatenate([seg[:, 0] + f * (seg[:, 1] - seg[:, 0]) for f in (0.0, 0.25, 0.5, 0.75)]) if len(seg) else np.zeros((0, 3))
    inside = np.zeros(len(pts), bool)
    for cid in real:
        corners = np.unique(cell_edges[cid].reshape(-1, 3), axis=0)
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        near_cell = ~inside & np.all((pts >= lo) & (pts <= hi), axis=1)
        if near_cell.any():
            inside[near_cell] |= Delaunay(corners).find_simplex(pts[near_cell]) >= 0
    n_inside = int(inside.sum())
    strain = np.array(strain)
    near = np.linalg.norm(seg.mean(axis=1) - center, axis=1) < radius + 2.5 if len(seg) else np.array([], bool)
    frame = dict(t=t, center=center, center_vol=center_vol, radius=radius, cells=cells, surf=surf,
                 cell_edges=cell_edges, fibers=seg, fiber_strain=strain,
                 links=np.array(links).reshape(-1, 2, 3), link_strain=np.array(lstrain), link_rest=link_l0, link_rest_each=np.array(lrest),
                 n_cells=len(real), n_links=len(links), fiber_points_inside=n_inside,
                 mean_abs_fiber_strain_near=float(np.mean(np.abs(strain[near]))) if near.any() else 0.0)

    # fiber orientation around the organoid (radial vs wrapping) and change since the start
    surface_radius = float(np.mean([np.linalg.norm(pts.mean(axis=0) - center) for _, pts in surf]))
    ref = reference_nodes(str(ref_ecm)) if ref_ecm is not None else None
    prof, summary = ecm_structure(nodes, fibers, center, surface_radius, ref)
    if len(seg):
        mids = seg.mean(axis=1) - center
        d = seg[:, 1] - seg[:, 0]
        fiber_cos = np.abs(np.einsum("ij,ij->i", mids, d)) / (np.linalg.norm(mids, axis=1) * np.linalg.norm(d, axis=1) + 1e-30)
    else:
        fiber_cos = np.zeros(0)
    frame.update(surface_radius=surface_radius, ecm_profile=prof, fiber_cos=fiber_cos, **summary)
    frame.update(extra)
    return frame

# ----------------------------------------------------------------------------------
# 5. Drawing
# ----------------------------------------------------------------------------------

P_LIM = (0.0, 8.0)          # pressure colour range (fixed so frames/runs are comparable)
STRAIN_LIM = (-0.05, 0.05)  # fiber strain colour range
ELEV, AZIM = 22.0, 35.0


def _view_dir():
    e, a = np.radians(ELEV), np.radians(AZIM)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def draw_3d(ax, fr, half_width):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    pnorm = plt.Normalize(*P_LIM)
    snorm = plt.Normalize(*STRAIN_LIM)
    cmapP, cmapS = plt.get_cmap("magma"), plt.get_cmap("coolwarm")
    view, c0 = _view_dir(), fr["center"]
    ax.computed_zorder = False

    # fibers: matplotlib 3D does not clip to the axes, so keep only segments inside the view cube
    fib, fstrain = fr["fibers"], fr["fiber_strain"]
    if len(fib):
        inside = np.all(np.abs(fib - c0).max(axis=2) < half_width, axis=1)
        fib, fstrain = fib[inside], fstrain[inside]
    if len(fib):
        depth = (fib.mean(axis=1) - c0) @ view
        back = depth < 0
        ax.add_collection3d(Line3DCollection(fib[back], colors=cmapS(snorm(fstrain[back])),
                                             linewidths=1.1, alpha=0.85, zorder=1))
    # linkers behind
    if len(fr["links"]):
        ldepth = (fr["links"].mean(axis=1) - c0) @ view
        ax.add_collection3d(Line3DCollection(fr["links"][ldepth < 0], colors="#ffd21f", linewidths=2.2, zorder=2))

    # spheroid: only faces turned towards the camera (convex blob -> no sorting artefacts)
    polys, cols = [], []
    for cid, pts in fr["surf"]:
        n = np.zeros(3)
        for i in range(len(pts)):  # Newell normal
            a, b = pts[i], pts[(i + 1) % len(pts)]
            n += np.cross(a, b)
        if np.dot(n, pts.mean(axis=0) - c0) < 0:
            n = -n
        if np.dot(n, view) > 0:
            polys.append(pts)
            cols.append(cmapP(pnorm(fr["cells"].get(cid, {}).get("P", 0.0))))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="#2a1a10", linewidths=0.35, zorder=3))

    if len(fr["links"]):
        ax.add_collection3d(Line3DCollection(fr["links"][ldepth >= 0], colors="#ffd21f", linewidths=2.6, zorder=4))
    if len(fib):
        ax.add_collection3d(Line3DCollection(fib[~back], colors=cmapS(snorm(fstrain[~back])),
                                             linewidths=1.1, alpha=0.35, zorder=5))
    for setlim, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setlim(c0[k] - half_width, c0[k] + half_width)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.view_init(ELEV, AZIM)
    ax.set_axis_off()


def draw_section(ax, fr, half_width, slab=0.6):
    """Mid-plane slice z = z_center: cells filled by pressure, fibers and linkers in the slab."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    pnorm, snorm = plt.Normalize(*P_LIM), plt.Normalize(*STRAIN_LIM)
    cmapP, cmapS = plt.get_cmap("magma"), plt.get_cmap("coolwarm")
    c0 = fr["center"]
    z0 = c0[2]
    shapes, cols = [], []
    for cid, edges in fr["cell_edges"].items():
        za, zb = edges[:, 0, 2] - z0, edges[:, 1, 2] - z0
        cross = za * zb < 0
        if cross.sum() < 3:
            continue
        a, b = edges[cross, 0], edges[cross, 1]
        s = (za[cross] / (za[cross] - zb[cross]))[:, None]
        pts = (a + s * (b - a))[:, :2]
        m = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - m[1], pts[:, 0] - m[0]))
        shapes.append(pts[order])
        cols.append(cmapP(pnorm(fr["cells"].get(cid, {}).get("P", 0.0))))
    if len(fr["fibers"]):
        f = fr["fibers"]
        inslab = np.abs(f.mean(axis=1)[:, 2] - z0) < slab
        ax.add_collection(LineCollection(f[inslab][:, :, :2], colors=cmapS(snorm(fr["fiber_strain"][inslab])),
                                         linewidths=1.6))
    ax.add_collection(PolyCollection(shapes, facecolors=cols, edgecolors="#1b1b1b", linewidths=0.8))
    if len(fr["links"]):
        l = fr["links"]
        inslab = np.abs(l.mean(axis=1)[:, 2] - z0) < 1.5 * slab
        ax.add_collection(LineCollection(l[inslab][:, :, :2], colors="#ffd21f", linewidths=2.6))
    ax.set_xlim(c0[0] - half_width, c0[0] + half_width)
    ax.set_ylim(c0[1] - half_width, c0[1] + half_width)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("mid-plane slice", fontsize=10, color="#cccccc")


SERIES = [("n_cells", "cells"), ("n_links", "cell–ECM linkers"),
          ("mean_P", "mean cell pressure P"), ("mean_abs_fiber_strain_near", "mean |strain| of nearby fibers")]


def draw_series(axes, runs, upto):
    """runs: list of (label, colour, frames). Draws every series up to frame index `upto`."""
    tmin = min(fr["t"] for _, _, frs in runs for fr in frs)
    tmax = max(fr["t"] for _, _, frs in runs for fr in frs)
    for ax, (key, title) in zip(axes, SERIES):
        ymax = max(max(fr.get(key, 0) for fr in frs) for _, _, frs in runs)
        for label, col, frs in runs:
            t = [fr["t"] for fr in frs[:upto + 1]]
            y = [fr.get(key, 0) for fr in frs[:upto + 1]]
            ax.plot(t, y, color=col, lw=1.8, label=label)
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(0, ymax * 1.15 + 1e-9)
        ax.set_title(title, fontsize=9, color="#dddddd", loc="left")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for s in ax.spines.values():
            s.set_color("#555555")
        ax.set_facecolor("#16161a")
    axes[-1].set_xlabel("simulation time", fontsize=8, color="#aaaaaa")
    if len(runs) > 1:
        axes[0].legend(fontsize=7, facecolor="#222228", labelcolor="#dddddd", edgecolor="#444444")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#101014", "axes.facecolor": "#101014",
                         "savefig.facecolor": "#101014", "text.color": "#e8e8e8"})
    return plt


def _colorbars(fig, plt):
    from matplotlib.cm import ScalarMappable
    cax1 = fig.add_axes([0.06, 0.06, 0.18, 0.015])
    cb = fig.colorbar(ScalarMappable(plt.Normalize(*P_LIM), "magma"), cax=cax1, orientation="horizontal")
    cb.set_label("cell pressure P = −2kv(V−V0)", fontsize=8, color="#cccccc")
    cb.ax.tick_params(labelsize=7, colors="#aaaaaa")
    cax2 = fig.add_axes([0.30, 0.06, 0.18, 0.015])
    cb = fig.colorbar(ScalarMappable(plt.Normalize(*STRAIN_LIM), "coolwarm"), cax=cax2, orientation="horizontal")
    cb.set_label("fiber strain (blue = compressed, red = stretched); yellow = linkers", fontsize=8, color="#cccccc")
    cb.ax.tick_params(labelsize=7, colors="#aaaaaa")


def render_run(run_dir, out=None, fps=6):
    """One run -> movie.mp4 (3D view, slice, time series)."""
    plt = _style()
    from matplotlib.animation import FFMpegWriter
    with open(os.path.join(run_dir, "frames.pkl"), "rb") as f:
        frames = pickle.load(f)
    out = out or os.path.join(run_dir, "movie.mp4")
    hw = max(fr["radius"] for fr in frames) + 2.5
    fig = plt.figure(figsize=(16, 9), dpi=100)
    writer = FFMpegWriter(fps=fps, bitrate=6000)
    with writer.saving(fig, out, dpi=100):
        for i, fr in enumerate(frames):
            fig.clf()
            ax3 = fig.add_axes([0.0, 0.1, 0.42, 0.82], projection="3d")
            axs = fig.add_axes([0.43, 0.15, 0.30, 0.72])
            series = [fig.add_axes([0.79, 0.73 - k * 0.205, 0.19, 0.15]) for k in range(4)]
            draw_3d(ax3, fr, hw)
            draw_section(axs, fr, hw)
            draw_series(series, [(os.path.basename(run_dir.rstrip("/")), "#4fc3f7", frames)], i)
            _colorbars(fig, plt)
            fig.suptitle(f"t = {fr['t']:.0f}    cells = {fr['n_cells']}    linkers = {fr['n_links']}    "
                         f"divisions this step = {fr.get('n_div', 0)}", fontsize=14, y=0.97)
            writer.grab_frame()
    plt.close(fig)
    return out


def render_compare(dir_a, dir_b, out, label_a="before fix", label_b="after fix", fps=6):
    """Two runs side by side + overlaid time series."""
    plt = _style()
    from matplotlib.animation import FFMpegWriter
    runs = []
    for d in (dir_a, dir_b):
        with open(os.path.join(d, "frames.pkl"), "rb") as f:
            runs.append(pickle.load(f))
    n = min(len(r) for r in runs)
    hw = max(fr["radius"] for r in runs for fr in r) + 2.5
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = FFMpegWriter(fps=fps, bitrate=8000)
    cols = ["#ff7a59", "#4fc3f7"]
    with writer.saving(fig, out, dpi=100):
        for i in range(n):
            fig.clf()
            for k, (r, lab) in enumerate(zip(runs, (label_a, label_b))):
                fr = r[i]
                x0 = 0.01 + k * 0.495
                ax3 = fig.add_axes([x0, 0.40, 0.25, 0.52], projection="3d")
                axs = fig.add_axes([x0 + 0.255, 0.44, 0.215, 0.44])
                draw_3d(ax3, fr, hw)
                draw_section(axs, fr, hw)
                fig.text(x0 + 0.24, 0.93, f"{lab}:  {fr['n_cells']} cells,  {fr['n_links']} linkers",
                         ha="center", fontsize=14, color=cols[k], weight="bold")
            series = [fig.add_axes([0.06 + k * 0.235, 0.14, 0.19, 0.20]) for k in range(4)]
            draw_series(series, [(label_a, cols[0], runs[0][:n]), (label_b, cols[1], runs[1][:n])], i)
            _colorbars(fig, plt)
            ta, tb = runs[0][i]["t"], runs[1][i]["t"]
            fig.suptitle(f"t = {ta:g}" if ta == tb else f"t = {ta:g} (left), {tb:g} (right)", fontsize=16, y=0.99)
            writer.grab_frame()
    plt.close(fig)
    return out


def plot_compare_png(dir_a, dir_b, out, label_a="before fix", label_b="after fix"):
    """Static 2x2 figure of the time series of two runs (for notes and slides)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = []
    for d in (dir_a, dir_b):
        with open(os.path.join(d, "frames.pkl"), "rb") as f:
            runs.append(pickle.load(f))
    for frs in runs:
        for fr in frs:
            L = fr["links"]
            fr["mean_link_length"] = float(np.linalg.norm(L[:, 1] - L[:, 0], axis=1).mean()) if len(L) else np.nan
    keys = SERIES[:2] + [("mean_link_length", "mean linker length (rest length 1.5 unless traction)"), SERIES[3]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=150)
    for ax, (key, title) in zip(axes.flat, keys):
        for frs, lab, col, ls in ((runs[0], label_a, "#d9534f", "--"), (runs[1], label_b, "#1f77b4", "-")):
            ax.plot([fr["t"] for fr in frs], [fr.get(key, 0) for fr in frs], ls, color=col, lw=2, label=lab)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("simulation time")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def draw_section_orientation(ax, fr, half_width, slab=0.6, view_center=None):
    """Mid-plane slice with fibers coloured by direction: blue = wrapping around the organoid
    (tangential), red = pointing at it (radial). Cells are grey. With view_center the window stays
    fixed there (so a drifting organoid visibly moves), with + = that point and a dot = the current centre."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    c0 = fr["center"]
    z0 = c0[2]
    shapes = []
    for cid, edges in fr["cell_edges"].items():
        za, zb = edges[:, 0, 2] - z0, edges[:, 1, 2] - z0
        cross = za * zb < 0
        if cross.sum() < 3:
            continue
        a, b = edges[cross, 0], edges[cross, 1]
        s = (za[cross] / (za[cross] - zb[cross]))[:, None]
        pts = (a + s * (b - a))[:, :2]
        m = pts.mean(axis=0)
        shapes.append(pts[np.argsort(np.arctan2(pts[:, 1] - m[1], pts[:, 0] - m[0]))])
    if len(fr["fibers"]):
        f = fr["fibers"]
        inslab = np.abs(f.mean(axis=1)[:, 2] - z0) < slab
        ax.add_collection(LineCollection(f[inslab][:, :, :2], colors=plt.get_cmap("coolwarm")(fr["fiber_cos"][inslab]),
                                         linewidths=1.8))
    ax.add_collection(PolyCollection(shapes, facecolors="#8a8f98", edgecolors="#1b1b1b", linewidths=0.6))
    if len(fr["links"]):
        l = fr["links"]
        inslab = np.abs(l.mean(axis=1)[:, 2] - z0) < 1.5 * slab
        ax.add_collection(LineCollection(l[inslab][:, :, :2], colors="#ffd21f", linewidths=2.2))
    cv = c0 if view_center is None else view_center
    if view_center is not None:
        ax.plot([cv[0]], [cv[1]], "+", color="white", ms=16, mew=2, zorder=5)
        ax.plot([c0[0]], [c0[1]], "o", color="#00e5ff", ms=6, zorder=6)
    ax.set_xlim(cv[0] - half_width, cv[0] + half_width)
    ax.set_ylim(cv[1] - half_width, cv[1] + half_width)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


GRID_SERIES = {   # time series that render_grid can plot under the slices (key in frames.pkl: title)
    "order_near": "fiber alignment within 2 of the surface (+1 radial, 0 random, -0.5 wrapping)",
    "rot_near": "fiber rotation since t = 0 (degrees), within 2 of the surface",
    "n_cells": "number of cells",
    "mean_P": "mean cell pressure (growth stops at P = P_h)",
    "mean_abs_fiber_strain_near": "mean |fiber stretch| near the organoid",
    "n_links": "cell-ECM linkers",
    "center_x": "organoid centre: x position since the start",   # computed in _series_value
}


def _series_value(fr, key, run):
    """One point of a GRID_SERIES curve (most are stored in frames.pkl; center_x is derived)."""
    if key == "center_x":
        return float(fr["center"][0] - run[0]["center"][0])
    return fr.get(key, np.nan)


def render_grid(dirs, labels, out, fps=6, series=("order_near", "rot_near", "n_cells"), fixed_view=False):
    """Up to 4 runs side by side as mid-plane slices (fibers coloured by direction), with three
    time series of all runs underneath (keys of GRID_SERIES)."""
    plt = _style()
    from matplotlib.animation import FFMpegWriter
    from matplotlib.cm import ScalarMappable
    if len(labels) != len(dirs):   # e.g. --labels left at its 2-entry default for 4 runs
        labels = [os.path.basename(os.path.normpath(d)) for d in dirs]
    runs = []
    for d in dirs:
        with open(os.path.join(d, "frames.pkl"), "rb") as f:
            runs.append(pickle.load(f))
    n = min(len(r) for r in runs)
    hw = max(fr["radius"] for r in runs for fr in r) + 3.0
    # two runs: blue vs red, so they are easy to tell apart
    cols = ["#5aa9ff", "#ff6b5a"] if len(runs) == 2 else ["#5aa9ff", "#9ad0ff", "#ffb35a", "#ff6b5a"]
    series = [(key, GRID_SERIES[key]) for key in series]
    pw = 0.988 / len(runs)   # panel slot width: the slices fill the row whatever the number of runs
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = FFMpegWriter(fps=fps, bitrate=8000)
    with writer.saving(fig, out, dpi=100):
        for i in range(n):
            fig.clf()
            for k, (r, lab) in enumerate(zip(runs, labels)):
                fr = r[i]
                ax = fig.add_axes([0.01 + k * pw, 0.42, pw - 0.012, 0.49])
                draw_section_orientation(ax, fr, hw, view_center=(r[0]["center"] if fixed_view else None))
                ax.set_title(f"{lab}\n{fr['n_cells']} cells, {fr['n_links']} linkers, alignment {fr.get('order_near', 0):+.2f}",
                             fontsize=12, color=cols[k % 4])
            for j, (key, title) in enumerate(series):
                ax = fig.add_axes([0.05 + j * 0.325, 0.08, 0.27, 0.24])
                tmax = max(fr["t"] for r in runs for fr in r[:n])
                for k, (r, lab) in enumerate(zip(runs, labels)):
                    ax.plot([fr["t"] for fr in r[:i + 1]], [_series_value(fr, key, r) for fr in r[:i + 1]],
                            color=cols[k % 4], lw=2, label=lab)
                ax.set_xlim(runs[0][0]["t"], tmax)
                vals = [_series_value(fr, key, r) for r in runs for fr in r[:n]]
                lo, hi = np.nanmin(vals), np.nanmax(vals)
                ax.set_ylim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9))
                ax.set_title(title, fontsize=9, color="#dddddd", loc="left")
                ax.tick_params(colors="#aaaaaa", labelsize=7)
                for s in ax.spines.values():
                    s.set_color("#555555")
                ax.set_facecolor("#16161a")
                ax.set_xlabel("simulation time", fontsize=8, color="#aaaaaa")
                if j == 0:
                    ax.legend(fontsize=8, facecolor="#222228", labelcolor="#dddddd", edgecolor="#444444")
            cax = fig.add_axes([0.35, 0.372, 0.3, 0.012])
            cb = fig.colorbar(ScalarMappable(plt.Normalize(0, 1), "coolwarm"), cax=cax, orientation="horizontal")
            cb.set_label("fiber direction: blue = wrapping around the organoid, red = pointing at it; yellow = linkers",
                         fontsize=9, color="#cccccc")
            cb.ax.xaxis.set_label_position("top")  # label above the bar, clear of the plot titles below
            cb.ax.tick_params(labelsize=7, colors="#aaaaaa")
            fig.suptitle(f"t = {runs[0][i]['t']:g}", fontsize=16, y=0.99)
            writer.grab_frame()
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--grid", nargs="+", metavar="RUN", help="up to 4 runs side by side, fibers coloured by direction")
    ap.add_argument("--fixed-view", action="store_true",
                    help="with --grid: keep each slice window fixed at the start position (shows drift)")
    ap.add_argument("--series", nargs=3, default=["order_near", "rot_near", "n_cells"], choices=sorted(GRID_SERIES),
                    help="with --grid: the three time series under the slices")
    ap.add_argument("--labels", nargs="+", default=["before fix", "after fix"])
    ap.add_argument("--out")
    ap.add_argument("--png", help="with --compare: also write a static figure of the time series")
    ap.add_argument("--fps", type=int, default=6)
    a = ap.parse_args()
    if a.grid:
        print(render_grid(a.grid, a.labels, a.out or "grid.mp4", fps=a.fps, series=a.series, fixed_view=a.fixed_view))
    elif a.compare:
        if a.png:
            print(plot_compare_png(a.compare[0], a.compare[1], a.png, *a.labels))
        print(render_compare(a.compare[0], a.compare[1], a.out or "compare.mp4", *a.labels, fps=a.fps))
    elif a.run_dir:
        print(render_run(a.run_dir, a.out, fps=a.fps))
    else:
        ap.print_help()
        sys.exit(1)
