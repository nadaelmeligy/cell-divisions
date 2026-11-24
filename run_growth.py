#!/usr/bin/env python3
"""
run_growth.py -- grow and divide the cells of the organoid, keep the cell-ECM linkers across
engine restarts, and finish with an MP4. One command, no ParaView:

    python3 run_growth.py --name my_first_run

Output in runs/my_first_run/:
    movie.mp4     3D view, mid-plane slice and time series
    metrics.csv   one row per time unit (E_* energy columns come from the engine log, rounded to 0.1)
    frames.pkl    the data behind the movie (re-render with: python3 growthviz.py runs/my_first_run)
    state.json    the last state; continue with --start-from runs/my_first_run --name <new name>
    seg_*/        one folder per engine run (only the last two, and the times in CONFIG
                  ecm_snapshots, keep the 3 MB ECM file)

THE PHYSICS (why growth does not need to be inside the C++)
-----------------------------------------------------------
The C++ engine moves every cell corner and fiber point with overdamped dynamics, dr/dt = mu F,
for the energy E = sum_cells kv (V - V0)^2 + (S - S0)^2 + fiber springs + linker springs
(+ contact). A cell relaxes to its target volume in much less than one time unit, while growth
takes tens of time units. So the mechanics is always close to equilibrium, and growth can be a
slow change of each cell's target volume V0 (how much material the cell has). We alternate:

    run the engine for one segment (1 time unit)  ->  read each cell's volume V
    pressure  P = -2 kv (V - V0)          (the engine's own formula, Energy/Volume.cpp)
    growth    V0 -> V0 * exp(rate * segment),  rate = (ln 2 / doubling_time) * max(0, 1 - P / P_h)
    division  when V0 >= 2 * V_birth: cut through the cell's centroid, perpendicular to its
              longest axis (Hertwig's rule); each daughter gets the mother's V0 times its share
              of the two daughters' volume
    linkers   decided by linkers.py (link_rules = "biology")
    restart the engine from the new geometry, with the linkers in links.input

Every cell grows from t = t_link on (nothing grows before, so runs that start from the same
state stay identical until then). Squeezed cells (high P) grow slower and stop at P_h; cells
pulled by linkers (lower P) grow faster. Initial cell "ages" are random, so divisions are not
synchronised.

All parameters are in CONFIG below; the common ones can be changed on the command line
(python3 run_growth.py --help). Needs Python 3.9+, numpy, scipy, matplotlib and ffmpeg.
"""
import argparse
import contextlib
import csv
import io
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import growthviz as gv  # noqa: E402
import linkers  # noqa: E402

CONFIG = dict(
    name="growth",
    engine="build/tvm",               # the engine built by ./compile.sh
    init_dir="sample_init_29cells",   # folder with sample.topo + ECM.topo to start from
    restore_links=True,               # pass each run's linkers to the next one (needs this branch's engine)
    seed=1,
    # --- time -------------------------------------------------------------------------
    t_start=0,        # first time (whole number: the engine names its files by floor(t))
    t_link=5.0,       # linkers are created, and growth starts, at this time
    t_end=40.0,       # last time
    segment=1,        # time units per engine run (whole number, same reason)
    dt=0.005,         # engine time step
    dtr=0.05,         # interval between neighbor-swap checks
    reconnect_origin=None,
    engine_seed=None,
    validate_topology=True,
    temperature=1e-4, # engine noise (conf "T")
    # --- cells ------------------------------------------------------------------------
    shape_index=5.2,  # S0 = shape_index * V0^(2/3): keeps the cell's shape stiffness as it grows
    kv=10.0,
    face_center_forces="legacy",
    link_rest="fixed",       # CHANGED 2026-11-24: "attach" = passive linkers (rest length = length when attaching)
    V_birth=0.7,      # target volume of a newborn cell; division when V0 reaches 2 * V_birth
    # --- growth law -------------------------------------------------------------------
    growth_law="pressure",   # "pressure": rate = g max(0, 1 - P/P_h);  "constant": rate = g
    doubling_time=20.0,      # doubling time of V0 for an unsqueezed cell (P = 0); g = ln 2 / this
    P_h=6.0,                 # homeostatic pressure: growth stops at P >= P_h
    # --- ECM and linkers (same meaning as the conf file) ------------------------------
    fiber=(10.0, 2.8284271, 0.001),               # ks, l0, kb
    link=(40, 10.0, 1.5, 0.2, 0.004, 1000.0),     # N, ks, l0, l1, shrink speed, shrink start time
                                                  # (N caps the engine's own linkers; Shabeeb's conf has 100)
    # --- how linkers form, break and survive division (see linkers.py) ----------------
    link_rules="biology",    # "biology": rules in linkers.py; "engine": original rules (one per cell, made once)
    link_area=1.0,           # contact area per linker: a surface cell holds min(4, max(1, round(A / link_area)))
    link_max_per_cell=4,     # ... at most this many (adhesion number saturates at large area)
    k_on=1.0,                # per time unit: an empty slot attaches (daughters re-attach within ~1 time unit)
    k_off=0.1,               # per time unit: a linker breaks (mean lifetime 10). The engine also removes a
                             # linker whose face is deleted by a neighbour swap; that is counted separately.
    capture=3.0,             # a face can grab any ECM node within this distance, on the outside of the cell
    # CHANGED 2026-11-24 (Matrigel-like mesh): how many linkers one ECM node may hold. 1 = the original rule
    # (every collagen-like run). The mesh merges the 3 crosslink nodes of each lattice site into one node, so
    # it uses 3 to keep the binding slots per site (experiments/matrigel/README.md). Bookkeeping only: the
    # forces of several linkers on one node simply add (Energy/FiberLink.cpp).
    link_node_capacity=1,
    # --- cell-fiber contact (fibers cannot enter cells) --------------------------------
    contact=(50.0, 0.2),     # k (per unit fiber length), range (fiber "thickness"). k = 0: off
    # --- speed ------------------------------------------------------------------------
    division="memory",       # CHANGED 2026-11-25: "memory" = divide_cells_memory (same divisions, no re-reading
                             # of the tissue for every cut); "reload" = the original divide_cells, for checks
    # --- output -----------------------------------------------------------------------
    keep_raw=False,   # True keeps every file the engine writes (hundreds of MB)
    ecm_snapshots=(5, 10, 20, 30, 40, 50, 60, 70, 80),   # times whose 3 MB ECM file is kept, for later analysis
    movie=True,
)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

# ----------------------------------------------------------------------------------
# engine I/O
# ----------------------------------------------------------------------------------

def link_rest_length(cfg, t, end_of_segment=False):
    """Linker rest length of the engine's shrink rule (Run.cpp) in one continuous run: one decrease of
    'shrink speed' at each time start + 10 k (k = 0, 1, ...), never below l1. A decrease due at time
    t happens in the engine's first step at t; when t ends a segment that is its very last step, so
    the state written at t (a frame) has the rest length before it: use end_of_segment=True there."""
    _, _, l0, l1, speed, start = cfg["link"][:6]
    if speed <= 0 or t < start - 1e-9 or (end_of_segment and t <= start + 1e-9):
        return l0
    if end_of_segment:
        n = int(np.ceil((t - start) / 10.0 - 1e-6))        # decreases strictly before t
    else:
        n = int(np.floor((t - start) / 10.0 + 1e-6)) + 1   # decreases at or before t
    return max(l1, l0 - speed * n)


def write_conf(folder, cfg, t0, t1, *, runtime_options=False):
    link = list(cfg["link"])
    # The engine restarts its shrink counter at 0 and the rest length at the conf value in every segment,
    # so after the shrink start it used to replay the decreases in the first steps of each segment
    # (linkers briefly too slack). Write the rest length already due at t0 instead, and move the shrink
    # start to the next due decrease, so the engine just continues the schedule.
    _, _, l0, l1, speed, start = link[:6]
    if speed > 0 and t0 > start + 1e-9:
        n_done = int(np.floor((t0 - start) / 10.0 + 1e-6)) + 1
        link[2] = link_rest_length(cfg, t0)
        link[5] = start + 10.0 * n_done
        link[4] = min(speed, max(link[2] - l1, 0.0)) or speed   # the next decrease stops at l1
    if abs(cfg["t_link"] - 500.0) > 1e-9:
        link.append(cfg["t_link"])   # optional 7th number; Shabeeb's engine only accepts 6
    num = lambda x: f"{x:.17g}"
    lines = [f"time {num(t0)} {num(t1)} {num(cfg['dt'])}",
             f"dump {'topo' if runtime_options and not cfg['keep_raw'] else 'vtk'} {num(cfg['segment'])}",
             f"log {num(cfg['segment'])}",
             "s0 5.2 1",
             "Lth 0.02",
             f"T {num(cfg['temperature'])}",
             f"kv {num(cfg['kv'])}",
             "box 64 64 64 p p p",
             # 4th number only for the worm-like-chain bending (engines before 2026-11 accept 3 numbers)
             "fiber " + " ".join(num(x) for x in cfg["fiber"]) + (" 1" if cfg.get("fiber_bend") == "wlc" else ""),
             "link " + " ".join(num(x) for x in link)]
    if runtime_options:
        lines.extend([
            f"dtr {num(cfg['dtr'])}",
            f"reconnect_origin {num(cfg['reconnect_origin'])}",
            f"seed {cfg['engine_seed'] if cfg['engine_seed'] is not None else cfg['seed']}",
            f"validate_topology {'on' if cfg['validate_topology'] else 'off'}",
        ])
    if cfg["contact"][0] > 0:
        lines.append(f"contact {num(cfg['contact'][0])} {num(cfg['contact'][1])}")
    if cfg["link_rules"] == "biology":
        lines.append("linkrules python")  # the engine uses our linkers and makes none itself
    if cfg.get("face_center_forces") == "exact":
        lines.append("face_center_forces exact")
    if cfg.get("link_rest", "fixed") == "attach":
        lines.append("link_rest attach")   # CHANGED 2026-11-24: passive linkers
    (folder / "conf").write_text("\n".join(lines) + "\n")


def write_cell_parameters(folder, cfg, V0):
    with open(folder / "cellParameters.input", "w") as f:
        f.write("id v0 s0 kv\n")
        for cid in sorted(V0):
            v0 = V0[cid]
            f.write(f"{cid} {v0:.17g} {cfg['shape_index'] * v0 ** (2.0 / 3.0):.17g} {cfg['kv']:.17g}\n")


def run_engine(folder, engine):
    with open(folder / "run.log", "w") as out:
        res = subprocess.run([str(engine)], cwd=folder, stdout=out, stderr=subprocess.STDOUT, timeout=3600)
    text = (folder / "run.log").read_text()
    # "-nan" (glibc, e.g. on Condor) and "inf" also mean a blown-up segment
    if res.returncode != 0 or re.search(r"(?<![\w.])-?(nan|inf)(?![\w.])", text.lower()) or "conf file error" in text:
        raise RuntimeError(f"engine failed in {folder} (see run.log)")
    return text

# ----------------------------------------------------------------------------------
# division
# ----------------------------------------------------------------------------------

def check_division(topo_path, mother, daughters, vol_mother, messages, dicts=None):
    """Return (ok, reason, [vol_a, vol_b]). Strict topology checks so the engine never gets a broken cell."""
    if "Error" in messages:
        return False, "division code reported: " + messages.strip().splitlines()[0], None
    # CHANGED 2026-11-25 (speed): divide_cells_memory passes the topology it just wrote as dicts
    # (topo_dicts), so the file need not be read back. Same values: write_topo keeps the order and
    # writes 17 digits, which read back exactly. Old: V, E, P, C = gv.read_topo(topo_path)
    V, E, P, C = gv.read_topo(topo_path) if dicts is None else dicts
    if mother in C or any(d not in C or C[d][1] <= 0 for d in daughters):
        return False, "daughter cells missing", None
    pc = gv.polygon_cells(C)
    if any(len(o) > 2 for o in pc.values()):
        return False, "a face belongs to more than two cells", None
    affected = {cid for daughter in daughters for face in C[daughter][0] for cid, _ in pc[face]}
    try:
        for cid in sorted(affected):
            faces = C[cid][0]
            if len(faces) < 4 or len(set(faces)) != len(faces):
                return False, f"cell {cid} has missing or repeated faces", None
            edges = Counter(e for face in faces for e in P[face])
            if any(count != 2 for count in edges.values()):
                return False, f"cell {cid} has an open shell edge", None
            loops = gv.oriented_face_loops(faces, P, E)
            vertices = {v for loop in loops.values() for v in loop}
            if len(vertices) - len(edges) + len(faces) != 2:
                return False, f"cell {cid} fails the shell Euler relation", None
            neighbors = Counter(owner for face in faces for owner, _ in pc[face] if owner != cid)
            if any(count > 1 for count in neighbors.values()):
                return False, f"cell {cid} shares multiple faces with a neighbor", None
        vols = [gv.cell_geometry(d, V, E, P, C)[0] for d in daughters]
    except (KeyError, ValueError, IndexError) as exc:
        return False, f"invalid shell after division ({exc})", None
    if any(not math.isfinite(volume) or volume <= 0 for volume in vols):
        return False, "daughter has nonpositive or nonfinite volume", None
    if abs(sum(vols) - vol_mother) / vol_mother > 0.02:
        return False, f"volume not conserved ({sum(vols):.4f} vs {vol_mother:.4f})", None
    if min(vols) / max(vols) < 0.4:
        return False, f"very unequal daughters ({vols[0]:.3f}, {vols[1]:.3f})", None
    return True, "", vols


def divide_cells(folder, topo_name, mothers, V0, rng):
    """Divide every mother in turn (re-loading the topology between divisions).
    Returns (topology file to start the next segment from, number of divisions, {mother: (a, b)})."""
    from toolbox.cellDivision import cellDivision
    current, done, divisions = topo_name, 0, {}
    for k, m in enumerate(mothers):
        V, E, P, C = gv.read_topo(folder / current)
        if m not in C:
            continue
        vol_m, centroid, pts = gv.cell_geometry(m, V, E, P, C)
        w, vecs = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))
        long_axis = vecs[:, -1]  # Hertwig's rule: the cut plane is perpendicular to the longest axis
        ok = False
        for attempt in range(5):
            axis = long_axis if attempt == 0 else long_axis + 0.35 * rng.normal(size=3)
            axis = axis / np.linalg.norm(axis)
            divider = cellDivision.from_spheroid_config(str(folder) + os.sep, current)
            divider.sample.cells_[m].center_ = centroid  # cut through the true centroid -> ~equal halves
            divider.set_motherCellID(m)
            divider.set_division_axis(list(axis))
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    divider.evaluatePostDivisionTopology()
            except Exception as exc:  # degenerate cut (e.g. plane through a vertex)
                buf.write(f"Error: {exc}")
            out_name = f"divided_{k:02d}.topo"
            if "Error" not in buf.getvalue():
                divider.sample.write_topo(out_name)
                ok, why, vols = check_division(folder / out_name, m, divider.daughterCellIDs, vol_m, buf.getvalue())
            else:
                why = buf.getvalue().strip().splitlines()[0]
            if ok:
                break
        if not ok:
            log(f"  cell {m}: division skipped this step ({why}); it will try again next step")
            continue
        a, b = (int(x) for x in divider.daughterCellIDs)
        v0 = V0.pop(m)
        V0[a], V0[b] = v0 * vols[0] / sum(vols), v0 * vols[1] / sum(vols)
        divisions[m] = (a, b)
        current, done = out_name, done + 1
    return current, done, divisions


# CHANGED 2026-11-25 (speed): divide_cells above re-reads the whole tissue from disk for every
# mother and every attempt (gv.read_topo, then Spheroid.from_config with all its derived
# quantities), and check_division reads the result back once more. At ~900 cells and ~40
# divisions per time unit that was most of a segment's time. divide_cells_memory makes the same
# divisions on one tissue kept in memory. It is the default (CONFIG division = "memory");
# division = "reload" keeps the original path, for checking that the two agree.
def topo_dicts(sample):
    """(V, E, P, C) of an in-memory toolbox Sample, exactly as gv.read_topo would return them for
    the file sample.write_topo writes: same order (write_topo writes the dicts in insertion order)
    and the same numbers (17 significant digits read back exactly)."""
    V = {int(i): (float(v.position_[0]), float(v.position_[1]), float(v.position_[2])) for i, v in sample.vertices_.items()}
    E = {int(i): (int(e.vertices_[0]), int(e.vertices_[1])) for i, e in sample.edges_.items()}
    P = {int(i): [int(x) for x in p.edges_] for i, p in sample.polygons_.items()}
    C = {int(i): ([int(x) for x in c.polygons_], int(c.type_)) for i, c in sample.cells_.items()}
    return V, E, P, C


def divide_cells_memory(folder, topo_name, mothers, V0, rng):
    """Same divisions, in the same order, with the same random numbers as divide_cells, but on one
    tissue kept in memory. What divide_cells gets from re-reading, this does by hand:
      - cellDivision.evaluatePostDivisionTopology reads Vertex.is_daughter_ of the tissue's
        vertices (toolbox/cellDivision.py, the sign tests): a fresh load sets it False for all, but
        in memory the vertices made by earlier divisions keep True, so it is reset before each cut;
      - the other flags it sets (is_mother_, intersection_vertex_) are only on edges and faces that
        the same division deletes;
      - it reads polygon.vertices_ of the mother's faces, which is recomputed before each cut (below).
    A failed attempt may leave the tissue half cut, so the next attempt reloads the last good file,
    as divide_cells does for every attempt. divided_kk.topo is still written after each cut
    (the engine and linkers.py read the last one).
    Loading reads only what the division and write_topo use: the topology (load_config) and the box
    (load_conf_file). Spheroid.from_config also computes volumes, areas, surface flags and more
    (~10 s at 900 cells), none of which the division reads, except polygon.vertices_
    (arrange_polygon_vertices): that is recomputed for the mother's faces just before each cut, with
    the same function on the same edges, so it is what a fresh load would give.
    Checked file-for-file against divide_cells: experiments/speed/check_division_memory.py."""
    from toolbox.cellDivision import cellDivision
    from toolbox.spheroid import Spheroid

    def load(name):
        s = Spheroid()
        s.time_ = 0
        s.set_config_dir(str(folder) + os.sep)
        s.set_file(name)
        s.load_config()      # vertices, edges, faces, cells (the same parse as Spheroid.from_config)
        s.load_conf_file()   # box size
        return s

    current, done, divisions = topo_name, 0, {}
    sample = load(current)
    # A cell across the periodic box is unwrapped while loading (tissue.py,
    # calculate_boundary_cell_attributes), which re-reading would redo: keep the original path then.
    # Same test as load_cross_boundary_attributes: an edge longer than half the box.
    # (Never the case for an organoid in the middle of the box; a division never makes such an edge.)
    ends = np.array([[sample.vertices_[v].position_ for v in e.vertices_[:2]] for e in sample.edges_.values()])
    if np.any(np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1) > sample.boxSize_ / 2):
        return divide_cells(folder, topo_name, mothers, V0, rng)
    for k, m in enumerate(mothers):
        if sample is None:
            sample = load(current)
        V, E, P, C = topo_dicts(sample)
        if m not in C:
            continue
        vol_m, centroid, pts = gv.cell_geometry(m, V, E, P, C)
        w, vecs = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))
        long_axis = vecs[:, -1]  # Hertwig's rule: the cut plane is perpendicular to the longest axis
        ok = False
        for attempt in range(5):
            axis = long_axis if attempt == 0 else long_axis + 0.35 * rng.normal(size=3)
            axis = axis / np.linalg.norm(axis)
            if sample is None:  # the last attempt changed the tissue: start again from the last good file
                sample = load(current)
            for vertex in sample.vertices_.values():
                vertex.is_daughter_ = False  # as after a fresh load (see the docstring)
            for pid in sample.cells_[m].polygons_:  # as arrange_polygon_vertices does on a fresh load
                poly = sample.polygons_[pid]
                poly.vertices_ = sample.resolve_polygon_edge_connectivity([sample.edges_[e].vertices_ for e in poly.edges_])
            divider = cellDivision()
            divider.sample = sample
            divider.sample.cells_[m].center_ = centroid  # cut through the true centroid -> ~equal halves
            divider.set_motherCellID(m)
            divider.set_division_axis(list(axis))
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    divider.evaluatePostDivisionTopology()
            except Exception as exc:  # degenerate cut (e.g. plane through a vertex)
                buf.write(f"Error: {exc}")
            out_name = f"divided_{k:02d}.topo"
            if "Error" not in buf.getvalue():
                divider.sample.write_topo(out_name)
                ok, why, vols = check_division(folder / out_name, m, divider.daughterCellIDs, vol_m, buf.getvalue(),
                                               dicts=topo_dicts(sample))
            else:
                why = buf.getvalue().strip().splitlines()[0]
            if ok:
                break
            sample = None  # this attempt may have cut part of the tissue: reload before the next one
        if not ok:
            log(f"  cell {m}: division skipped this step ({why}); it will try again next step")
            continue
        a, b = (int(x) for x in divider.daughterCellIDs)
        v0 = V0.pop(m)
        V0[a], V0[b] = v0 * vols[0] / sum(vols), v0 * vols[1] / sum(vols)
        divisions[m] = (a, b)
        current, done = out_name, done + 1
    return current, done, divisions

# ----------------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------------

def growth_rate(cfg, P):
    g = math.log(2.0) / cfg["doubling_time"]
    if cfg["growth_law"] == "constant":
        return g
    return g * max(0.0, 1.0 - P / cfg["P_h"])


def cleanup(folder, keep):
    for f in folder.iterdir():
        if f.is_file() and f.name not in keep:
            f.unlink()


def resolve_saved(value, src):
    """A path stored in state.json: relative to the run folder (current format) or absolute (older runs)."""
    p = Path(value)
    if not p.is_absolute():
        return src / p
    if p.exists():
        return p
    return src / p.parent.name / p.name   # an older run that was moved or copied


def main(cfg, *, reset_python_rng=False, reset_engine_rng=False):
    if any(not math.isfinite(t) or t != int(t)
           for t in (cfg["t_start"], cfg["t_end"], cfg["segment"])) or cfg["segment"] <= 0:
        sys.exit("t_start, t_end and segment must be whole numbers, with segment > 0")
    if not (math.isfinite(cfg["dt"]) and math.isfinite(cfg["dtr"]) and
            0 < cfg["dt"] <= min(cfg["dtr"], cfg["segment"])):
        sys.exit("dt must be positive and no larger than dtr or the Python segment")
    for seed in (cfg["seed"], cfg["engine_seed"]):
        if seed is not None and not 0 <= seed < 2**64:
            sys.exit("Seeds must be unsigned 64-bit integers")
    if cfg["reconnect_origin"] is None:
        cfg["reconnect_origin"] = cfg["t_start"]
    if cfg["link_rules"] == "biology" and not cfg["restore_links"]:
        sys.exit("--no-restore-links needs --link-rules engine: with the biology rules the engine "
                 "only gets linkers through links.input")
    rng = np.random.default_rng(cfg["seed"])
    run_dir = ROOT / "runs" / cfg["name"]
    src = None
    if cfg.get("start_from"):
        src = Path(cfg["start_from"])
        src = (src if src.is_absolute() or src.exists() else ROOT / src).resolve()
        if src == run_dir.resolve() or run_dir.resolve() in src.parents:
            sys.exit("--start-from is the output folder itself; give the continued run a new --name")
        st = json.loads((src / "state.json").read_text())
        if st["t"] >= cfg["t_end"] - 1e-9:
            sys.exit(f"nothing to run: {src.name} is already at t = {st['t']:g}; give a larger --t-end")
    if run_dir.exists():
        if not cfg["overwrite"]:
            sys.exit(f"{run_dir} exists; use --overwrite or another --name")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    engine = (ROOT / cfg["engine"]).resolve()
    if not engine.exists():
        sys.exit(f"engine not found: {engine}  (run ./compile.sh)")
    # Supply placeholder file paths so binaries predating the probe exit cleanly.
    probe = subprocess.run([str(engine), "--runtime-features", "-", "-"], cwd=run_dir,
                           capture_output=True, text=True, timeout=10)
    features = probe.stdout.strip() if probe.returncode == 0 else ""
    # CHANGED 2026-11-24: + "tvm-runtime-v3" (several linkers per ECM node). Old lines:
    #     runtime_options = features in ("tvm-runtime-v1", "tvm-runtime-v2")
    #     if cfg.get("link_rest", "fixed") == "attach" and features != "tvm-runtime-v2":
    runtime_options = features in ("tvm-runtime-v1", "tvm-runtime-v2", "tvm-runtime-v3")
    if cfg.get("link_rest", "fixed") == "attach" and features not in ("tvm-runtime-v2", "tvm-runtime-v3"):
        sys.exit("This TVM binary does not support --link-rest attach (rebuild with ./compile.sh)")
    if cfg.get("link_node_capacity", 1) > 1 and features != "tvm-runtime-v3":
        # an older engine would silently drop the 2nd and 3rd linker on a node (Run::loadLinks)
        sys.exit("This TVM binary keeps only one linker per ECM node: rebuild with ./compile.sh "
                 "for --link-node-capacity > 1")
    if not runtime_options:
        if cfg["engine_seed"] is not None or not math.isclose(cfg["dtr"], 10 * cfg["dt"]):
            sys.exit("This TVM binary does not support --engine-seed or an independent --dtr")
        log(f"using legacy runtime settings for {engine}")

    # ---- initial state -----------------------------------------------------------
    frames, rows = [], []
    if src is not None:
        start = run_dir / "start"
        start.mkdir()
        files = {}
        for key in ("topo", "ecm", "links", "rng"):
            if key == "rng" and (reset_engine_rng or not runtime_options):
                continue
            if st.get(key):
                path = resolve_saved(st[key], src)
                shutil.copy2(path, start / path.name)
                files[key] = str(start / path.name)
        state = dict(t=st["t"], topo=files["topo"], ecm=files["ecm"], links=files.get("links"),
                     rng=files.get("rng"), V0={int(k): v for k, v in st["V0"].items()},
                     links_seeded=st.get("links_seeded", False))
        if st.get("python_rng") is not None and not reset_python_rng:
            rng.bit_generator.state = st["python_rng"]
        with open(src / "frames.pkl", "rb") as f:
            frames = pickle.load(f)          # keep the history so the movie starts at the first run's start
        with open(src / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
        ref_src = src / "reference_ECM.topo"
        shutil.copy2(ref_src if ref_src.exists() else files["ecm"], run_dir / "reference_ECM.topo")
        log(f"starting from {src.name} at t = {state['t']:g}")
    else:
        init = ROOT / cfg["init_dir"]
        _, _, _, C = gv.read_topo(init / "sample.topo")
        V0 = {cid: cfg["V_birth"] * 2.0 ** rng.uniform(0.0, 0.95) for cid, (_, typ) in C.items() if typ > 0}
        state = dict(t=float(cfg["t_start"]), topo=str(init / "sample.topo"), ecm=str(init / "ECM.topo"),
                     links=None, V0=V0)
        shutil.copy2(init / "ECM.topo", run_dir / "reference_ECM.topo")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    ref_ecm = run_dir / "reference_ECM.topo"   # the gel at the start: fiber rotation/movement is measured from it

    l0_fiber = cfg["fiber"][1]
    previous = []
    while state["t"] < cfg["t_end"] - 1e-9:
        t0 = state["t"]
        t1 = min(t0 + cfg["segment"], cfg["t_end"])
        tag = f"{int(math.floor(t1 + 1e-6)):07d}"
        seg = run_dir / f"seg_{tag}"
        seg.mkdir()
        shutil.copy2(state["topo"], seg / "sample.topo")
        shutil.copy2(state["ecm"], seg / "ECM.topo")
        if cfg["restore_links"] and state["links"]:
            shutil.copy2(state["links"], seg / "links.input")
        if state.get("rng"):
            shutil.copy2(state["rng"], seg / "rng.input")
        write_conf(seg, cfg, t0, t1, runtime_options=runtime_options)
        V0 = state["V0"]
        write_cell_parameters(seg, cfg, V0)

        tic = time.time()
        run_engine(seg, engine)
        topo, ecm, links = seg / f"{tag}.sample.topo", seg / f"{tag}.ECM.topo", seg / f"{tag}.linkInfo.txt"
        vols = gv.read_cell_volumes(seg / "cellVolume.txt")[int(round(t1))]
        energies = gv.read_last_log_row(seg / "run.log")

        cells = {}
        for cid, v0 in V0.items():
            if cid in vols:
                cells[cid] = dict(V=vols[cid], V0=v0, P=-2.0 * cfg["kv"] * (vols[cid] - v0))
        Ps = np.array([c["P"] for c in cells.values()])

        # growth (only from t_link on, so runs that start from the same state stay identical until then)
        growing = t0 >= cfg["t_link"] - 1e-9
        if growing:
            for cid, c in cells.items():
                V0[cid] = V0[cid] * math.exp(growth_rate(cfg, c["P"]) * (t1 - t0))
        mothers = [cid for cid, v0 in V0.items() if v0 >= 2.0 * cfg["V_birth"]]
        n_div, divisions = 0, {}
        next_topo = topo
        if mothers:
            # CHANGED 2026-11-25 (speed): the in-memory path unless division = "reload".
            # Old: name, n_div, divisions = divide_cells(seg, topo.name, mothers, V0, rng)
            divide = divide_cells if cfg.get("division", "memory") == "reload" else divide_cells_memory
            name, n_div, divisions = divide(seg, topo.name, mothers, V0, rng)
            next_topo = seg / name

        # linkers for the next segment
        next_links, lstats = links, {}
        if cfg["link_rules"] == "biology" and t1 >= cfg["t_link"] - 1e-9:
            survived = gv.read_links(links)
            written = gv.read_links(seg / "links.input")
            new, lstats = linkers.update_links(topo, next_topo, ecm, survived, divisions, cfg, rng,
                                               dt=t1 - t0, seed=not state.get("links_seeded"))
            lstats["lost_in_engine"] = len(set(written) - set(survived))  # faces deleted by neighbour swaps
            state["links_seeded"] = True
            next_links = seg / "links_next.txt"
            # CHANGED 2026-11-24: passive linkers keep their rest length across segments
            rests = gv.read_link_rests(links) if cfg.get("link_rest", "fixed") == "attach" else None
            linkers.write_links(next_links, new, rests)

        extra = dict(n_div=n_div, mean_P=float(Ps.mean()), median_P=float(np.median(Ps)),
                     frac_stalled=float(np.mean(Ps >= cfg["P_h"])), mean_V=float(np.mean([c["V"] for c in cells.values()])),
                     mean_V0=float(np.mean([c["V0"] for c in cells.values()])), **energies,
                     **{f"links_{k}": v for k, v in lstats.items()})
        # the engine's rest length at the end of the segment (it shrinks when cells pull)
        fr = gv.make_frame(t1, topo, ecm, links, cells, l0_fiber, link_rest_length(cfg, t1, end_of_segment=True), extra, ref_ecm=ref_ecm)
        frames.append(fr)
        rows.append({k: fr[k] for k in ("t", "n_cells", "n_links", "n_div", "mean_V", "mean_V0", "mean_P",
                                        "median_P", "frac_stalled", "mean_abs_fiber_strain_near",
                                        "fiber_points_inside", "radius", "surface_radius",
                                        "order_near", "order_far", "rot_near", "rot_far",
                                        "disp_near", "disp_far", "n_fibers_near")}
                    | {k: energies.get(k) for k in ("E_link", "E_ecm_stretch", "E_total")}
                    | {k: extra.get(k) for k in ("links_formed", "links_broken", "links_lost_in_engine",
                                                 "links_released_at_division", "links_tethers",
                                                 "links_linked_surface_fraction", "links_links_per_surface_cell")})
        churn = (f"  +{lstats['formed']} -{lstats['broken']} linkers (-{lstats['lost_in_engine']} on deleted faces), "
                 f"{100 * lstats['linked_surface_fraction']:.0f}% of surface cells linked" if lstats else "")
        log(f"t={t1:6.1f}  cells={fr['n_cells']:4d}  linkers={fr['n_links']:3d}  divisions={n_div}  "
            f"mean P={extra['mean_P']:.2f}  stalled={100 * extra['frac_stalled']:.0f}%  "
            f"link energy={energies.get('E_link', 0):.1f}{churn}  ({time.time() - tic:.0f}s)")

        # keep only the small files (unless keep_raw); the 3 MB ECM file only for the last two segments
        if not cfg["keep_raw"]:
            keep = {"conf", "cellParameters.input", "links.input", "run.log", "cellVolume.txt",
                    "rng.output", "run_summary.txt", "reconnections.txt",
                    topo.name, ecm.name, links.name, next_topo.name, "links_next.txt"}
            cleanup(seg, keep)
            previous.append(seg)
            if len(previous) > 2:
                old = previous.pop(0)
                if int(old.name[4:]) not in set(int(x) for x in cfg["ecm_snapshots"]):
                    for f in old.glob("*.ECM.topo"):
                        f.unlink()

        state = dict(t=t1, topo=str(next_topo), ecm=str(ecm), links=str(next_links), V0=V0,
                     rng=str(seg / "rng.output") if runtime_options else None,
                     python_rng=rng.bit_generator.state,
                     links_seeded=state.get("links_seeded", False))
        saved = {**state, "V0": {str(k): v for k, v in V0.items()}}
        for key in ("topo", "ecm", "links", "rng"):
            if state[key] is not None:
                saved[key] = os.path.relpath(state[key], run_dir)
        (run_dir / "state.json").write_text(json.dumps(saved, indent=1))
        with open(run_dir / "frames.pkl", "wb") as f:
            pickle.dump(frames, f)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with open(run_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    if cfg["movie"] and frames:
        log("rendering movie ...")
        log("wrote " + gv.render_run(str(run_dir)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--engine")
    ap.add_argument("--init-dir")
    ap.add_argument("--start-from", help="continue from another run's last state (its runs/<name> folder); "
                                         "use a new --name for the continued run")
    ap.add_argument("--no-restore-links", action="store_true",
                    help="do NOT pass linkers between runs (original behaviour; use with --link-rules engine)")
    ap.add_argument("--t-start", type=float)
    ap.add_argument("--t-link", type=float)
    ap.add_argument("--t-end", type=float)
    ap.add_argument("--dt", type=float, help="C++ integration timestep")
    ap.add_argument("--dtr", type=float, help="simulation time between neighbor-swap checks")
    ap.add_argument("--doubling-time", type=float)
    ap.add_argument("--P-h", type=float, dest="P_h")
    ap.add_argument("--growth-law", choices=["pressure", "constant"])
    ap.add_argument("--face-center-forces", choices=["legacy", "exact"],
                    help="use the full face-center derivative from Tao Zhang's Rust TVM")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--engine-seed", type=int, help="start a new mechanical noise stream with this seed")
    ap.add_argument("--validate-topology", action=argparse.BooleanOptionalAction, default=None,
                    help="validate the starting mesh and proposed neighbor swaps (default on)")
    ap.add_argument("--link-rules", choices=["biology", "engine"],
                    help="biology: linkers.py rules (default); engine: original one-per-cell rule")
    ap.add_argument("--contact", type=float, nargs=2, metavar=("K", "RANGE"),
                    help="cell-fiber contact: stiffness per unit fiber length and range; 0 0.2 = off")
    ap.add_argument("--k-on", type=float, help="linker attachment rate per empty slot (per time unit)")
    # CHANGED 2026-11-24: see link_node_capacity in CONFIG
    ap.add_argument("--link-node-capacity", type=int, help="linkers one ECM node may hold (default 1; mesh gel: 3)")
    ap.add_argument("--k-off", type=float, help="linker breaking rate (per time unit)")
    ap.add_argument("--link-area", type=float, help="contact area per linker")
    ap.add_argument("--link-rest", choices=["fixed", "attach"],
                    help="linker rest length: 'fixed' = the link line's l0 (1.5) for every linker (default); 'attach' "
                         "= passive linkers, each with its own length when it attached (no built-in pull)")
    ap.add_argument("--link-ks", type=float, help="linker spring stiffness (0 retains attachment bookkeeping and removes spring forces)")
    ap.add_argument("--shrink-start", type=float, help="time at which linkers start pulling (rest length shrinks)")
    ap.add_argument("--shrink-speed", type=float, help="rest-length decrease every 10 time units")
    ap.add_argument("--fiber-ks", type=float, help="fiber stretch stiffness ks (default 10). The engine's time "
                    "step is explicit: keep 8 * ks * dt below ~1, i.e. ks <= 25 at dt = 0.005")
    ap.add_argument("--fiber-kb", type=float, help="fiber bending stiffness kb (default 0.001). With --fiber-bend "
                    "wlc, kb / (ks l0^2) is the dimensionless bending rigidity of the fibers")
    ap.add_argument("--temperature", type=float, help="mechanical noise amplitude (default 1e-4)")
    ap.add_argument("--ecm-snapshots", type=float, nargs="+", metavar="T",
                    help="times whose 3 MB ECM file is kept (default 5 10 20 30 ... 80)")
    ap.add_argument("--fiber-bend", choices=["original", "wlc"],
                    help="bending energy: original 0.5 kb (1 + cos)^2 (no stiffness against small bends), or "
                    "wlc = worm-like chain kb (1 + cos) ~ 0.5 kb (bend angle)^2 (needs this branch's engine)")
    # CHANGED 2026-11-25 (speed): choose the division code path; both give the same files
    ap.add_argument("--division", choices=["memory", "reload"],
                    help="memory: divide in memory (default, fast); reload: original path that re-reads the tissue for every cut")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = dict(CONFIG)
    if a.start_from:
        # A continued run keeps the physics of the run it continues (fiber, bending form, noise, contact,
        # growth law, linker settings, engine, ...); only the options given here change. Before 2026-11-19
        # it silently went back to CONFIG, e.g. a --fiber-bend wlc run continued with the original bending.
        src = Path(a.start_from)
        src = src if src.is_absolute() or src.exists() else ROOT / src
        if (src / "config.json").exists():
            old = json.loads((src / "config.json").read_text())
            skip = {"name", "start_from", "overwrite", "t_end", "t_start", "keep_raw", "movie"}
            for k, v in old.items():
                if k in skip or (k not in CONFIG and k != "fiber_bend"):
                    continue
                cfg[k] = tuple(v) if isinstance(CONFIG.get(k), tuple) and isinstance(v, list) else v
    for k in ("name", "engine", "init_dir", "t_start", "t_link", "t_end", "doubling_time", "P_h", "growth_law", "seed",
              "face_center_forces", "dt", "dtr", "engine_seed", "validate_topology",
              "link_rules", "k_on", "k_off", "link_area", "contact", "temperature", "link_rest",
              "division", "link_node_capacity"):  # CHANGED 2026-11-25: + "division"; CHANGED 2026-11-24: + "link_node_capacity"
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    link = list(cfg["link"])  # N, ks, l0, l1, shrink speed, shrink start
    if cfg.get("link_node_capacity", 1) < 1:   # CHANGED 2026-11-24
        ap.error("--link-node-capacity must be at least 1")
    if a.link_ks is not None:
        if not math.isfinite(a.link_ks) or a.link_ks < 0:
            ap.error("--link-ks must be finite and nonnegative")
        link[1] = a.link_ks
    if a.shrink_speed is not None:
        link[4] = a.shrink_speed
    if a.shrink_start is not None:
        link[5] = a.shrink_start
    cfg["link"] = tuple(link)
    fiber = list(cfg["fiber"])  # ks, l0, kb (l0 stays: the gels are built with segments of 2.83)
    if a.fiber_ks is not None:
        fiber[0] = a.fiber_ks
    if a.fiber_kb is not None:
        fiber[2] = a.fiber_kb
    cfg["fiber"] = tuple(fiber)
    if a.fiber_bend is not None:
        cfg["fiber_bend"] = a.fiber_bend
    if a.ecm_snapshots is not None:
        cfg["ecm_snapshots"] = tuple(a.ecm_snapshots)
    cfg["start_from"] = a.start_from
    cfg["restore_links"] = cfg["restore_links"] and not a.no_restore_links
    cfg["keep_raw"] = a.keep_raw or cfg["keep_raw"]
    cfg["movie"] = cfg["movie"] and not a.no_movie
    cfg["overwrite"] = a.overwrite
    main(cfg, reset_python_rng=a.seed is not None,
         reset_engine_rng=a.engine_seed is not None or a.seed is not None)
