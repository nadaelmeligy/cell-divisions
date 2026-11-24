"""
linkers.py -- how cell-ECM linkers (integrin adhesions) form, break and survive division.
Called by run_growth.py between engine segments when link_rules = "biology".

Three rules, from the biology:
  1. Contact area sets the number. A surface cell can hold
         n_max = min(link_max_per_cell, max(1, round(A / link_area)))
     linkers, A = area of its faces touching the medium (bound integrins rise with adhesive
     area, then saturate: Gallant et al. 2005, MBoC).
  2. Division. A dividing cell lets go of its adhesions (Chen et al. 2022, Nat Cell Biol)
     except one basal tether (beta4-integrin, Carroll et al. 2017, J Cell Sci, intestinal
     organoids), chosen at random among its linkers. If the tether's face was cut, its ECM
     node moves to the nearest free daughter face that passes face_ok(). The other daughter
     starts with none and re-attaches through rule 3.
  3. Turnover. Every linker breaks with probability 1 - exp(-k_off dt) per segment. Every
     empty slot, with probability 1 - exp(-k_on dt), takes the cell's free faces that pass
     face_ok() in random order and links the first one that has a free ECM node within
     `capture`, on the outward side of the face, to its nearest such node. Any fiber node can
     be grabbed (not only the fiber ends at the hole). At the very first step (seed) every
     slot tries once; it stays empty if no such face and node exist.

A linker is a (face id, ECM node id) pair, exactly the format of the engine's linkInfo.txt.
The engine runs with "linkrules python": it starts each segment from these linkers and removes
one only when a neighbour swap deletes its face (run_growth.py counts these as lost_in_engine).
"""
import math
from collections import defaultdict, Counter  # CHANGED 2026-11-24: + Counter

import numpy as np
from scipy.spatial import cKDTree

import growthviz as gv


def surface_faces(C):
    """face id -> the real cell that owns it, for faces between a real cell and the medium."""
    owner = {}
    for p, owners in gv.polygon_cells(C).items():
        reals = [c for c, t in owners if t > 0]
        medium = [c for c, t in owners if t == 0]
        if len(reals) == 1 and len(medium) == 1:
            owner[p] = reals[0]
    return owner


def face_center_area(p, V, E, P):
    c = gv.polygon_center(P[p], E, V)
    pts = np.array([V[v] for v in gv.polygon_loop(P[p], E)])
    area = 0.5 * sum(np.linalg.norm(np.cross(pts[i] - c, pts[(i + 1) % len(pts)] - c)) for i in range(len(pts)))
    return c, area


def face_ok(p, V, E, P, center, area, min_area=0.1, min_edge=0.02, max_edge=1.5):
    """Same test as the engine's Polygon::checkShape(), which the original linker rules use:
    area >= 0.1, every edge between 0.02 and 1.5 long, and at every corner each edge makes
    at least about 2.6 degrees with the line from that corner to the face centre. New linkers
    are only placed on faces that pass, so they do not sit on thin slivers."""
    if area < min_area:
        return False
    for e in P[p]:
        a, b = np.array(V[E[e][0]]), np.array(V[E[e][1]])
        l = np.linalg.norm(b - a)
        if l < min_edge or l > max_edge:
            return False
        for x, y in ((a, b), (b, a)):
            vv = (x - y) / l
            cv = (x - center) / np.linalg.norm(x - center)
            if np.dot(vv, cv) > 0.999:
                return False
    return True


def write_links(path, links, rests=None):
    """rests (CHANGED 2026-11-24, passive linkers): {(face, node): rest length} read from the engine's link file.
    A linker that keeps its face and node keeps its rest length (3rd column); a new or moved one gets none, so
    the engine sets it to its length when it loads it (it attaches then)."""
    with open(path, "w") as f:
        f.write(f"{'PolygonID':<12}{'NodeID':<12}" + (" RestLength" if rests is not None else "") + "\n")
        for p, n in links:
            r = rests.get((p, n)) if rests is not None else None
            f.write(f"{p:<12}{n:<12}" + (f" {r!r}" if r is not None else "") + "\n")


def update_links(old_topo, new_topo, ecm_path, links, divisions, cfg, rng, dt, seed=False):
    """Apply the three rules once.
    old_topo : geometry the links refer to (end of the segment, before divisions)
    new_topo : geometry the next segment starts from (after divisions)
    links    : [(face, node)] that survived the segment
    divisions: {mother id: (daughter a, daughter b)} done this step
    seed     : True the first time: every slot tries to attach once (cells embedded in the gel)
    Returns (new links, stats dict)."""
    stats = dict(formed=0, broken=0, released_at_division=0, tethers=0)
    _, _, _, Co = gv.read_topo(old_topo)
    old_pc = gv.polygon_cells(Co)
    V, E, P, C = gv.read_topo(new_topo)
    nodes, _, _ = gv.read_ecm(ecm_path)
    owner = surface_faces(C)
    geom = {p: face_center_area(p, V, E, P) for p in owner}
    faces_of = defaultdict(list)
    for p, c in owner.items():
        faces_of[c].append(p)
    center = {}
    for c, fs in faces_of.items():
        pts = np.array([V[v] for p in C[c][0] for e in P[p] for v in E[e]])
        center[c] = pts.mean(axis=0)

    def old_owner(p):
        reals = [c for c, t in old_pc.get(p, []) if t > 0]
        return reals[0] if reals else None

    # ---- rule 2: division -------------------------------------------------------
    kept, by_mother = [], defaultdict(list)
    for p, n in links:
        if n not in nodes:
            continue
        m = old_owner(p)
        (by_mother[m] if m in divisions else kept).append((p, n))
    used_faces = {p for p, _ in kept}
    for m, ls in by_mother.items():
        a, b = divisions[m]
        p, n = ls[rng.integers(len(ls))]          # the basal tether
        stats["released_at_division"] += len(ls) - 1
        if not (p in owner and owner[p] in (a, b)):
            # its face was cut: hand the same ECM node to the nearest free daughter face
            # that can hold a linker (not one of the thin slivers the cut leaves behind)
            cands = [q for q in owner if owner[q] in (a, b) and q not in used_faces
                     and face_ok(q, V, E, P, *geom[q])]
            if not cands:
                stats["released_at_division"] += 1
                continue
            x = np.array(nodes[n])
            p = min(cands, key=lambda q: np.linalg.norm(geom[q][0] - x))
        kept.append((p, n))
        used_faces.add(p)
        stats["tethers"] += 1

    # faces that stopped being on the surface lose their linker
    # CHANGED 2026-11-24: a node may hold up to cfg["link_node_capacity"] linkers (default 1 = the original
    # rule). Old lines:
    #     seen_f, seen_n, links = set(), set(), []
    #     ... if p in owner and p not in seen_f and n not in seen_n: ... seen_n.add(n)
    m_node = int(cfg.get("link_node_capacity", 1))
    seen_f, seen_n, links = set(), Counter(), []
    for p, n in kept:
        if p in owner and p not in seen_f and seen_n[n] < m_node:
            links.append((p, n))
            seen_f.add(p)
            seen_n[n] += 1
        else:
            stats["broken"] += 1

    # ---- rule 3a: turnover ------------------------------------------------------
    if not seed:
        p_off = 1.0 - math.exp(-cfg["k_off"] * dt)
        survive = [l for l in links if rng.random() >= p_off]
        stats["broken"] += len(links) - len(survive)
        links = survive

    # ---- rule 1: capacity from contact area -------------------------------------
    area = {c: sum(geom[p][1] for p in fs) for c, fs in faces_of.items()}
    cap = {c: min(cfg["link_max_per_cell"], max(1, int(round(A / cfg["link_area"])))) for c, A in area.items()}
    per_cell = defaultdict(list)
    for l in links:
        per_cell[owner[l[0]]].append(l)
    links = []
    for c, ls in per_cell.items():
        rng.shuffle(ls)
        links += ls[:cap.get(c, 0)]
        stats["broken"] += max(0, len(ls) - cap.get(c, 0))
        per_cell[c] = ls[:cap.get(c, 0)]

    # ---- rule 3b: formation -----------------------------------------------------
    used_faces = {p for p, _ in links}
    used_nodes = Counter(n for _, n in links)   # CHANGED 2026-11-24: was a set {n for _, n in links}
    ids = np.array(list(nodes))
    tree = cKDTree(np.array([nodes[i] for i in ids]))
    p_on = 1.0 if seed else 1.0 - math.exp(-cfg["k_on"] * dt)
    cells = list(cap)
    rng.shuffle(cells)
    for c in cells:
        for _ in range(cap[c] - len(per_cell[c])):
            if rng.random() >= p_on:
                continue
            faces = [q for q in faces_of[c] if q not in used_faces and face_ok(q, V, E, P, *geom[q])]
            rng.shuffle(faces)
            for q in faces:
                fc = geom[q][0]
                out = fc - center[c]
                out /= np.linalg.norm(out)
                near = tree.query_ball_point(fc, cfg["capture"])
                near.sort(key=lambda i: np.linalg.norm(tree.data[i] - fc))
                best = next((int(ids[i]) for i in near
                             if used_nodes[int(ids[i])] < m_node and np.dot(tree.data[i] - fc, out) > 0), None)
                # CHANGED 2026-11-24: was `if int(ids[i]) not in used_nodes and ...`
                if best is not None:
                    links.append((q, best))
                    per_cell[c].append((q, best))
                    used_faces.add(q)
                    used_nodes[best] += 1   # CHANGED 2026-11-24: was used_nodes.add(best)
                    stats["formed"] += 1
                    break

    n_surface = len(cap)
    stats["linked_surface_fraction"] = len({owner[p] for p, _ in links}) / max(1, n_surface)
    stats["links_per_surface_cell"] = len(links) / max(1, n_surface)
    stats["capacity"] = int(sum(cap.values()))
    return links, stats
