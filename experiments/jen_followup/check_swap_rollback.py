"""Independent check that rejected neighbor swaps leave a valid mesh.

Uses the aggressive case from check_tao_port.py (Tao's periodic 64-cell tissue, swap threshold Lth = 0.8,
so many proposed swaps would invert cells) and checks the final mesh with a separate Python checker (not the
engine's MeshValidation): every face is one closed loop owned by exactly two cells, every cell is a closed
shell (each edge in exactly two of its faces, V - E + F = 2), and the cells tile the periodic box: their volumes
(signed, consistently oriented) add up to the box volume 4^3 = 64 and none is negative (inside out). Control: the same run with validate_topology off.
    python3 experiments/jen_followup/check_swap_rollback.py
"""
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import growthviz as gv  # noqa: E402
from check_tao_port import output_at, setup, summary  # noqa: E402
import subprocess  # noqa: E402
import numpy as np  # noqa: E402


def mesh_problems(topo):
    V, E, P, C = gv.read_topo(topo)
    owners = gv.polygon_cells(C)
    bad = [f"face {f} has {len(o)} owners" for f, o in owners.items() if len(o) != 2]
    for cid, (faces, _typ) in C.items():
        edges = Counter(e for f in faces for e in P[f])
        verts = {v for e in edges for v in E[e]}
        if len(faces) < 4 or len(set(faces)) != len(faces):
            bad.append(f"cell {cid}: missing or repeated faces")
        if any(n != 2 for n in edges.values()):
            bad.append(f"cell {cid}: open shell")
        if len(verts) - len(edges) + len(faces) != 2:
            bad.append(f"cell {cid}: Euler V-E+F = {len(verts) - len(edges) + len(faces)}")
        for f in faces:
            try:
                gv.polygon_loop(P[f], E)
            except ValueError as exc:
                bad.append(f"face {f}: {exc}")
    # volumes: unwrap each cell along its own edges (each edge by minimum image; unwrapping around one
    # reference vertex fails for cells stretched over half the box), orient all cells consistently through
    # their shared faces, then the signed volumes must add up to the box volume and none may be negative.
    L = 4.0
    X = {v: np.asarray(x, float) for v, x in V.items()}
    vol, loops = {}, {}
    for cid, (faces, _typ) in C.items():
        try:
            loops[cid] = gv.oriented_face_loops(faces, P, E)
        except ValueError as exc:
            bad.append(f"cell {cid}: {exc}")
            continue
        adj = {}
        for f in faces:
            for e in P[f]:
                a, b = E[e]
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        r0 = next(iter(adj))
        U, todo = {r0: X[r0]}, [r0]
        while todo:
            a = todo.pop()
            for b in adj[a]:
                if b not in U:
                    d = X[b] - X[a]
                    U[b] = U[a] + d - L * np.round(d / L)
                    todo.append(b)
        v = 0.0
        for loop in loops[cid].values():
            pts = np.array([U[q] for q in loop])
            c = pts.mean(axis=0)
            v += sum(np.dot(c, np.cross(pts[k], pts[(k + 1) % len(pts)])) for k in range(len(pts))) / 6.0
        vol[cid] = v
    nb = {}
    for f, o in owners.items():
        if len(o) != 2 or o[0][0] not in loops or o[1][0] not in loops:
            continue
        (a, _), (b, _) = o
        la, lb = loops[a][f], loops[b][f]
        i = lb.index(la[0])
        same = lb[i:] + lb[:i] == la                 # walked the same way round in both cells
        nb.setdefault(a, []).append((b, same))
        nb.setdefault(b, []).append((a, same))
    first = min(vol)
    sign, todo = {first: 1.0 if vol[first] > 0 else -1.0}, [first]
    while todo:
        a = todo.pop()
        for b, same in nb.get(a, []):
            want = -sign[a] if same else sign[a]        # a shared face must be walked oppositely
            if b not in sign:
                sign[b] = want
                todo.append(b)
            elif sign[b] != want:
                bad.append(f"cells {a} and {b}: inconsistent face orientation")
    signed = {c: sign.get(c, 1.0) * v for c, v in vol.items()}
    bad += [f"cell {c} is inside out (volume {v:.4f})" for c, v in signed.items() if v <= 0]
    total = sum(signed.values())
    if abs(total - L ** 3) > 1e-6 * L ** 3:
        bad.append(f"cell volumes add up to {total:.6f}, not {L ** 3:g}")
    return bad, len(C), total


def main():
    tvm = ROOT / "build" / "tvm"
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, validate in [("aggressive", True), ("control_no_validation", False)]:
            folder = setup(root, name, 0., 0.002, threshold=0.8)
            if not validate:
                conf = folder / "conf"
                conf.write_text(conf.read_text().replace("validate_topology on", "validate_topology off"))
            res = subprocess.run([str(tvm)], cwd=folder, capture_output=True, text=True, timeout=120)
            if res.returncode:
                print(f"{name}: engine stopped with an error: {(res.stdout + res.stderr).strip().splitlines()[-1]}")
                ok &= not validate
                continue
            s = summary(folder)
            bad, n, total = mesh_problems(output_at(folder, ".sample.topo", 0.002))
            print(f"{name}: {n} cells, total volume {total:.9f}, swaps accepted {s['accepted_reconnection_proposals']}, rejected "
                  f"{s['rejected_reconnection_proposals']}; independent mesh check: "
                  + ("valid" if not bad else f"{len(bad)} problems, e.g. {bad[:3]}"))
            if validate:
                ok &= not bad and int(s["rejected_reconnection_proposals"]) > 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
