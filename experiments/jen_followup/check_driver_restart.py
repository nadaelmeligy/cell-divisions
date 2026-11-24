"""Restart check for the Python growth driver (run_growth.py), with noise and divisions.

A runs t = 0 -> 6 in one go. B runs t = 0 -> 3, then C continues B to t = 6 (--start-from, no new seeds, so
the engine's random stream, the Python random state, V0 and the linkers are all carried over). With noise on
(T = 1e-4) and fast growth (doubling time 4, so cells divide around the split), C must end byte-identical to A.
Control: D continues B with a new engine seed and must differ, which shows the comparison can fail.
    python3 experiments/jen_followup/check_driver_restart.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ["--temperature", "1e-4", "--doubling-time", "4", "--no-movie", "--overwrite"]


def grow(name, *args):
    cmd = [sys.executable, str(ROOT / "run_growth.py"), "--name", name, *args, *COMMON]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"{name} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return ROOT / "runs" / name


def final_state(run):
    st = json.loads((run / "state.json").read_text())
    files = {}
    for key in ("topo", "ecm", "links", "rng"):
        if st.get(key):
            p = Path(st[key])
            files[key] = (p if p.is_absolute() else run / p).read_bytes()
    return st, files


def same(a, b):
    (sa, fa), (sb, fb) = a, b
    diffs = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    for k in ("t", "V0", "python_rng"):
        if sa.get(k) != sb.get(k):
            diffs.append(k)
    return diffs


def main():
    A = grow("restart_check_A", "--t-end", "6", "--seed", "7", "--engine-seed", "11")
    B = grow("restart_check_B", "--t-end", "3", "--seed", "7", "--engine-seed", "11")
    C = grow("restart_check_C", "--start-from", str(B), "--t-end", "6")
    D = grow("restart_check_D", "--start-from", str(B), "--t-end", "6", "--engine-seed", "12")
    sa, sc, sd = final_state(A), final_state(C), final_state(D)
    n_cells = [len(s[0]["V0"]) for s in (final_state(B), sa)]
    print(f"cells at t = 3: {n_cells[0]}, at t = 6: {n_cells[1]} (divisions after the split: {n_cells[1] - n_cells[0]})")
    d_ac, d_ad = same(sa, sc), same(sa, sd)
    print("A vs C (straight vs stop-and-continue):", "identical" if not d_ac else f"DIFFER in {d_ac}")
    print("A vs D (control, new engine seed):    ", "identical (BAD: test cannot fail)" if not d_ad else f"differ in {d_ad}")
    ok = not d_ac and d_ad and n_cells[1] > n_cells[0]
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
