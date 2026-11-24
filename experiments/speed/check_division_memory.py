"""Check that run_growth.divide_cells_memory (the fast, in-memory division of Nov 25, 2026) makes
exactly the same divisions as the original run_growth.divide_cells.

For saved tissues of the run tao_all_on_p1_s7 (t = 20, 40, 60, 80), both functions divide the same
mothers with the same random seed, each in its own copy of the folder. The check passes only if
every file they write (divided_kk.topo) is byte-identical, and the returned file name, number of
divisions, daughters, new target volumes V0 and the random-number state all agree. The mothers are
a random set of real cells plus neighbours of the first one (so cuts share faces), and at t = 80
cell 8734, whose division failed in the real run (so the retry path is tested as well).

    python3 experiments/speed/check_division_memory.py
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import growthviz as gv  # noqa: E402
import run_growth as rg  # noqa: E402

RUN = ROOT / "runs" / "tao_all_on_p1_s7"


def mothers_for(topo, rng, extra=()):
    V, E, P, C = gv.read_topo(topo)
    real = sorted(c for c, (_, typ) in C.items() if typ > 0)
    first = int(rng.choice(real))
    owners = gv.polygon_cells(C)
    neighbours = sorted({o for f in C[first][0] for o, _ in owners[f] if o != first and C[o][1] > 0})
    rest = [int(c) for c in rng.choice(real, size=10, replace=False) if c != first and c not in neighbours]
    return [first] + neighbours[:4] + rest + [c for c in extra if c in C], real


def run(func, src_topo, conf, mothers, real, seed):
    d = Path(tempfile.mkdtemp(prefix="divcheck_"))
    shutil.copy2(src_topo, d / src_topo.name)
    shutil.copy2(conf, d / "conf")
    V0 = {c: 1.0 + 0.01 * (c % 17) for c in real}
    rng = np.random.default_rng(seed)
    tic = time.perf_counter()
    name, n, divisions = func(d, src_topo.name, list(mothers), V0, rng)
    dt = time.perf_counter() - tic
    files = {p.name: p.read_bytes() for p in sorted(d.glob("divided_*.topo"))}
    out = dict(name=name, n=n, divisions=divisions, V0=V0, rng=rng.bit_generator.state, files=files, seconds=dt)
    shutil.rmtree(d)
    return out


def failing(check, fail_first, fail_all):
    """check_division that fails the first `fail_first` checks of every third mother and all checks of
    the mothers in `fail_all`, so the retry path (tilted axes, reload after a half cut) and the
    skip path run in both functions alike."""
    counts = {}

    def wrapped(topo_path, mother, daughters, vol_mother, messages, dicts=None):
        counts[mother] = counts.get(mother, 0) + 1
        if mother in fail_all or (mother % 3 == 0 and counts[mother] <= fail_first):
            return False, "forced failure (test)", None
        return check(topo_path, mother, daughters, vol_mother, messages, dicts=dicts)
    return wrapped


def main():
    ok_all = True
    original = rg.check_division
    for t in (40, 80):  # forced failures: retries and skips
        seg = RUN / f"seg_{t:07d}"
        topo = seg / f"{t:07d}.sample.topo"
        mothers, real = mothers_for(topo, np.random.default_rng(t + 1))
        outs = []
        for func in (rg.divide_cells, rg.divide_cells_memory):
            rg.check_division = failing(original, 2, {mothers[1]})
            outs.append(run(func, topo, seg / "conf", mothers, real, seed=2000 + t))
        rg.check_division = original
        old, new = outs
        same = {k: old[k] == new[k] for k in ("name", "n", "divisions", "V0", "rng", "files")}
        ok = all(same.values())
        ok_all &= ok
        print(f"t = {t:2d}, forced failures: {len(mothers)} mothers, {old['n']} divided  "
              f"{'IDENTICAL' if ok else 'DIFFERENT ' + str([k for k, v in same.items() if not v])}  "
              f"time {old['seconds']:.1f} s -> {new['seconds']:.1f} s")
    for t, extra in ((20, ()), (40, ()), (60, ()), (80, (8734,))):
        seg = RUN / f"seg_{t:07d}"
        topo = seg / f"{t:07d}.sample.topo"
        mothers, real = mothers_for(topo, np.random.default_rng(t), extra)
        old = run(rg.divide_cells, topo, seg / "conf", mothers, real, seed=1000 + t)
        new = run(rg.divide_cells_memory, topo, seg / "conf", mothers, real, seed=1000 + t)
        same = {k: old[k] == new[k] for k in ("name", "n", "divisions", "V0", "rng", "files")}
        ok = all(same.values())
        ok_all &= ok
        print(f"t = {t:2d}: {len(real)} cells, {len(mothers)} mothers, {old['n']} divided, {len(old['files'])} files  "
              f"{'IDENTICAL' if ok else 'DIFFERENT ' + str([k for k, v in same.items() if not v])}  "
              f"time {old['seconds']:.1f} s -> {new['seconds']:.1f} s")
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
