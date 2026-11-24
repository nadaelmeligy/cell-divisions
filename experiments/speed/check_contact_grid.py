"""Check that the grid search in Energy/CellFiberContact.cpp (Nov 25, 2026) gives the same engine
results as the original search over every surface triangle.

From the last saved state of a run (default runs/tao_all_on_p1_s7, t = 80, ~930 cells, contact on),
one engine segment of one time unit is run three times in fresh folders:
  base   an engine built from the sources before the change (give its path),
  grid   the new engine (default search),
  brute  the new engine with the conf line 'contact_search brute' (the original loops, kept).
Every file the engine writes must be byte-identical between the three, except the 'Rte' column of
run.log (the wall-clock seconds of the run), which is compared after blanking it.

    python3 experiments/speed/check_contact_grid.py BASE_TVM NEW_TVM [RUN_DIR]
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_growth as rg  # noqa: E402


def make_segment(run_dir):
    """A segment folder from the run's last state, written as run_growth would write it."""
    st = json.loads((run_dir / "state.json").read_text())
    cfg = dict(rg.CONFIG)
    cfg.update(json.loads((run_dir / "config.json").read_text()))
    for k, v in list(cfg.items()):
        if isinstance(rg.CONFIG.get(k), tuple) and isinstance(v, list):
            cfg[k] = tuple(v)
    seg = Path(tempfile.mkdtemp(prefix="contactcheck_"))
    shutil.copy2(rg.resolve_saved(st["topo"], run_dir), seg / "sample.topo")
    shutil.copy2(rg.resolve_saved(st["ecm"], run_dir), seg / "ECM.topo")
    if st.get("links"):
        shutil.copy2(rg.resolve_saved(st["links"], run_dir), seg / "links.input")
    if st.get("rng"):
        shutil.copy2(rg.resolve_saved(st["rng"], run_dir), seg / "rng.input")
    t0 = st["t"]
    rg.write_conf(seg, cfg, t0, t0 + 1, runtime_options=True)
    rg.write_cell_parameters(seg, cfg, {int(k): v for k, v in st["V0"].items()})
    return seg, t0


def run(engine, seg, extra_conf=None):
    d = Path(tempfile.mkdtemp(prefix="contactrun_"))
    for f in seg.iterdir():
        shutil.copy2(f, d / f.name)
    if extra_conf:
        with open(d / "conf", "a") as fh:
            fh.write(extra_conf + "\n")
    tic = time.perf_counter()
    with open(d / "stdout.txt", "w") as out:
        subprocess.run([str(engine)], cwd=d, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=7200)
    return d, time.perf_counter() - tic


def contents(d):
    out = {}
    for f in sorted(d.iterdir()):
        if f.name in ("stdout.txt", "conf"):
            continue
        b = f.read_bytes()
        if f.name == "run.log":   # blank the wall-clock column (second column of the energy table)
            # with the spaces after it: the column is fixed-width, so 6.1 and 12.8 pad differently
            b = re.sub(rb"(?m)^(\d+\.\d+\s+)\d+\.\d+\s+", rb"\1- ", b)
        out[f.name] = b
    return out


def main():
    base, new = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    run_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "runs" / "tao_all_on_p1_s7"
    seg, t0 = make_segment(run_dir)
    results = {}
    for name, engine, extra in (("base", base, None), ("grid", new, None), ("brute", new, "contact_search brute")):
        d, secs = run(engine, seg, extra)
        results[name] = (contents(d), secs)
        shutil.rmtree(d)
    shutil.rmtree(seg)
    ok = True
    ref = results["base"][0]
    for name in ("grid", "brute"):
        got = results[name][0]
        diff = sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))
        ok &= not diff
        print(f"{name:5s} vs base: {len(got)} files, {'IDENTICAL' if not diff else 'DIFFERENT: ' + ', '.join(diff)}")
    print("engine time for one unit from t = %g: base %.0f s, grid %.0f s, brute %.0f s"
          % (t0, results["base"][1], results["grid"][1], results["brute"][1]))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
