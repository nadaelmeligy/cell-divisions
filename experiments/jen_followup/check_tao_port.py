"""Run the Tao TVM numerical reference and C++ restart regression checks."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


FIXTURE = Path(__file__).parent / "fixtures" / "tao_periodic64"


def setup(root, name, start, end, *, dt=0.001, dump="topo", threshold=0.02):
    folder = root / name
    folder.mkdir()
    shutil.copyfile(FIXTURE / "sample.topo", folder / "sample.topo")
    cells_path = folder / "sample.topo"
    header, cells = cells_path.read_text().split("\ncells ")
    count, *rows = cells.splitlines()
    # Rust's periodic fixture stores only the face IDs. C++ also needs the real-cell type.
    cells_path.write_text(header + "\ncells " + count + "\n" +
                          "".join(f"{row} 1\n" for row in rows if row.strip()))
    (folder / "ECM.topo").write_text("nodes 0\nfibers 0\n")
    (folder / "conf").write_text(
        f"time {start:.17g} {end:.17g} {dt:.17g}\n"
        f"dtr 0.002\nreconnect_origin 0\ndump {dump} 0.004\nlog 0.004\n"
        f"s0 5.6 0\nLth {threshold}\nT 0.0001\nkv 10\nbox 4 4 4 p p p\n"
        "fiber 0 1 0\nlink 0 0 1 1 0 1000\nlinkrules python\n"
        "seed 20261028\nface_center_forces exact\nvalidate_topology on\n"
    )
    return folder


def run(binary, folder, *args, succeeds=True):
    result = subprocess.run([str(binary), *map(str, args)], cwd=folder, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert (result.returncode == 0) == succeeds, result.stdout
    return result.stdout


def output_at(folder, suffix, time):
    matches = [path for path in folder.glob(f"*{suffix}") if path.name != suffix[1:]
               and float(path.name[:-len(suffix)]) == time]
    assert len(matches) == 1, (folder, time, matches)
    return matches[0]


def summary(folder):
    return dict(line.split() for line in (folder / "run_summary.txt").read_text().splitlines())


def main(tvm, reference):
    with tempfile.TemporaryDirectory(prefix="tvm-tao-port-") as tmp:
        root = Path(tmp)
        initial = setup(root, "reference", 0., 0.)
        print(run(reference, initial, FIXTURE.resolve() / "reference.tsv").splitlines()[-1])

        end = 0.0065
        continuous = setup(root, "continuous", 0., end)
        first = setup(root, "first", 0., 0.003)
        resumed = setup(root, "resumed", 0.003, end)
        run(tvm, continuous)
        run(tvm, first)
        shutil.copyfile(output_at(first, ".sample.topo", 0.003), resumed / "sample.topo")
        shutil.copyfile(first / "rng.output", resumed / "rng.input")
        run(tvm, resumed)
        for suffix in (".sample.topo", ".ECM.topo", ".linkInfo.txt"):
            assert output_at(continuous, suffix, end).read_bytes() == output_at(resumed, suffix, end).read_bytes()
        assert (continuous / "rng.output").read_bytes() == (resumed / "rng.output").read_bytes()
        expected_sweeps = 3
        assert int(summary(continuous)["reconnection_sweeps"]) == expected_sweeps
        assert int(summary(continuous)["accepted_reconnection_proposals"]) > 0
        assert sum(int(summary(p)["reconnection_sweeps"]) for p in (first, resumed)) == expected_sweeps
        assert float(summary(continuous)["final_time"]) == end

        vtk = setup(root, "vtk", 0., end, dump="vtk")
        run(tvm, vtk)
        assert output_at(vtk, ".sample.topo", end).read_bytes() == output_at(continuous, ".sample.topo", end).read_bytes()
        assert list(vtk.glob("*.vtk")) and not list(continuous.glob("*.vtk"))
        fine = setup(root, "fine", 0., end, dt=0.0005)
        run(tvm, fine)
        assert int(summary(fine)["reconnection_sweeps"]) == expected_sweeps

        malformed = setup(root, "malformed", 0., end)
        path = malformed / "sample.topo"
        lines = path.read_text().splitlines()
        first_face = next(j for j, line in enumerate(lines) if line.startswith("polygons ")) + 1
        lines[first_face] += " " + lines[first_face].split()[1]
        path.write_text("\n".join(lines) + "\n")
        log = run(tvm, malformed, succeeds=False)
        assert "Face repeats an edge" in log and not (malformed / "run_summary.txt").exists(), log

        # Oversized new edges propose inverted local cells. The run should reject
        # those candidates, roll back the edits, and still save a valid final mesh.
        aggressive = setup(root, "aggressive", 0., 0.002, threshold=0.8)
        run(tvm, aggressive)
        assert int(summary(aggressive)["rejected_reconnection_proposals"]) > 0
        verified = setup(root, "verified", 0.002, 0.002)
        shutil.copyfile(output_at(aggressive, ".sample.topo", 0.002), verified / "sample.topo")
        run(tvm, verified)
        print("Exact endpoint, fixed swap interval, stochastic restart, VTK parity and mesh rollback passed.")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
