# Changes to the 3DVM-ECM code, November 2026

Branch `organoid-growth`, compared with the last commit before this work (`2a3462e`, Nov 17, on top of
Shabeeb's fork). Every change in the code is also marked in place with a `CHANGED 2026-11` comment that
keeps the old line.

**Ground rule: with the original settings the engine behaves exactly as before.** Everything new is
switched on by a new line in `conf` (or by `run_growth.py`, which writes those lines). Checked: Shabeeb's
`sample_init` with its own conf, run to t = 2 with swaps on (about 1,900 swaps), gives the same mesh as
the original engine. Positions agree to the original's 6-digit output rounding (relative difference
under 5e-6). The test is described in "How it was checked" below.

## 1. Bugs fixed in the original

| What | Why it mattered | Where |
|---|---|---|
| Linkers were lost at every restart | The engine writes its linkers out but never read them back, and it creates them only once (at t = 500). Every run continued from a saved state had no linkers at all. New: `links.input` is read back (`Run::loadLinks`). With no such file, nothing changes. | `Run/Run.cpp`, `tvm.cpp` |
| Linker start time was typed into the code (500) | A run starting after 500 never made linkers. It is now the optional 7th number of the `link` line (default 500). | `Run/Run.cpp`, `Energy/FiberLink.h` |
| The ECM file check looked at the wrong file | `InitializeAll` tested the cell file instead of the ECM file. | `tvm.cpp` |
| Shrinking linkers could go below their floor | Rest length 0.3 minus a 0.2 decrease gave 0.1, below the floor l1 = 0.2. | `Run/Run.cpp` |
| Division code edited lists while looping over them | Python skips the next item, so a second cut edge in a face could be missed. No division in our samples changed. | `toolbox/cellDivision.py` |

## 2. New physics, each off unless switched on

| Option (conf line) | What it does | Why |
|---|---|---|
| `contact k range` | Cell-fiber contact force: fiber points closer than `range` to a cell face are pushed out, with the opposite force on the face (momentum conserved). | Without it the growing organoid passed straight through every fiber it was not linked to. |
| `fiber ks l0 kb 1` (5th number) | Worm-like-chain bending kb (1 + cos θ) instead of the original ½ kb (1 + cos θ)². | The original form grows like (bend angle)⁴, so it has no stiffness against small bends. |
| `linkrules python` | `linkers.py` forms and breaks linkers between segments, by contact area, at division and by turnover. | Biology-based linker rules. The engine's own rule removed 25-45% of linkers per time unit whenever a face was briefly badly shaped. |
| `link_rest attach` | Passive linkers: each linker's rest length is its own length when it attaches, saved across restarts. | Jen's request. Fixed rest 1.5 with attachment up to 3.0 away means every linker starts stretched and pulls. |
| `face_center_forces exact` | Exact derivative of the face centre (perimeter-weighted edge midpoints) when spreading face-centre forces to corners. From Tao Zhang's Rust TVM. | The original split assumes edge lengths stay fixed when a corner moves. |

## 3. Reproducibility and checks (from Tao Zhang's Rust TVM, commit 1676eb89; ported and audited here)

| Option | What it does | Default |
|---|---|---|
| `seed N` | One seeded noise stream per run (`Run/Random.h`), saved in `rng.output` so a restart continues it (`rng.input`). | No `seed` line: a random seed is drawn and printed, so repeated runs still differ, as with the original. |
| `dtr T`, `reconnect_origin T` | Swap checks every `dtr` from a fixed origin, so a run cut into 1-unit pieces checks at the same times as one long run. Runs stop exactly at `t_end`. | `dtr` = 10 dt and the original schedule, as before. `run_growth.py` sets the origin to the run's start. |
| (always) | Restart files `sample.topo` and `ECM.topo` written with 17 digits instead of 6. Also the Python division output. | Needed for exact restarts. The original nudged every position at every 1-unit restart. |
| `validate_topology on` | Checks the mesh at start-up and every proposed swap: closed faces and cells, positive volumes, volume conserved. A failing swap is undone and counted (`run_summary.txt`). | Off: no checks, and the two original error exits in the H→I swap are kept. |
| `dump topo P` | Writes the topology files only, no VTK (about 18% of engine time). | `dump vtk P` as before. |
| `tvm --runtime-features` | Prints which options the binary has, so `run_growth.py` refuses to use an older binary without saying so. | |

## 4. Python pipeline (new files)

- `run_growth.py` runs the engine in 1-unit segments. Between segments it updates growth (each cell's
  target volume grows with pressure feedback) and division (Hertwig's rule, with the new mesh checked
  before it is used), and applies the linker rules. It ends with a movie. Options are listed by
  `python3 run_growth.py -h`.
- `growthviz.py` reads the restart files, measures (fiber alignment, pressure, divisions...) and
  draws the movies.
- `linkers.py` holds the biology linker rules, with references in its header.
- `scripts/tvm/ECM64.py` gains flags `--hole-radius`, `--p`, `--p-left` and `--seed`. The same seed
  gives the same gel, and half gels are exact mirror images. Without the flags it behaves as before.

## 5. How it was checked

| Check | Result |
|---|---|
| Original conf, sample_init, t = 0-2, swaps on, vs the original engine built from `2a3462e` | identical mesh; positions within output rounding |
| Same, swaps off | positions within output rounding |
| Tao's 64-cell reference (`ctest` in `build/`): forces, volumes, energies, noise | matches |
| Engine restart with noise vs one long run (`ctest`) | byte-identical |
| `run_growth.py` stopped at t = 3 and continued vs straight (`experiments/jen_followup/check_driver_restart.py`) | byte-identical, with noise and divisions |
| Rejected swaps leave a valid mesh, separate Python checker (`check_swap_rollback.py`) | valid, cells tile the box exactly |
| Passive linkers: rest length = attach length, kept across restarts | to 1e-15 |
| Contact, face-centre forces: finite-difference gradient checks | match |

## 6. Found during the audit (Nov 24) and fixed

- **Noise seed.** The first port used the same fixed seed for every run without a `seed` line, so
  repeated runs (for example on Condor) would have been identical. Now a random seed is drawn and
  printed.
- **Medium speed cap.** The first port rewrote it to cap the displacement at exactly 0.1. That differs from the
  original (0.1/dte)·dte in the last digit, and the medium's chaotic motion grew that to differences of
  0.02 within one time unit. The original arithmetic is restored. Runs made before this fix differ from
  the original only at that last-digit level, far below the noise they already had (T = 1e-4).
- **H→I swap.** The first port turned two "stop the simulation" checks into "skip this swap" in all modes.
  That now happens only with `validate_topology on`.
- **Unneeded change removed.** Early exits for zero linker stiffness changed no result.

## 7. Open questions for Jen and Shabeeb (not changed)

- **Medium force.** The `EMPTY CELL CHECK` in `Volume.cpp` and `Interface.cpp` skips some forces on
  boundary vertices. Forces then don't balance, which gives the organoid small spurious drifts.
  Intended?
- **Contact on irregular faces.** The opposite force on a cell face is spread over its corners in a
  way that can add a small spurious torque.
- **Speed.** Done on Nov 25, without changing any result: section 8.

## 8. Faster runs (Nov 25), same results bit for bit

A full growth run (t = 0-80, ~950 cells) now takes 22 min instead of 103 min, with identical output.
Measured on one late segment (t = 80 to 81, ~930 cells, ~45 divisions): 86% of the time was Python
division, 12% the engine, and most of the engine was the contact search.

| Change | Where | Why it is the same | Old path kept as |
|---|---|---|---|
| Division in memory: the tissue is read once per segment, the cells are cut one after another in memory, and the check uses the topology just written instead of reading it back. | `run_growth.py`: `divide_cells_memory`, `topo_dicts`, `check_division(..., dicts=)` | The division reads only the topology, positions, the mother's centre, the mother's face vertex lists (recomputed before each cut, as a load would) and one flag, `Vertex.is_daughter_`, which a load sets False and is reset before each cut. A failed attempt reloads the last good file, as before. | `--division reload` (CONFIG `division`) |
| Grid search for contact: each fiber point looks only at the surface triangles that can pass the existing distance test (3D grid of face centres), and the inside/outside rays only at triangles whose shadow covers the point (2D grid per ray). | `Energy/CellFiberContact.cpp` (`Grid3`, `Grid2`, `countCrossings`), `tvm.cpp` | The same triangles are tested with the same arithmetic. On an exact tie the lowest triangle index wins, as in the original loop; a crossing count does not depend on order. | conf line `contact_search brute` |

Checks (all byte-identical except the wall-clock column of `run.log`):

| Check | Result |
|---|---|
| `experiments/speed/check_division_memory.py`: saved tissues at t = 20, 40, 60, 80; 15-16 mothers including neighbours; plus forced failed attempts (retries, reload, skip) at t = 40 and 80 | identical files, daughters, V0 and random state; 16 cuts at t = 80: 93 s -> 3.3 s |
| Short full run t = 0-10 with fast growth (doubling time 3), `--division reload` vs `memory` | all 116 files identical |
| One engine unit from the saved t = 80 state: engine built before the change vs the new one, and the new one with `contact_search brute` (`experiments/speed/check_contact_grid.py`) | identical; engine 69 s -> 16 s |
| One driver segment t = 80 to 81 (966 cells), old vs new engine | all 81 files identical; 89 s -> 34 s per segment (both with the new division) |
| `ctest` (Tao's 64-cell reference) in a build of the new code | passes |
| `check_driver_restart.py` (stop at t = 3 and continue, with divisions in memory) and `check_swap_rollback.py`, on the rebuilt `build/tvm` | both pass |
| **Full run t = 0-80** (p = 1 gel, seed 7, the settings of `runs/tao_all_on_p1_s7`, 946 cells at the end): old engine + `--division reload` vs new engine + `--division memory`, run side by side | all 888 files identical (only `config.json` differs: name, engine, division); **103 min -> 22 min** |

`build/` was rebuilt with the current code on Nov 25 (`./compile.sh`). The engine that made the four
runs `runs/tao_*_p1_s7` (built before the Nov 24 audit fixes) is kept in `build-preaudit/`, so those
runs can still be reproduced bit for bit.

## 9. Several linkers per gel node, for merged-node gels (Nov 24)

| Change | What it does | Why |
|---|---|---|
| `--link-node-capacity M` (run_growth.py, default 1) | Under the Python link rules a gel node may hold up to M linkers (linkers.py counts node use instead of keeping a set). | The Matrigel-like mesh merges the 3 crosslink nodes of each lattice site into one; with M = 1 it had a third of the binding slots, so linkers were limited by free nodes (108 vs 404). M = 3 keeps the slots per site. |
| `Run::loadLinks` accepts a repeated node under `linkrules python` | The engine no longer drops the 2nd and 3rd linker on one node; Python enforces M. Engine linker rules are unchanged. | Bookkeeping only: the forces of several linkers on one node already add (FiberLink.cpp). |
| `tvm --runtime-features` prints `tvm-runtime-v3` | run_growth.py refuses M > 1 with an older engine. | An older engine would silently drop the extra linkers. |

With M = 1 every run is unchanged (the same linker files are written). Checked: a 5-step mesh run seeds 50 linkers
at t = 2 with M = 3 (25 with M = 1; the fiber gel seeds 50). Details: experiments/matrigel/README.md, "Reruns".

## Where things are

- Experiments, each with a README:
  - `experiments/ecm_connectivity`: gel connectivity, durotaxis;
  - `experiments/fiber_angle`: which fibers push back;
  - `experiments/jen_followup`: Jen's notes, Tao port tests, passive linkers;
  - `experiments/speed`: the checks that the faster division and contact search change nothing.
- Not committed: `runs/` (simulation output), `build*/`, the large movies, and the gel input files
  (about 80 MB, kept locally).
