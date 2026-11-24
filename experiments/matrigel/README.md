# A Matrigel-like gel in the growing-ball test (24 Nov 2026)

**Question (Nada):** the experiments use Matrigel, not collagen. Does the flattening of the gel next to a
growing organoid (Result 1) still happen in a Matrigel-like gel?

**Why our gel is collagen-like:** long straight fibers of 2.83-long segments on an FCC lattice, held mainly by
stretching along each fiber, with almost no bending stiffness. Matrigel is a basement-membrane gel (mainly laminin,
collagen IV, entactin and a heparan sulfate proteoglycan; Emonard 1987) whose structure is "interconnected thin
sheets" (Kleinman 1986), and it is soft: a few hundred Pa (AFM at 37 °C, Soofi 2009).

**Proxy:** an FCC lattice where every lattice bond is its own short strand (about 11 per node near the hole), no long
fibers and no bending: a dense spring mesh that behaves like an elastic solid. Spacing a = 1.4 and a = 1.0
(a cell is 1.2 across). Same hole (no node closer than 3.2 to the centre), same test as
`experiments/contact_sweep` (rigid ball grown quasi-statically from R = 3.0 to 6.6, gel clamped beyond 20,
contact k = 50, δ = 0.2). Script: `ball_test.py A`; mesh files in `inputs/`, results in `results/a{A}.csv`.

| gel | S next to the ball at R = 3.0 | 4.8 | 6.6 | push carried by flat strands |
|---|---:|---:|---:|---:|
| collagen-like fibers (k50_d0.2 of the sweep) | −0.07 | −0.24 | −0.30 | 100% |
| Matrigel-like mesh, a = 1.4 | −0.07 | −0.16 | −0.23 | 100% |
| Matrigel-like mesh, a = 1.0 | −0.03 | −0.16 | −0.23 | 100% |

**Answer:** yes. The mesh next to the ball also turns flat, about three quarters as strongly as the long fibers
(S = −0.23 vs −0.30 at the largest ball), the same for both mesh sizes, and all of the push is carried by flat
strands. Prediction written before the run: an elastic solid around a growing ball is squeezed radially and
stretched around the ball, so strands turn flat even without long fibers, but less than fibers that can also
buckle and swing aside. Both held.

Not covered here: Matrigel's viscoelastic and plastic behaviour (it relaxes stress), its true pore size (much
finer than this mesh), and the absolute stiffness relative to the cells (this quasi-static test only compares
geometry and the push pattern). Full organoid runs: next section.

## Full organoid runs (run_growth.py, t = 0–80)

**Gel:** `make_full_gel.py` builds a mesh on the SAME lattice sites as the collagen-like gel
`experiments/ecm_connectivity/inputs/p1.0_seed7` (same hole, box 64, spacing 2.83); every nearest-neighbour bond
is its own two-node strand and strands at a site share one node (the three crosslink nodes at each site merged);
no bending. Files: `inputs/mesh_p1.0_seed7/`.
In the growing-ball test on a separately built lattice (`results/a2.82843.csv`) the mesh pushed back 2.2× harder at
ks = 10, so the first softened run used ks = 4.5. Correction (24 Nov): on the gel these runs actually use
(`ball_cohort.py`, `results/ball_cohort.csv`) the mesh pushes back about 5× harder near R = 6 (3.09 vs 0.61), and
4.5 still 2.4× harder; ks = 2 matches (0.67 vs 0.61 at R = 6.0). See "Reruns" below.

**Runs:** the settings of Result 1's "all forces" run (`runs/tao_all_on_p1_s7`), only the gel changed:

    python3 run_growth.py --name mg_mesh_ks{KS}_p1_s7 --init-dir experiments/matrigel/inputs/mesh_p1.0_seed7 \
        --seed 7 --engine-seed 7 --t-start 0 --t-link 1 --t-end 80 --doubling-time 10 \
        --face-center-forces exact --validate-topology --contact 50 0.2 --link-rules biology \
        --fiber-ks {KS} --fiber-kb 0 --no-movie --overwrite          # KS = 10 and 4.5; about 35 min each

**Measured** (`full_runs.py` -> `results/full_runs.csv`): S of the same strands over time (midpoints between 3.15
and 5.15 from the starting centre at t = 0, as Result 1: 145 fibers, 150 mesh strands), cells and mean pressure.

| gel | S at t = 40 | S at t = 80 | cells at t = 80 | mean P at t = 80 | linkers at t = 80 |
|---|---:|---:|---:|---:|---:|
| collagen-like fibers, ks = 10 | −0.16 | −0.47 | 929 | 2.27 | 404 |
| Matrigel-like mesh, ks = 10 | −0.01 | −0.26 | 501 | 3.46 | 108 |
| Matrigel-like mesh, ks = 4.5 | −0.04 | −0.42 | 645 | 2.70 | 123 |

**Answer:** in the full model the mesh next to the organoid also turns flat, later than the fibers and less at
the same strand stiffness; the softened mesh, which pushes back like the fiber gel, ends close to the fibers
(−0.42 vs −0.47). The organoid grows more slowly in both meshes and its cells are more squeezed (P 2.7 and 3.5
vs 2.3). Caveat: the mesh has one node per lattice site where the fiber gel has three (six fibers pass through every
site, joined in pairs: each node joins two fibers, four strands; checked from ECM.topo), so far fewer linkers form
(108 and 123 vs 404); part of the difference may come from that, not from the gel's structure. One seed per gel.

**Structure, checked from the files:** both gels have about 12 strands per site (collagen-like 95,208 segments,
mesh 98,262 strands, 16,380 sites). The collagen-like gel joins them at 3 nodes per site (4 strands per node:
coordination 4, below the 6 needed for a rigid central-force network, so it relies on bending, kb = 0.001); the
mesh joins all 12 at one node (coordination 12, rigid without bending).


## Reruns with the linker slots matched and ks = 2 (24 Nov 2026)

**Why:** the first mesh runs had two problems. (1) A gel node could hold one linker, and the mesh merges the
3 crosslink nodes of each site into one, so it had a third of the binding slots: linkers were limited by free
nodes, not by the cells (108 and 123 vs 404). (2) ks = 4.5 did not match the fibers' push on this mesh.
**Fix:** `run_growth.py --link-node-capacity 3` (a merged node holds up to 3 linkers; default 1 leaves every
collagen run unchanged; linkers.py, plus one condition in Run::loadLinks, engine feature tvm-runtime-v3), and
ks = 2. A 5-step check gives 50 linkers at t = 2 on the mesh, as on the fibers (it was 25). All runs below use one
engine build; the linker-off pair is the control with no code change (`--link-ks 0`).

| run (t = 80) | S, same strands | cells | linkers | mean P |
|---|---:|---:|---:|---:|
| `v3_collagen_all_on` fibers | −0.48 | 903 | 382 | 2.26 |
| `v3_mesh_ks2_m3` mesh, ks 2, 3 slots per node | −0.48 | 696 | 313 | 2.43 |
| `v3_mesh_ks10_m3` mesh, ks 10, 3 slots per node | −0.27 | 428 | 226 | 3.78 |
| `v3_collagen_nolink` fibers, linker force off | −0.49 | 993 | – | 2.08 |
| `v3_mesh_ks10_nolink` mesh ks 10, linker force off | −0.28 | 669 | – | 3.52 |

**Answer:** with the push and the linker slots matched, the Matrigel-like mesh flattens exactly as much as the
fibers (−0.48). The stiff mesh (ks 10) flattens less with or without linker force, so that difference is the gel's
stiffness, not the linkers. Linker force speeds the early flattening (fibers, S at t = 40: −0.15 with, −0.06
without) but not the end state. One seed per run. `full_runs.py` (reruns) or `full_runs.py old` (first runs).
