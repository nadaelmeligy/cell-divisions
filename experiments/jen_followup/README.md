# Jen's follow-up notes (Nov 23-24, 2026)

Work on Jen's notes: Shabeeb's vs our alignment measure, faster growth, linker and contact checks, and parts
of Tao Zhang's new Rust TVM (commit 1676eb89) ported to our C++. This file records the checks and the passive-linker test.

## Passive linkers (Nov 24)

Jen asked to make the linkers passive: no rest length they actively return to. Until now every linker had
rest length 1.5 but attaches up to 3.0 away (`capture`), so it starts stretched and pulls at once. Passive
now means a linker's rest length is its own length when it attaches. It starts tension-free and only resists
being stretched further.

- Engine: conf line `link_rest attach` (default `link_rest fixed` = as before). Each linker stores its own
  rest length (`Link::restLength_`). The link files get a 3rd column with it, so it survives restarts.
  A linker moved to a new face, for example at a division, counts as a new attachment.
- Driver: `python3 run_growth.py ... --link-rest attach` (needs the rebuilt engine, `tvm-runtime-v2`).
- Checks, all at round-off level:
  - at the start, rest length = length when attaching (119 linkers, error 9e-16);
  - kept linkers carry their rest length unchanged into the next segment (107 of 107);
  - new linkers start at their attach length (15 linkers, error 7e-16).
- The earlier "no linker force" run is a different test: it set the stiffness to 0, which removes
  linker force altogether.

**Result.** Continuation of the p = 1.0, seed 7 run from t = 80 to 94 (T = 0, same settings as the earlier
pilots). All runs end at nearly the same tissue volume (203-204). S is the alignment of the same 239 fiber
segments tracked from t = 80 (in the R to R+2 shell then), measured against their t = 80 radial directions:

| Run | S at t = 80 | S at t = 94 | change | linker energy |
|---|---:|---:|---:|---:|
| fixed linkers (rest 1.5), today's engine | -0.175 | -0.258 | -0.083 | 3.1 |
| **passive linkers** | -0.175 | **-0.203** | **-0.028** | 0.3 |
| no linker force (earlier run, older engine) | -0.175 | -0.207 | -0.032 | 0 |
| fixed linkers (earlier run, older engine) | -0.175 | -0.272 | -0.097 | 2.0 |

- **With passive linkers the fibers rotate about a third as much** as with the fixed rest length: -0.028
  vs -0.083. That is almost the same as no linker force at all.
- So in these 14 time units, about two thirds of the fibers' turn toward the surface comes from the
  linkers' built-in pull. The rest comes from the tissue pushing the fibers (contact).
- The two fixed-linker runs differ by 0.014. Their only differences are full-precision restarts and swap
  validation, so 0.014 is a rough noise floor. The passive effect (0.055) is about 4 times that.
- **Caveats:** one seed, one checkpoint, 14 time units, no noise.

Reproduce:

```
python3 run_growth.py --name jen_passive_link_p1_s7 --start-from runs/ecm_p1.0_s7 --t-end 94 --temperature 0 --link-rest attach
python3 run_growth.py --name jen_baseline_v2_p1_s7 --start-from runs/ecm_p1.0_s7 --t-end 94 --temperature 0
python3 experiments/jen_followup/compare_pilots.py 94 baseline_v2 passive_link baseline no_link_force
```

## Full runs from t = 0: contact makes the fibers wrap (Nov 24)

The 14-unit test above started from a state already shaped by 80 time units of pulling linkers. To settle
the mechanism, four runs from t = 0 differ in one force only. All use the current engine with Tao's exact
face-centre forces, growth twice as fast (τ = 10), T = 1e-4 and one shared seed; the gel is p = 1.0, seed 7.
The segments in the R to R+2 shell at t = 40 (when the tissue reaches the gel) are followed to t = 80
(`alignment_mechanism.py --t-ref 40`).

| Run | S of the same segments, t = 40 → 80 | fiber points inside cells at t = 80 |
|---|---|---:|
| all forces | −0.137 → −0.470 | 0 |
| passive linkers | −0.112 → −0.428 | 0 |
| no linker force | −0.109 → −0.491 | 0 |
| no contact | −0.036 → −0.071 | 1,555 |

- **Contact causes the wrapping.** The growing tissue pushes the fibers flat; without contact they barely
  turn and pass through the cells.
- **Linkers change it by about 10%.** This supersedes the short test's "two thirds from linker pull".
- All four tissues end at about the same size (radius 5.75 to 5.97).
- **Caveat:** one seed per condition.

## Tests in this folder

| Test | What it checks |
|---|---|
| `check_tao_port.py` (run by `ctest` in `build/`) | Tao's 64-cell reference (forces, volumes, energies, noise); exact segment endpoints; swap interval independent of dt; restart with noise reproduces a straight run; skipping VTK changes nothing; a malformed mesh is refused; rejected swaps are rolled back |
| `check_driver_restart.py` | the same restart check through `run_growth.py`, with noise and divisions: a run stopped at t = 3 and continued ends byte-identical to a straight run; a new engine seed makes it differ |
| `check_swap_rollback.py` | after many forced swap rejections the mesh is still valid, checked with separate Python code (closed faces and cells, Euler relation, cell volumes tile the box exactly) |
| `face_center_gradient_check.cpp`, `contact_link_check.cpp` | A finite-difference check of the ported face-centre forces, and small contact and linker force cases |
