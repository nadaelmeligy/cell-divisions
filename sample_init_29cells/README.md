# 29-cell starting configuration (made 2026-11-10)

- `sample.topo` — the 29-cell spheroid (radius about 2.2), copied unchanged from `video-run-r2/sample.topo`.
- `ECM.topo` — a NEW fiber network generated for this spheroid:

      cd sample_init_29cells && python3 ../scripts/tvm/ECM64.py --hole-radius 2.5 --seed 7

  The old `video-run-r2/ECM.topo` has its hole cut at radius 5.5 (sized for the 243-cell spheroid),
  so its linkable nodes are at least 3.96 from every cell face, more than the 3.0 capture distance
  in `Run::initializeLinks`. No linkers can form there. In this ECM, all 36 linkable nodes are
  within reach, and the closest fiber node sits at r = 3.32 (outside the cells).
- `ECM_generation.log` — the generator's output.
