#!/bin/bash
# ECM connectivity sweep: grow the 29-cell organoid in gels with bond probability p = 1.0, 0.9, 0.8, 0.7
# (3 gels per p: seeds 7, 8, 9), and record how the fibers around it move and turn.
# Inputs: experiments/ecm_connectivity/inputs/p<p>_seed<s>/ (made with
#   python3 -I scripts/tvm/ECM64.py --hole-radius 2.5 --p <p> --seed <s>; sample.topo = sample_init_29cells)
# Every run: this branch's model (pressure-limited growth, biology linker rules, contact 50 0.2),
# linkers and growth from t = 1, t = 0-80, run seed = gel seed. Output: runs/ecm_p<p>_s<s>/.
# About 30 minutes on 8 cores.
cd "$(dirname "$0")/../.."
mkdir -p runs
for s in 7 8 9; do for p in 1.0 0.9 0.8 0.7; do echo "$p $s"; done; done | \
  xargs -P 8 -n 2 sh -c 'python3 run_growth.py --name ecm_p$0_s$1 --init-dir experiments/ecm_connectivity/inputs/p$0_seed$1 \
      --seed $1 --t-start 0 --t-link 1 --t-end 80 --no-movie --overwrite > runs/ecm_p$0_s$1.log 2>&1; echo "done p=$0 seed=$1"'
