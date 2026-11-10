#!/usr/bin/env bash
# Core grid for the 12-hour deadline. MEASURE one run first (python llm_experiment.py ... --steps 200) and scale --steps.
# 3 tasks x 5 policies x r_tgt=4 x 1 seed = 15 runs. Add SEEDS="0 1 2" and R_TGTS="1 4" if time remains.
set -e
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
STEPS=${STEPS:-2000}
SEEDS=${SEEDS:-"0"}
R_TGTS=${R_TGTS:-"4"}
for seed in $SEEDS; do for r in $R_TGTS; do for task in boolq arc_challenge gsm8k; do
  for pol in direct post cont in_min rg; do
    f=results/llm/${task}_${pol}_r${r}_s${seed}.json
    [ -f "$f" ] && continue
    python llm_experiment.py --model "$MODEL" --task $task --policy $pol --r-src 64 --r-tgt $r \
      --steps $STEPS --seed $seed --lr ${LR:-2e-4}
  done
done; done; done
python make_report.py
