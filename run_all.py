"""Cross-platform core grid (Windows/Linux). Usage:
  python run_all.py --model Qwen/Qwen2.5-1.5B --steps 2000 [--seeds 0 1 2] [--r-tgts 4 1] [--dry-run]
Skips runs whose result JSON already exists, so it is safe to stop and restart."""
import argparse, os, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--r-tgts", type=int, nargs="+", default=[4])
ap.add_argument("--tasks", nargs="+", default=["boolq", "arc_challenge", "gsm8k"])
ap.add_argument("--policies", nargs="+", default=["direct", "post", "cont", "in_min", "rg"])
ap.add_argument("--lr", default="2e-4")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
for seed in a.seeds:
    for r in a.r_tgts:
        for task in a.tasks:
            for pol in a.policies:
                f = os.path.join("results", "llm", f"{task}_{pol}_r{r}_s{seed}.json")
                if os.path.exists(f):
                    continue
                cmd = [sys.executable, "llm_experiment.py", "--model", a.model, "--task", task, "--policy", pol,
                       "--r-src", "64", "--r-tgt", str(r), "--steps", str(a.steps), "--seed", str(seed), "--lr", a.lr]
                print(" ".join(cmd), flush=True)
                if not a.dry_run:
                    subprocess.run(cmd, check=True)
if not a.dry_run:
    subprocess.run([sys.executable, "make_report.py"], check=True)
