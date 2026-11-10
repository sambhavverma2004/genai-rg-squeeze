"""Aggregate results/llm/*.json into a markdown table (task x policy), mean over seeds."""
import glob, json, collections
rows = collections.defaultdict(list)
for f in glob.glob("results/llm/*.json"):
    r = json.load(open(f)); rows[(r["task"], r["r_tgt"], r["policy"])].append(r["acc"])
tasks = sorted({k[0] for k in rows}); pols = ["direct", "post", "cont", "in_std", "in_min", "rg"]
for rt in sorted({k[1] for k in rows}):
    print(f"\n### r_tgt = {rt}\n| task | " + " | ".join(pols) + " |\n|---|" + "---|" * len(pols))
    for t in tasks:
        cells = []
        for p in pols:
            v = rows.get((t, rt, p))
            cells.append(f"{100*sum(v)/len(v):.2f} (n={len(v)})" if v else "-")
        print(f"| {t} | " + " | ".join(cells) + " |")
