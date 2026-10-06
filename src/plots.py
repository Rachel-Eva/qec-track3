import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = json.load(open("shared/results_shots3x.json"))
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
for case in dict.fromkeys(r["case"] for r in rows):
    rr = [r for r in rows if r["case"] == case]
    ax.plot([r["shots"] for r in rr], [r["caught_rate"] for r in rr], "o-", label=case)
ax.set_xscale("log"); ax.set_xlabel("shots"); ax.set_ylabel("fraction of planted defects caught")
ax.set_title("Shots needed (3x defects, d=5, 8 rounds, 4 repeats per point)")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("figures/shots_needed.png")
