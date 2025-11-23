# plot_results.py  (updated)
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IN = Path("results/log.json")
OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

with IN.open("r") as f:
    data = json.load(f)

# If entries have 'step', use it; otherwise build index list
if data and "step" in data[0]:
    x = [int(entry["step"]) for entry in data]
else:
    x = list(range(len(data)))

P = [float(entry.get("Ptotal", 0.0)) for entry in data]
L = [float(entry.get("Ltotal", 0.0)) for entry in data]

# Simple moving average smoother (window size w). Set w=1 to disable smoothing.
def moving_avg(arr, w=1):
    if w <= 1:
        return np.array(arr)
    a = np.array(arr)
    return np.convolve(a, np.ones(w)/w, mode="same")

# Choose smoothing window (small number because dataset short)
smooth_w = 1
P_s = moving_avg(P, smooth_w)
L_s = moving_avg(L, smooth_w)

# ---- Plot 1: Energy ----
plt.figure(figsize=(6.2,3.6))
plt.plot(x, P_s, marker='o', linewidth=1.6, markersize=6)
plt.title("Total Energy Consumption Over Steps", fontsize=12)
plt.xlabel("Simulation Step", fontsize=10)
plt.ylabel("Ptotal (Watts)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
# nice y-limits with margin
ymin, ymax = min(P_s), max(P_s)
yrange = max(1.0, (ymax - ymin))
plt.ylim(ymin - 0.08*yrange, ymax + 0.08*yrange)
plt.tight_layout()
plt.savefig(OUT / "energy_plot.png", dpi=300)
plt.close()

# ---- Plot 2: Load variance ----
plt.figure(figsize=(6.2,3.6))
plt.plot(x, L_s, marker='s', linewidth=1.6, markersize=6)
plt.title("Load Balance Variance Over Steps", fontsize=12)
plt.xlabel("Simulation Step", fontsize=10)
plt.ylabel("Ltotal (Variance)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
ymin, ymax = min(L_s), max(L_s)
yrange = max(1e-6, (ymax - ymin))
plt.ylim(ymin - 0.08*yrange, ymax + 0.08*yrange)
plt.tight_layout()
plt.savefig(OUT / "load_plot.png", dpi=300)
plt.close()

# ---- Plot 3: Combined (normalized second axis) ----
fig, ax1 = plt.subplots(figsize=(6.2,3.6))
ax1.plot(x, P_s, label="Ptotal (W)", linewidth=1.6)
ax1.set_xlabel("Simulation Step", fontsize=10)
ax1.set_ylabel("Ptotal (Watts)", fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)
ax2 = ax1.twinx()
# scale L to a readable range on the right axis
ax2.plot(x, L_s, color='tab:orange', label="Ltotal (var)", linewidth=1.6)
ax2.set_ylabel("Ltotal (Variance)", fontsize=10)
# legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=9)
plt.title("Energy and Load Variation", fontsize=12)
fig.tight_layout()
plt.savefig(OUT / "combined_plot.png", dpi=300)
plt.close()

# ---- Summary stats (print) ----
def stats(name, arr):
    a = np.array(arr)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}

print("Plots written to:", OUT)
print("Ptotal stats:", stats("Ptotal", P))
print("Ltotal stats:", stats("Ltotal", L))
