import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Data ──────────────────────────────────────────────────────────────────────
lambdas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
accuracy  = [0.6431, 0.6417, 0.6420, 0.6467, 0.6456, 0.6385, 0.6506, 0.6462, 0.6467]
baseline  = 0.643   # WavLM baseline (no KD, λ=0)
best_idx  = accuracy.index(max(accuracy))

# ── Style (match original) ────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

fig, ax = plt.subplots(figsize=(7.5, 5))

# Baseline dashed line
ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.4, zorder=1)

# Main line + markers
ax.plot(lambdas, accuracy,
        color="#1a3a5c", linewidth=1.8,
        marker="o", markersize=7, markerfacecolor="#1a3a5c", markeredgewidth=0,
        zorder=3)

# Highlight best point
ax.scatter([lambdas[best_idx]], [accuracy[best_idx]],
           color="#c0392b", s=120, zorder=4)

# ── Labels on every point ─────────────────────────────────────────────────────
# place_above=True → label deasupra punctului, False → dedesubt
GAP = 0.00055
overrides = {
    0.1: (True,   0.00),
    0.2: (False,  0.00),
    0.3: (False,  0.00),
    0.4: (True,   0.00),
    0.5: (True,   0.00),
    0.6: (True,  -0.05),
    0.7: (True,   0.00),
    0.8: (False,  0.00),
    0.9: (True,   0.00),
}

for x, y in zip(lambdas, accuracy):
    is_best       = (x == lambdas[best_idx])
    color         = "#c0392b" if is_best else "#1a3a5c"
    above, x_off  = overrides[round(x, 1)]
    va            = "bottom" if above else "top"
    sign          = +1       if above else -1

    label = f"best: {y:.4f}" if is_best else f"{y:.4f}"
    ax.text(x + x_off, y + sign * GAP, label,
            ha="center", va=va, fontsize=10.5, color=color)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlabel("Distillation weight λ", fontsize=13)
ax.set_ylabel("Accuracy", fontsize=13)

ax.set_xticks(lambdas)
ax.set_xlim(0.05, 0.95)

y_min = min(accuracy) - 0.005
y_max = max(accuracy) + 0.008
ax.set_ylim(y_min, y_max)

# 3 zecimale pe axa OY + include valoarea exactă a baseline-ului
auto_ticks = mpl.ticker.AutoLocator()
ax.yaxis.set_major_locator(auto_ticks)
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))

ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))

# Tick-uri pornind de la 0.655 în jos cu pas 0.003
yticks = np.arange(0.655, y_min, -0.003)
yticks = sorted(yticks)
# Adaugă baseline dacă nu e deja inclus
if not any(abs(t - baseline) < 1e-6 for t in yticks):
    yticks = sorted(list(yticks) + [baseline])
ax.set_yticks(yticks)
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))

# Colorează tick-ul baseline-ului cu gri
for tick, val in zip(ax.yaxis.get_major_ticks(), ax.get_yticks()):
    if abs(val - baseline) < 1e-6:
        tick.label1.set_color("gray")

# dotted horizontal grid lines (match original style)
ax.yaxis.grid(True, linestyle=":", linewidth=0.8, color="gray", alpha=0.6)
ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpl.lines.Line2D([0], [0], color="gray",    linestyle="--", linewidth=1.4,
                     label=r"WavLM baseline (no KD, $\lambda=0$)"),
    mpl.lines.Line2D([0], [0], color="#1a3a5c", linestyle="-",  linewidth=1.8,
                     marker="o", markersize=6,
                     label="WavLM student with KD"),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=10.5, loc="lower right")

plt.tight_layout()
plt.savefig("wavlm_kd_lambda_sweep_full.pdf")
print("Saved: wavlm_kd_lambda_sweep_full.pdf")
plt.show()