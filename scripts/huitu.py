import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data
# =========================
categories = [
    "Faithfulness /\nAdequacy",
    "Fluency /\nNaturalness",
    "Style /\nTone",
    "Literalness",
    "Mistranslation /\nMeaning error",
    "Word choice /\nExpression",
    "Awkwardness /\nGrammar",
    "Context\nSensitivity"
]

human_raw = np.array([13, 11, 7, 13, 9, 18, 11, 5], dtype=float)
deepseek_raw = np.array([3, 14, 13, 5, 0, 16, 8, 3], dtype=float)
gemini_raw = np.array([5, 11, 4, 8, 6, 3, 8, 4], dtype=float)

# =========================
# Normalize by source total
# =========================
human = human_raw / human_raw.sum()
deepseek = deepseek_raw / deepseek_raw.sum()
gemini = gemini_raw / gemini_raw.sum()

# =========================
# Radar prep
# =========================
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_closed = np.concatenate([angles, [angles[0]]])

human_plot = np.concatenate([human, [human[0]]])
deepseek_plot = np.concatenate([deepseek, [deepseek[0]]])
gemini_plot = np.concatenate([gemini, [gemini[0]]])

max_value = max(human.max(), deepseek.max(), gemini.max())
upper = max_value + 0.04

levels = 5
r_ticks = np.linspace(upper / levels, upper, levels)

# =========================
# Figure
# =========================
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(10, 10.5))
ax = plt.subplot(111, polar=True)

# First axis at top, clockwise
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_ylim(0, upper)

# Remove default circular grid and border
ax.grid(False)
ax.spines["polar"].set_visible(False)

# Remove default ticks
ax.set_yticks([])
ax.set_xticks([])

# =========================
# Custom polygon grid
# =========================
grid_color = "#B8B8B8"

for r in r_ticks:
    ax.plot(
        angles_closed,
        [r] * len(angles_closed),
        linestyle="dashed",
        linewidth=0.9,
        color=grid_color,
        alpha=0.8,
        zorder=0
    )

for angle in angles:
    ax.plot(
        [angle, angle],
        [0, upper],
        linestyle="dashed",
        linewidth=0.9,
        color=grid_color,
        alpha=0.8,
        zorder=0
    )

# =========================
# Category labels
# 顶部和底部单独调整，避免碰标题/图例
# =========================
label_radius_default = upper + 0.06
label_radius_top = upper + 0.015
label_radius_bottom = upper + 0.015

for i, (angle, label) in enumerate(zip(angles, categories)):
    if i == 0:        # top
        radius = label_radius_top
    elif i == 4:      # bottom
        radius = label_radius_bottom
    else:
        radius = label_radius_default

    ax.text(
        angle,
        radius,
        label,
        fontsize=11,
        ha="center",
        va="center",
        clip_on=False
    )

# =========================
# Plot series
# =========================
line1, = ax.plot(
    angles_closed, human_plot,
    linewidth=2.3,
    label="Human comments",
    zorder=3
)
ax.fill(angles_closed, human_plot, alpha=0.16, zorder=2)

line2, = ax.plot(
    angles_closed, deepseek_plot,
    linewidth=2.3,
    label="DeepSeek comments",
    zorder=3
)
ax.fill(angles_closed, deepseek_plot, alpha=0.16, zorder=2)

line3, = ax.plot(
    angles_closed, gemini_plot,
    linewidth=2.3,
    label="Gemini comments",
    zorder=3
)
ax.fill(angles_closed, gemini_plot, alpha=0.16, zorder=2)

# =========================
# Title
# =========================
plt.title(
    "Comparative Emphases in Translation Evaluation Across Different Sources",
    fontsize=18,
    pad=36
)

# =========================
# Legend at bottom, horizontal
# =========================
ax.legend(
    handles=[line1, line2, line3],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    frameon=False,
    fontsize=11
)

plt.tight_layout()
plt.show()