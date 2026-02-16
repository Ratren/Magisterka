#!/usr/bin/env python3
"""Generate figures for the GEMV optimization paper."""

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Consistent styling
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("deep", 3)
FIGWIDTH = 4.8  # inches, fits LNCS 122mm column width

# ============================================================
# Benchmark data (median GFLOPS, 5 runs, CLOCK_MONOTONIC_RAW)
# AMD Ryzen 5 5600 (Zen 3), GCC 15.2.1
# ============================================================

presets = ["tiny\n64x64", "small\n256x256", "medium\n1024x1024",
           "large\n4096x4096", "wide\n256x8192", "tall\n8192x256"]
preset_labels_short = ["tiny", "small", "medium", "large", "wide", "tall"]

# Single-threaded (1T, pinned to core 0)
zengemv_1t  = [41.08, 26.10, 22.35, 10.01, 23.33, 21.76]
openblas_1t = [29.85, 22.91, 21.82,  9.35, 21.79, 20.42]
blis_1t     = [35.19, 21.26, 23.18,  9.35, 23.18, 21.73]

# Multi-threaded (6 physical cores, pinned to 0-5)
zengemvp_mt  = [41.01, 58.91, 75.43, 10.00, 74.32, 74.70]
openblas_mt  = [29.43, 23.22, 77.47,  6.61, 74.78, 73.54]
blis_mt      = [34.29, 58.61, 80.76,  6.01, 72.28, 80.26]

# Zen 3 parameters for roofline
PEAK_GFLOPS = 70.4     # 2 FMA * 4 doubles * 2 ops * 4.4 GHz
BW_L1  = 256.0          # 2*32 B/cycle * 4.4 GHz ~ 281 GB/s (practical ~256)
BW_L2  = 140.0          # 32 B/cycle * 4.4 GHz (practical ~140 GB/s)
BW_L3  = 106.0          # 24 B/cycle * 4.4 GHz (shared L3, per-core practical ~106)
BW_DRAM = 42.0          # DDR4-3200 dual-channel practical ~42 GB/s

# Working set sizes and arithmetic intensities per preset
# Working set = (M*N + N + 2*M) * 8 bytes
# AI = 2*M*N / (8*(M*N + N + 2*M))
preset_dims = [(64,64), (256,256), (1024,1024), (4096,4096), (256,8192), (8192,256)]

def compute_ai(m, n):
    return 2.0*m*n / (8.0*(m*n + n + 2*m))

def compute_ws(m, n):
    return (m*n + n + 2*m) * 8  # bytes

ai_values = [compute_ai(m,n) for m,n in preset_dims]
ws_values = [compute_ws(m,n) for m,n in preset_dims]

# ============================================================
# Figure 1: Roofline Model
# ============================================================
def plot_roofline():
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 3.5))

    ai_range = np.logspace(-1.2, 1.0, 500)  # narrowed: ~0.06 to 10

    # Bandwidth ceilings
    ceil_styles = [
        (BW_DRAM, "DRAM (42 GB/s)",  "-",  1.2),
        (BW_L3,   "L3 (106 GB/s)",   "--", 1.0),
        (BW_L2,   "L2 (140 GB/s)",   "-.", 1.0),
        (BW_L1,   "L1d (256 GB/s)",  ":",  1.2),
    ]
    ceil_lines = []
    ceil_labels = []
    for bw, label, ls, lw in ceil_styles:
        perf = np.minimum(PEAK_GFLOPS, bw * ai_range)
        line, = ax.loglog(ai_range, perf, ls, color='gray', alpha=0.6, linewidth=lw)
        ceil_lines.append(line)
        ceil_labels.append(label)

    # Peak compute ceiling
    peak_line = ax.axhline(y=PEAK_GFLOPS, color='darkred', linewidth=1.5, alpha=0.7)
    ceil_lines.append(peak_line)
    ceil_labels.append(f"Peak ({PEAK_GFLOPS} GFLOPS)")

    # Plot measured ZenGEMV performance at AI ≈ 0.25 for all presets
    # (exact AI varies from 0.239 to 0.250 due to vector terms, but
    #  the differences are negligible — use 0.25 for clarity)
    GEMV_AI = 0.25
    colors_pts = sns.color_palette("bright", 6)
    # Sort by performance so higher GFLOPS dots are drawn on top
    order = sorted(range(len(zengemv_1t)), key=lambda i: zengemv_1t[i])
    preset_handles = [None] * len(zengemv_1t)
    for z, i in enumerate(order):
        perf = zengemv_1t[i]
        h = ax.plot(GEMV_AI, perf, 'o', markersize=4, color=colors_pts[i],
                zorder=5 + z, markeredgecolor='black', markeredgewidth=0.4)[0]
        preset_handles[i] = h

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_xlim(0.08, 8)
    ax.set_ylim(5, 80)
    ax.set_title("Roofline Model: ZenGEMV on Zen 3", fontsize=10)

    # Simple readable tick labels
    from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
    xticks = [0.1, 0.25, 0.5, 1, 2, 5]
    yticks = [5, 10, 20, 30, 50, 70]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    def fmt_tick(val, pos):
        if val == int(val):
            return f"{int(val)}"
        return f"{val:g}"
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tick))

    # Legend: combine ceiling lines and preset points
    all_handles = ceil_lines + preset_handles
    all_labels = ceil_labels + [f"{name} ({perf:.1f} GFLOPS)" for name, perf
                                in zip(preset_labels_short, zengemv_1t)]
    ax.legend(all_handles, all_labels, fontsize=5.5, loc='lower right',
              ncol=2, framealpha=0.9)

    plt.tight_layout()
    fig.savefig("roofline.pdf", bbox_inches='tight')
    plt.close()
    print("Generated roofline.pdf")


# ============================================================
# Figure 2: Single-Threaded Comparison
# ============================================================
def plot_single_threaded():
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 3.8))

    x = np.arange(len(presets))
    width = 0.28

    bars1 = ax.bar(x - width, zengemv_1t, width, label="ZenGEMV",
                   color=PALETTE[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, openblas_1t, width, label="OpenBLAS",
                   color=PALETTE[1], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, blis_1t, width, label="BLIS",
                   color=PALETTE[2], edgecolor='black', linewidth=0.5)

    ax.set_ylabel("GFLOPS (median, 5 runs)")
    ax.set_title("Single-Threaded GEMV Performance", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(presets, fontsize=7)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, max(zengemv_1t) * 1.4)

    # GFLOPS values + speedup over OpenBLAS on ZenGEMV bars
    for i, (v, o) in enumerate(zip(zengemv_1t, openblas_1t)):
        speedup = v / o
        ax.annotate(f"{v:.1f}\n{speedup:.2f}x", xy=(x[i] - width, v),
                    ha='center', va='bottom', fontsize=5, fontweight='bold',
                    color=PALETTE[0])

    # GFLOPS values on OpenBLAS bars
    for i, val in enumerate(openblas_1t):
        ax.annotate(f"{val:.1f}", xy=(x[i], val),
                    ha='center', va='bottom', fontsize=5,
                    color=PALETTE[1])

    # GFLOPS values on BLIS bars
    for i, val in enumerate(blis_1t):
        ax.annotate(f"{val:.1f}", xy=(x[i] + width, val),
                    ha='center', va='bottom', fontsize=5,
                    color=PALETTE[2])

    plt.tight_layout()
    fig.savefig("single_threaded.pdf", bbox_inches='tight')
    plt.close()
    print("Generated single_threaded.pdf")


# ============================================================
# Figure 3: Multi-Threaded Comparison
# ============================================================
def plot_multi_threaded():
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 3.8))

    x = np.arange(len(presets))
    width = 0.28

    bars1 = ax.bar(x - width, zengemvp_mt, width, label="ZenGEMV-P",
                   color=PALETTE[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, openblas_mt, width, label="OpenBLAS",
                   color=PALETTE[1], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, blis_mt, width, label="BLIS",
                   color=PALETTE[2], edgecolor='black', linewidth=0.5)

    ax.set_ylabel("GFLOPS (median, 5 runs)")
    ax.set_title("Multi-Threaded GEMV Performance (6 cores)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(presets, fontsize=7)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8, loc='upper right')

    # GFLOPS values + speedup over OpenBLAS on ZenGEMV-P bars
    for i, (v, o) in enumerate(zip(zengemvp_mt, openblas_mt)):
        speedup = v / o
        ax.annotate(f"{v:.1f}\n{speedup:.2f}x", xy=(x[i] - width, v),
                    ha='center', va='bottom', fontsize=5, fontweight='bold',
                    color=PALETTE[0])

    # GFLOPS values on OpenBLAS bars
    for i, val in enumerate(openblas_mt):
        ax.annotate(f"{val:.1f}", xy=(x[i], val),
                    ha='center', va='bottom', fontsize=5,
                    color=PALETTE[1])

    # GFLOPS values on BLIS bars
    for i, val in enumerate(blis_mt):
        ax.annotate(f"{val:.1f}", xy=(x[i] + width, val),
                    ha='center', va='bottom', fontsize=5,
                    color=PALETTE[2])

    plt.tight_layout()
    fig.savefig("multi_threaded.pdf", bbox_inches='tight')
    plt.close()
    print("Generated multi_threaded.pdf")


if __name__ == '__main__':
    plot_roofline()
    plot_single_threaded()
    plot_multi_threaded()
