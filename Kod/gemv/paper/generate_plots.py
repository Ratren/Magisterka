#!/usr/bin/env python3
"""Generate figures for the GEMV optimization paper."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("deep", 3)
FIGWIDTH = 4.8_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_1T = os.path.join(_DIR, "results_1t.json")
JSON_6T = os.path.join(_DIR, "results_6t.json")

PRESET_NAMES = ["tiny", "small", "medium", "large", "wide", "tall"]

def _match_impl(impl_names, pattern, exclude=None):
    for name in impl_names:
        if pattern in name and (exclude is None or exclude not in name):
            return name
    return None

def load_results(json_path, zengemv_pattern, zengemv_exclude=None):
    with open(json_path) as f:
        data = json.load(f)

    by_name = {p["name"]: p for p in data["presets"]}
    zengemv_vals, openblas_vals, blis_vals = [], [], []

    for preset_name in PRESET_NAMES:
        if preset_name not in by_name:
            sys.exit(f"Error: preset '{preset_name}' not found in {json_path}")
        impls = by_name[preset_name]["implementations"]
        impl_names = list(impls.keys())

        zkey = _match_impl(impl_names, zengemv_pattern, zengemv_exclude)
        okey = _match_impl(impl_names, "OpenBLAS")
        bkey = _match_impl(impl_names, "BLIS")

        if not zkey:
            sys.exit(f"Error: could not find '{zengemv_pattern}' implementation in {json_path}")
        if not okey:
            sys.exit(f"Error: could not find 'OpenBLAS' implementation in {json_path}")
        if not bkey:
            sys.exit(f"Error: could not find 'AOCL-BLAS' implementation in {json_path}")

        zengemv_vals.append(impls[zkey]["median"])
        openblas_vals.append(impls[okey]["median"])
        blis_vals.append(impls[bkey]["median"])

    return zengemv_vals, openblas_vals, blis_vals

zengemv_1t, openblas_1t, blis_1t   = load_results(JSON_1T, "V3", zengemv_exclude="OMP")
zengemvp_mt, openblas_mt, blis_mt  = load_results(JSON_6T, "V3_OMP")

print(f"Loaded 1T results from {JSON_1T}")
print(f"Loaded 6T results from {JSON_6T}")

presets = ["tiny\n64x64", "small\n256x256", "medium\n1024x1024",
           "large\n4096x4096", "wide\n256x8192", "tall\n8192x256"]
preset_labels_short = ["tiny", "small", "medium", "large", "wide", "tall"]

PEAK_GFLOPS = 70.4
BW_L1  = 267.0
BW_L2  = 132.0
BW_L3  = 106.0
BW_DRAM = 42.0

preset_dims = [(64,64), (256,256), (1024,1024), (4096,4096), (256,8192), (8192,256)]

def compute_ai(m, n):
    return 2.0*m*n / (8.0*(m*n + n + 2*m))

ai_values = [compute_ai(m,n) for m,n in preset_dims]

def plot_roofline():
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 3.5))

    ai_range = np.logspace(-1.2, 1.0, 500)

    ceil_styles = [
        (BW_DRAM, "DRAM (42 GB/s)",  "-",  1.2),
        (BW_L3,   "L3 (106 GB/s)",   "--", 1.0),
        (BW_L2,   "L2 (132 GB/s)",   "-.", 1.0),
        (BW_L1,   "L1d (267 GB/s)",  ":",  1.2),
    ]
    ceil_lines, ceil_labels = [], []
    for bw, label, ls, lw in ceil_styles:
        perf = np.minimum(PEAK_GFLOPS, bw * ai_range)
        line, = ax.loglog(ai_range, perf, ls, color='gray', alpha=0.6, linewidth=lw)
        ceil_lines.append(line)
        ceil_labels.append(label)

    peak_line = ax.axhline(y=PEAK_GFLOPS, color='darkred', linewidth=1.5, alpha=0.7)
    ceil_lines.append(peak_line)
    ceil_labels.append(f"Peak ({PEAK_GFLOPS} GFLOPS)")

    colors_pts = sns.color_palette("bright", 6)
    order = sorted(range(len(zengemv_1t)), key=lambda i: zengemv_1t[i])
    preset_handles = [None] * len(zengemv_1t)
    for z, i in enumerate(order):
        perf = zengemv_1t[i]
        ai = ai_values[i]
        h = ax.plot(ai, perf, 'o', markersize=4, color=colors_pts[i],
                zorder=5 + z, markeredgecolor='black', markeredgewidth=0.4)[0]
        preset_handles[i] = h

    ax.set_xlabel("Operational Intensity (FLOP/byte)", fontsize=8.5)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=8.5)
    ax.set_xlim(0.08, 8)
    ax.set_ylim(5, 80)
    ax.set_title("Roofline Model: ZenGEMV on Zen 3", fontsize=10)

    from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
    xticks = [0.1, 0.25, 0.5, 1, 2, 5]
    yticks = [5, 10, 20, 30, 50, 70]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    def fmt_tick(val, pos):
        return f"{int(val)}" if val == int(val) else f"{val:g}"
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tick))
    ax.tick_params(axis='both', labelsize=6)

    all_handles = ceil_lines + preset_handles
    all_labels = ceil_labels + [f"{name} (OI={ai:.3f}, {perf:.1f} GFLOPS)"
                                 for name, ai, perf
                                 in zip(preset_labels_short, ai_values, zengemv_1t)]
    ax.legend(all_handles, all_labels, fontsize=4.5, loc='lower right',
              ncol=2, framealpha=0.9)

    plt.tight_layout()
    fig.savefig("roofline.pdf", bbox_inches='tight')
    plt.close()
    print("Generated roofline.pdf")


def plot_single_threaded():
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 3.8))

    x = np.arange(len(presets))
    width = 0.30

    ax.bar(x - width, zengemv_1t, width, label="ZenGEMV",
           color=PALETTE[0], edgecolor='black', linewidth=0.5)
    ax.bar(x, openblas_1t, width, label="OpenBLAS",
           color=PALETTE[1], edgecolor='black', linewidth=0.5)
    ax.bar(x + width, blis_1t, width, label="AOCL-BLAS",
           color=PALETTE[2], edgecolor='black', linewidth=0.5)

    ax.set_ylabel("GFLOPS (median, 5 runs)", fontsize=8.5)
    ax.set_title("Single-Threaded GEMV Performance", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(presets, fontsize=7)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_ylim(0, max(zengemv_1t) * 1.4)

    for i, (v, o) in enumerate(zip(zengemv_1t, openblas_1t)):
        ax.annotate(f"{v:.1f}\n{v/o:.2f}x", xy=(x[i] - width, v),
                    ha='center', va='bottom', fontsize=5, fontweight='bold',
                    color=PALETTE[0])
    for i, val in enumerate(openblas_1t):
        ax.annotate(f"{val:.1f}", xy=(x[i], val),
                    ha='center', va='bottom', fontsize=5, color=PALETTE[1])
    for i, val in enumerate(blis_1t):
        ax.annotate(f"{val:.1f}", xy=(x[i] + width, val),
                    ha='center', va='bottom', fontsize=5, color=PALETTE[2])

    plt.tight_layout()
    fig.savefig("single_threaded.pdf", bbox_inches='tight')
    plt.close()
    print("Generated single_threaded.pdf")


def plot_multi_threaded():
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 3.8))

    x = np.arange(len(presets))
    width = 0.30

    ax.bar(x - width, zengemvp_mt, width, label="ZenGEMV-P",
           color=PALETTE[0], edgecolor='black', linewidth=0.5)
    ax.bar(x, openblas_mt, width, label="OpenBLAS",
           color=PALETTE[1], edgecolor='black', linewidth=0.5)
    ax.bar(x + width, blis_mt, width, label="AOCL-BLAS",
           color=PALETTE[2], edgecolor='black', linewidth=0.5)

    ax.set_ylabel("GFLOPS (median, 5 runs)", fontsize=8.5)
    ax.set_title("Multi-Threaded GEMV Performance (6 cores)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(presets, fontsize=7)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=7, loc='upper left')

    for i, (v, o) in enumerate(zip(zengemvp_mt, openblas_mt)):
        ax.annotate(f"{v:.1f}\n{v/o:.2f}x", xy=(x[i] - width, v),
                    ha='center', va='bottom', fontsize=5, fontweight='bold',
                    color=PALETTE[0])
    for i, val in enumerate(openblas_mt):
        ax.annotate(f"{val:.1f}", xy=(x[i], val),
                    ha='center', va='bottom', fontsize=5, color=PALETTE[1])
    for i, val in enumerate(blis_mt):
        ax.annotate(f"{val:.1f}", xy=(x[i] + width, val),
                    ha='center', va='bottom', fontsize=5, color=PALETTE[2])

    plt.tight_layout()
    fig.savefig("multi_threaded.pdf", bbox_inches='tight')
    plt.close()
    print("Generated multi_threaded.pdf")


if __name__ == '__main__':
    plot_roofline()
    plot_single_threaded()
    plot_multi_threaded()
