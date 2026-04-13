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
FIGWIDTH = 4.8
_DIR = os.path.dirname(os.path.abspath(__file__))

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
BW_DRAM = 51.2

preset_dims = [(64,64), (256,256), (1024,1024), (4096,4096), (256,8192), (8192,256)]

def compute_ai(m, n):
    return 2.0*m*n / (8.0*(m*n + n + 2*m))

ai_values = [compute_ai(m,n) for m,n in preset_dims]

def plot_roofline():
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator

    ai_range = np.logspace(-1.2, 1.0, 500)

    # Diagonal bandwidth lines — subtle background context
    diag_styles = [
        (BW_DRAM, "DRAM",  "-"),
        (BW_L3,   "L3",    "--"),
        (BW_L2,   "L2",    "-."),
        (BW_L1,   "L1d",   ":"),
    ]
    # Position labels along each diagonal at fixed y-values so they're
    # evenly spaced vertically and visible within the zoomed axes
    diag_label_y = {'L1d': 20, 'L2': 14, 'L3': 11, 'DRAM': 11}
    for bw, label, ls in diag_styles:
        perf = np.minimum(PEAK_GFLOPS, bw * ai_range)
        ax.loglog(ai_range, perf, ls, color='#bbbbbb', alpha=0.5, linewidth=0.8)
        ly = diag_label_y[label]
        lx = ly / bw  # point on the diagonal
        ax.text(lx, ly * 0.88, label, fontsize=5.5, color='#999999',
                ha='center', va='top', rotation=28)

    # Peak compute line
    ax.axhline(y=PEAK_GFLOPS, color='darkred', linewidth=1.2, alpha=0.5)
    ax.text(0.09, PEAK_GFLOPS, f"Peak {PEAK_GFLOPS}", fontsize=5.5,
            color='darkred', alpha=0.7, va='bottom', ha='left')

    # Horizontal GEMV ceilings — the key reference lines
    # Skip L1d ceiling (66.8) — too close to Peak (70.4), adds clutter
    gemv_ceilings = [
        (BW_L2 * 0.25,   "L2 ceiling (33.0)",   '#1f77b4', (4, 3)),
        (BW_L3 * 0.25,   "L3 ceiling (26.5)",   '#2ca02c', (4, 3)),
        (BW_DRAM * 0.25, "DRAM ceiling (12.8)",  '#d62728', (6, 2)),
    ]
    for ceil_gflops, label, color, dashes in gemv_ceilings:
        ax.axhline(y=ceil_gflops, color=color, linewidth=1.0, alpha=0.55,
                   dashes=dashes)
        ax.text(1.4, ceil_gflops * 1.03, label, fontsize=5.5, color=color,
                alpha=0.85, va='bottom', ha='right')

    # Vertical reference at OI = 0.25
    ax.axvline(x=0.25, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')

    # Data points — ZenGEMV only
    for i in range(len(PRESET_NAMES)):
        ax.plot(ai_values[i], zengemv_1t[i], 'o', markersize=5, color=PALETTE[0],
                markeredgecolor='black', markeredgewidth=0.5, zorder=10)
    ax.plot([], [], 'o', color=PALETTE[0], markeredgecolor='black',
            markeredgewidth=0.5, label='ZenGEMV', markersize=5)

    # Preset labels
    annot_cfg = {
        'tiny':   (16,  2),
        'small':  (16,  0),
        'medium': (16, -2),
        'large':  (16,  2),
        'wide':   (16, -8),
        'tall':   (16,  6),
    }
    for i, name in enumerate(preset_labels_short):
        dx, dy = annot_cfg[name]
        ax.annotate(name, xy=(ai_values[i], zengemv_1t[i]), fontsize=6,
                    ha='left', va='center', xytext=(dx, dy),
                    textcoords='offset points', fontstyle='italic',
                    arrowprops=dict(arrowstyle='-', color='gray',
                                   lw=0.5, alpha=0.5))

    ax.set_xlabel("Operational Intensity (FLOP/byte)", fontsize=9)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=9)
    ax.set_xlim(0.08, 1.5)
    ax.set_ylim(8, 75)
    ax.set_title("Roofline Model: GEMV on Zen 3", fontsize=11)

    xticks = [0.1, 0.15, 0.25, 0.5, 1]
    yticks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    def fmt_tick(val, pos):
        return f"{int(val)}" if val == int(val) else f"{val:g}"
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tick))
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(axis='y', linewidth=0.3, alpha=0.6)
    ax.grid(axis='x', visible=False)

    ax.legend(fontsize=7, loc='upper right', framealpha=0.9,
              edgecolor='gray', fancybox=False)

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
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis='y', linewidth=0.3, alpha=0.6)
    ax.grid(axis='x', visible=False)

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
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis='y', linewidth=0.3, alpha=0.6)
    ax.grid(axis='x', visible=False)

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


BW_CSV = os.path.join(_DIR, "bandwidth_results.csv")

def parse_bandwidth_csv_all(csv_path):
    """Parse all sections from the bandwidth tool's CSV output.
    CSV sizes are in bytes; we convert to KB for plotting.
    CSV bandwidth is in MB/s; we convert to GB/s."""
    sections = {}
    current = None
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line[0].isdigit():
                current = line.strip()
                sections[current] = ([], [])
                continue
            if current is not None:
                parts = line.split(',')
                size_bytes = float(parts[0].strip())
                bw_mbs = float(parts[1].strip())
                sections[current][0].append(size_bytes / 1024.0)  # -> KB
                sections[current][1].append(bw_mbs / 1000.0)      # -> GB/s
    return {k: (np.array(s), np.array(b)) for k, (s, b) in sections.items()}


def plot_bandwidth():
    from matplotlib.ticker import NullLocator, MultipleLocator

    all_data = parse_bandwidth_csv_all(BW_CSV)

    fig, ax = plt.subplots(figsize=(FIGWIDTH, 4.5))

    # Group by operation type with distinct colors and line styles
    line_specs = {
        'Sequential 64-bit reads':              ('#1f77b4', '-',  '64-bit reads'),
        'Sequential 128-bit reads':             ('#aec7e8', '-',  '128-bit reads'),
        'Sequential 256-bit reads':             ('#0b3d91', '-',  '256-bit reads'),
        'Sequential 128-bit nontemporal reads': ('#7fcdbb', '--', '128-bit NT reads'),
        'Sequential 256-bit nontemporal reads': ('#2ca02c', '--', '256-bit NT reads'),
        'Sequential 64-bit writes':             ('#d62728', '-',  '64-bit writes'),
        'Sequential 128-bit writes':            ('#ff9896', '-',  '128-bit writes'),
        'Sequential 256-bit writes':            ('#8c0000', '-',  '256-bit writes'),
        'Sequential 64-bit nontemporal writes': ('#ff7f0e', '--', '64-bit NT writes'),
        'Sequential 128-bit nontemporal writes':('#ffbb78', '--', '128-bit NT writes'),
        'Sequential 256-bit nontemporal writes':('#c45b00', '--', '256-bit NT writes'),
        'Sequential 64-bit copy':               ('#9467bd', '-',  '64-bit copy'),
        'Sequential 128-bit copy':              ('#c5b0d5', '-',  '128-bit copy'),
        'Sequential 256-bit copy':              ('#6a0dad', '-',  '256-bit copy'),
    }

    for section_name, (color, ls, label) in line_specs.items():
        if section_name not in all_data:
            continue
        sizes_kb, bw = all_data[section_name]
        ax.semilogx(sizes_kb, bw, ls, color=color, linewidth=0.4,
                    label=label)

    # Cache boundary shading — gradient-like distinct colors
    xmin_kb = 0.25    # 256 B
    xmax_kb = 131072  # 128 MB
    regions = [
        (xmin_kb, 32,     '#d0e8ff'),   # L1d — blue tint
        (32,      512,    '#c8e6c8'),   # L2  — green tint
        (512,     32768,  '#fff3cc'),   # L3  — yellow tint
        (32768,   xmax_kb,'#fdd'),      # DRAM — red tint
    ]
    for lo, hi, c in regions:
        ax.axvspan(lo, hi, color=c, alpha=0.25, zorder=0)

    # Region labels at top
    ax.text(np.sqrt(xmin_kb * 32), 278, 'L1d', fontsize=6, color='#555555',
            ha='center', va='center', fontstyle='italic')
    ax.text(np.sqrt(32 * 512), 278, 'L2', fontsize=6, color='#555555',
            ha='center', va='center', fontstyle='italic')
    ax.text(np.sqrt(512 * 32768), 278, 'L3', fontsize=6, color='#555555',
            ha='center', va='center', fontstyle='italic')
    ax.text(np.sqrt(32768 * xmax_kb), 278, 'DRAM', fontsize=6, color='#555555',
            ha='center', va='center', fontstyle='italic')

    ax.set_xlabel("Buffer Size", fontsize=8.5)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=8.5)
    ax.set_title("AMD Ryzen 5 5600 Memory Bandwidth", fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    ax.set_xlim(xmin_kb, xmax_kb)
    ax.set_ylim(0, 290)

    # Y-axis: gridlines every 25 GB/s, only horizontal
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.grid(axis='y', linewidth=0.3, alpha=0.6)
    ax.grid(axis='x', visible=False)

    xticks_kb =  [0.25,   1,     4,     32,    512,    4096,  32768,  131072]
    xtick_lbls = ['256B', '1K', '4K', '32K', '512K', '4M',  '32M',  '128M']
    ax.set_xticks(xticks_kb)
    ax.set_xticklabels(xtick_lbls, fontsize=6)
    ax.xaxis.set_minor_locator(NullLocator())

    ax.legend(fontsize=4.5, loc='upper center', ncol=3,
              bbox_to_anchor=(0.5, -0.15),
              framealpha=0.9, edgecolor='gray', fancybox=False)

    plt.tight_layout()
    fig.savefig("bandwidth.pdf", bbox_inches='tight')
    plt.close()
    print("Generated bandwidth.pdf")


if __name__ == '__main__':
    plot_bandwidth()
    plot_roofline()
    plot_single_threaded()
    plot_multi_threaded()
