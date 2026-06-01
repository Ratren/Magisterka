#!/usr/bin/env python3
"""Generate comparison figures for all benchmark kernels.

Reads the combined_<N>t.json files produced by run_suite.py and emits one
PDF per (kernel, thread mode) showing the best of our implementations
against the vendor libraries, in the same visual style as
Kod/gemv/paper/generate_plots.py (excluding the roofline analysis).

  $ python3 Kod/benchmark_suite/generate_plots.py
  $ python3 Kod/benchmark_suite/generate_plots.py --results <dir> --out <dir>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("deep", 6)
FIGWIDTH = 6.0

VENDOR_PATTERNS = ("OpenBLAS", "BLIS", "libxsmm")
NAIVE_PATTERNS = ("Naive", "naive")


# Polish descriptive labels per implementation, keyed by the C-side
# benchmark name. Each label says *what optimisation* the variant adds on
# top of the previous step. Vendor libraries keep their proper names.
# Substring match — entries earlier in this dict have priority when
# multiple keys appear in the same name (e.g. "OMP" is part of "MT").
IMPL_LABELS = {
    # dot_product
    "SIMD MultiAcc":         "SIMD AVX2 (4 akumulatory)",
    "SIMD":                  "SIMD AVX2 (1 akumulator)",
    "OMP":                   "SIMD AVX2 + OpenMP",

    # gemv
    "SIMD + Prefetch":       "SIMD AVX2 + prefetch",
    "AVX+FMA Blocked":       "AVX2/FMA blokowy (4 wiersze)",
    "AVX+FMA V2":            "AVX2/FMA + 4 akumulatory",
    "AVX+FMA V3_OMP":        "ZenGEMV-P (OpenMP)",
    "AVX+FMA V3":            "AVX2/FMA + 8 akumulatorów (ZenGEMV)",

    # gemm
    "Loop Reorder ikj":      "Zmiana kolejności pętli (ikj)",
    "Loop Reorder":          "Zmiana kolejności pętli",
    "Blocked":               "Blokowanie cache",
    "ST 6x8 packed":         "Pakowanie 6×8 (1T)",
    "MT 6x8 packed":         "Pakowanie 6×8 + OpenMP",
    "ST 4x12 packed":        "Pakowanie 4×12 (1T)",
    "ST 4x12 tiny":          "4×12 bez pakowania (małe macierze)",
    "MT 4x12 per-thread B":  "4×12 (MT, osobne B)",
    "MT 4x12 shared B":      "4×12 (MT, wspólne B)",
    "MT 4x12 best":          "4×12 + dyspozytor (MT)",
    "MT Strassen":           "Strassen (7 mnożeń, MT)",

    # conv
    "Packed Direct":         "Pakowanie 6×16 (1T)",
    "OMP Packed":            "Pakowanie 6×16 + OpenMP",
    "NCHWc direct":          "Układ blokowy NCHWc8",
    "1x1 (SGEMM)":           "Splot 1×1 → SGEMM",
    "Winograd F(2,3)":       "Winograd F(2×2, 3×3)",
    "Zen3 dispatch OMP":     "Dyspozytor Zen3 (MT)",
    "im2col + OpenBLAS":     "im2col + OpenBLAS",
    "im2col + BLIS":         "im2col + BLIS",
    "libxsmm":               "im2col + libxsmm",

    # universal
    "Naive":                 "Naiwna pętla",
}


def polish_label(name: str) -> str:
    """Translate a C-side impl name to its Polish descriptive label.
    Strips any trailing thread-count suffix like "(6T)" before matching;
    vendor names that already contain their library identifier are kept
    as-is when no match is found."""
    stripped = name
    for sep in (" (", "("):
        idx = stripped.rfind(sep)
        if idx != -1 and ("T)" in stripped[idx:] or "T " in stripped[idx:]):
            stripped = stripped[:idx].rstrip()
            break
    for key, label in IMPL_LABELS.items():
        if key in stripped:
            return label
    return name


def is_vendor(name: str) -> bool:
    return any(p in name for p in VENDOR_PATTERNS)


def is_naive(name: str) -> bool:
    return any(p in name for p in NAIVE_PATTERNS)


def best_of_ours(impls: dict) -> tuple[str, float] | None:
    """Return (name, median) of the best non-vendor non-naive impl, or None."""
    best = None
    for name, vals in impls.items():
        if is_vendor(name) or is_naive(name):
            continue
        m = vals.get("median", 0.0)
        if not m or m <= 0.0:
            continue
        if best is None or m > best[1]:
            best = (name, m)
    return best


def get_vendor(impls: dict, pattern: str) -> tuple[str, float] | None:
    for name, vals in impls.items():
        if pattern in name:
            m = vals.get("median", 0.0)
            if not m or m <= 0.0:
                return None
            return name, m
    return None


def collect(kernel_data: dict) -> list[dict]:
    rows = []
    for preset in kernel_data.get("presets", []):
        impls = preset["implementations"]
        ours = best_of_ours(impls)
        if ours is None:
            continue
        vendors = []
        for vp in VENDOR_PATTERNS:
            v = get_vendor(impls, vp)
            if v:
                vendors.append((vp, v[1]))
        rows.append({
            "preset": preset["name"],
            "params": preset.get("params", {}),
            "ours_name": ours[0],
            "ours_median": ours[1],
            "vendors": vendors,
        })
    return rows


def short_preset_label(preset_name: str, params: dict, kernel: str) -> str:
    base = preset_name.split("_", 1)[1] if "_" in preset_name else preset_name
    if kernel == "dot" and "size" in params:
        return f"{base}\n{params['size']}"
    if kernel == "gemv" and "rows" in params and "cols" in params:
        return f"{base}\n{params['rows']}x{params['cols']}"
    if kernel == "gemm" and {"m", "n", "k"} <= params.keys():
        return f"{base}\n{params['m']}x{params['n']}x{params['k']}"
    if kernel == "conv" and {"cin", "h", "w", "k", "cout"} <= params.keys():
        return (f"{base}\n{params['cin']}x{params['h']}x{params['w']}"
                f"\nK={params['k']} Co={params['cout']}")
    return base


def _thread_word(n: int) -> str:
    """Polish plural form for "thread"."""
    if n == 1:
        return "wątek"
    if 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
        return "wątki"
    return "wątków"


def plot_kernel(kernel: str, rows: list[dict], threads: int, out_path: Path):
    if not rows:
        return

    n_presets = len(rows)
    vendor_set = sorted({v[0] for r in rows for v in r["vendors"]},
                        key=lambda x: VENDOR_PATTERNS.index(x))
    n_bars = 1 + len(vendor_set)
    group_width = 0.8
    width = group_width / n_bars
    x = np.arange(n_presets)

    fig, ax = plt.subplots(figsize=(FIGWIDTH, 4.0))

    ours_vals = [r["ours_median"] for r in rows]
    ours_offset = -(n_bars - 1) * width / 2
    ax.bar(x + ours_offset, ours_vals, width,
           label="Autorskie", color=PALETTE[0],
           edgecolor='black', linewidth=0.5)

    # Vendor BLAS libraries don't ship native convolution kernels;
    # everything in the conv "vendor" group goes through im2col + SGEMM.
    # Make the legend say so explicitly.
    vendor_label_prefix = "im2col + " if kernel == "conv" else ""

    vendor_vals = {}
    for i, vp in enumerate(vendor_set):
        vals = []
        for r in rows:
            v = next((v[1] for v in r["vendors"] if v[0] == vp), 0.0)
            vals.append(v)
        vendor_vals[vp] = vals
        offset = ours_offset + (i + 1) * width
        ax.bar(x + offset, vals, width,
               label=f"{vendor_label_prefix}{vp}",
               color=PALETTE[(i + 1) % len(PALETTE)],
               edgecolor='black', linewidth=0.5)

    ax.set_ylabel("GFLOPS (mediana z 5 uruchomień)", fontsize=8.5)
    if threads == 1:
        title_thread = "jednowątkowo"
    else:
        title_thread = f"{threads} {_thread_word(threads)}"
    ax.set_title(f"Wydajność {kernel.upper()} — {title_thread}", fontsize=10)
    ax.set_xticks(x)
    labels = [short_preset_label(r["preset"], r["params"], kernel) for r in rows]
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9,
              edgecolor='gray', fancybox=False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis='y', linewidth=0.3, alpha=0.6)
    ax.grid(axis='x', visible=False)

    all_max = max(ours_vals + [v for vs in vendor_vals.values() for v in vs])
    ax.set_ylim(0, all_max * 1.30)

    for i, r in enumerate(rows):
        ref = next((v[1] for v in r["vendors"] if v[0] == "OpenBLAS"), None)
        if ref and ref > 0 and r["ours_median"] > 0:
            txt = f"{r['ours_median']:.1f}\n{r['ours_median']/ref:.2f}x"
        else:
            txt = f"{r['ours_median']:.1f}"
        ax.annotate(txt, xy=(x[i] + ours_offset, r["ours_median"]),
                    ha='center', va='bottom', fontsize=5,
                    fontweight='bold', color=PALETTE[0])
    for j, vp in enumerate(vendor_set):
        offset = ours_offset + (j + 1) * width
        for i, v in enumerate(vendor_vals[vp]):
            if v <= 0:
                continue
            ax.annotate(f"{v:.1f}", xy=(x[i] + offset, v),
                        ha='center', va='bottom', fontsize=5,
                        color=PALETTE[(j + 1) % len(PALETTE)])

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


FIGWIDTH_BIG = 14.0


def _params_str(params: dict, kernel: str) -> str:
    """Compact one-line parameter summary for a subplot title."""
    if kernel == "dot" and "size" in params:
        return f"N={params['size']}"
    if kernel == "gemv" and "rows" in params and "cols" in params:
        return f"{params['rows']}x{params['cols']}"
    if kernel == "gemm" and {"m", "n", "k"} <= params.keys():
        return f"M={params['m']}, N={params['n']}, K={params['k']}"
    if kernel == "conv" and {"cin", "h", "w", "k", "cout"} <= params.keys():
        return (f"Cin={params['cin']} HxW={params['h']}x{params['w']} "
                f"K={params['k']} Cout={params['cout']}")
    return ""


def plot_kernel_all_implementations(kernel: str,
                                    presets_1t: list, presets_6t: list,
                                    threads_modes: tuple[int, int],
                                    out_path: Path):
    """Big grid plot for one kernel — one subplot per preset, horizontal
    bars showing every implementation at both thread modes side by side."""
    presets_by_name_1t = {p["name"]: p for p in presets_1t}
    presets_by_name_6t = {p["name"]: p for p in presets_6t}
    order = [p["name"] for p in presets_1t] or [p["name"] for p in presets_6t]
    # Deduplicate while keeping insertion order, then add any 6T-only presets
    seen = set()
    preset_names = []
    for n in order + [p["name"] for p in presets_6t]:
        if n not in seen:
            seen.add(n)
            preset_names.append(n)
    if not preset_names:
        return

    n = len(preset_names)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig_height = max(3.4, 2.8 * nrows + 1.6)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIGWIDTH_BIG, fig_height),
                             squeeze=False)
    axes_flat = axes.flatten()

    t_st, t_mt = threads_modes
    color_st = PALETTE[0]
    color_mt = PALETTE[3]
    # Slightly thicker bars + a small gap between impl groups so the
    # row labels and value annotations have breathing room.
    h_bar = 0.40
    y_step = 1.15

    for idx, pname in enumerate(preset_names):
        ax = axes_flat[idx]
        p1 = presets_by_name_1t.get(pname, {"implementations": {}, "params": {}})
        p6 = presets_by_name_6t.get(pname, {"implementations": {}, "params": {}})
        params = p1.get("params") or p6.get("params") or {}

        impl_names = sorted(set(p1["implementations"]) | set(p6["implementations"]))

        def _key(name):
            m6 = (p6["implementations"].get(name) or {}).get("median", 0.0) or 0.0
            m1 = (p1["implementations"].get(name) or {}).get("median", 0.0) or 0.0
            return -(m6 if m6 > 0 else m1)
        impl_names.sort(key=_key)

        if not impl_names:
            ax.set_visible(False)
            continue

        y = np.arange(len(impl_names)) * y_step
        v1 = [(p1["implementations"].get(name) or {}).get("median", 0.0) or 0.0
              for name in impl_names]
        v6 = [(p6["implementations"].get(name) or {}).get("median", 0.0) or 0.0
              for name in impl_names]
        labels = [polish_label(name) for name in impl_names]

        ax.barh(y - h_bar / 2, v1, h_bar,
                label=f"{t_st} {_thread_word(t_st)}",
                color=color_st, edgecolor='black', linewidth=0.4)
        ax.barh(y + h_bar / 2, v6, h_bar,
                label=f"{t_mt} {_thread_word(t_mt)}",
                color=color_mt, edgecolor='black', linewidth=0.4)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel("GFLOPS", fontsize=8)

        params_str = _params_str(params, kernel)
        title = f"{pname}" if not params_str else f"{pname}  ({params_str})"
        ax.set_title(title, fontsize=9, pad=4)

        ax.grid(axis='x', linewidth=0.3, alpha=0.5)
        ax.grid(axis='y', visible=False)

        xmax = max(v1 + v6 + [0.0]) * 1.22
        if xmax > 0:
            ax.set_xlim(0, xmax)
        # Invert by setting ylim with the larger value first so row 0
        # (the best-performing impl) sits at the top with breathing room.
        ax.set_ylim((len(impl_names) - 1) * y_step + y_step * 0.6,
                    -y_step * 0.6)

        for yi, val in zip(y - h_bar / 2, v1):
            if val > 0:
                ax.annotate(f"{val:.1f}", xy=(val, yi),
                            xytext=(2, 0), textcoords='offset points',
                            va='center', fontsize=6, color=color_st)
        for yi, val in zip(y + h_bar / 2, v6):
            if val > 0:
                ax.annotate(f"{val:.1f}", xy=(val, yi),
                            xytext=(2, 0), textcoords='offset points',
                            va='center', fontsize=6, color=color_mt)

        if idx == 0:
            ax.legend(fontsize=7, loc='lower right', framealpha=0.9,
                      edgecolor='gray', fancybox=False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Porównanie wszystkich implementacji — {kernel.upper()}",
                 fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--results", default=str(here / "results"),
                    help="directory containing combined_<N>t.json files")
    ap.add_argument("--out", default=str(here / "figures"),
                    help="output directory for PDFs")
    args = ap.parse_args()

    res_dir = Path(args.results)
    out_dir = Path(args.out)
    if not res_dir.is_dir():
        sys.exit(f"results directory not found: {res_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(res_dir.glob("combined_*t.json"))
    if not json_paths:
        sys.exit(f"no combined_*t.json files in {res_dir} — run run_suite.py first")

    # Per-(kernel, thread) "Autorskie vs vendor" plots.
    by_threads = {}
    for json_path in json_paths:
        try:
            threads = int(json_path.stem.split("_")[1].rstrip("t"))
        except (IndexError, ValueError):
            continue
        with json_path.open() as f:
            data = json.load(f)
        by_threads[threads] = data

        for kernel, kernel_data in data.get("kernels", {}).items():
            rows = collect(kernel_data)
            suffix = "1t" if threads == 1 else f"{threads}t"
            out_path = out_dir / f"{kernel}_{suffix}.pdf"
            plot_kernel(kernel, rows, threads, out_path)

    # One big grid plot per kernel — every implementation, every preset.
    if len(by_threads) >= 2:
        thread_modes = sorted(by_threads.keys())
        st = thread_modes[0]
        mt = thread_modes[-1]
        kernels_seen = set()
        for data in by_threads.values():
            kernels_seen.update(data.get("kernels", {}).keys())
        for kernel in sorted(kernels_seen):
            p1 = by_threads[st]["kernels"].get(kernel, {}).get("presets", [])
            p6 = by_threads[mt]["kernels"].get(kernel, {}).get("presets", [])
            out_path = out_dir / f"{kernel}_all_implementations.pdf"
            plot_kernel_all_implementations(kernel, p1, p6, (st, mt), out_path)


if __name__ == "__main__":
    main()
