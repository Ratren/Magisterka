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

VENDOR_PATTERNS = ("OpenBLAS", "AOCL-BLAS")
NAIVE_PATTERNS = ("Naive", "naive")


# Klucze odpowiadaja nazwom implementacji ze strony C (benchmark.c); wartosci to
# czyste, opisowe nazwy uzywane w pracy magisterskiej (rozdz. 5). Klucze swoiste
# (dluzsze/jednoznaczne) wpisane przed ogolnymi (krotkimi, wspoldzielonymi przez
# kilka jader) -- polish_label woli dopasowanie dokladne, a przy podciagu wybiera
# najdluzszy pasujacy klucz, wiec np. "OMP Packed" nie jest lapane przez "OMP".
IMPL_LABELS = {
    # dot_product
    "SIMD MultiAcc":         "Wektoryzacja AVX2 (8 akumulatorów)",

    # gemv
    "SIMD + Prefetch":       "AVX2 z prefetchem programowym",
    "AVX+FMA Blocked":       "Jądro FMA z blokowaniem cache",
    "AVX+FMA V2":            "Jądro FMA, blokowanie 4-wierszowe",
    "AVX+FMA V3_OMP":        "Autorskie jądro AVX2/FMA — OpenMP",
    "AVX+FMA V3":            "Autorskie jądro AVX2/FMA (4 wiersze, 8 akum.)",

    # gemm
    "Loop Reorder ikj":      "Zmiana kolejności pętli (i-k-j)",
    "Loop Reorder":          "Zmiana kolejności pętli",
    "ST 6x8 packed":         "Pakowanie i mikrojądro 6×8",
    "MT 6x8 packed":         "Mikrojądro 6×8 + OpenMP",
    "ST 4x12 intrinsics":    "Mikrojądro 4×12 (funkcje wbudowane)",
    "ST 4x12 packed":        "Mikrojądro 4×12 (asembler)",
    "ST 4x12 tiny":          "Ścieżka dla małych macierzy",
    "MT 4x12 per-thread B":  "Wariant równoległy (osobny bufor B)",
    "MT 4x12 shared B":      "Wariant równoległy (wspólny bufor B)",
    "MT 4x12 best":          "Dyspozytor zależny od rozmiaru",
    "MT Strassen":           "Algorytm Strassena (7 mnożeń)",

    # conv
    "Packed Direct":         "Mikrojądro bezpośrednie z pakowaniem (6×16)",
    "OMP Packed":            "Mikrojądro bezpośrednie + OpenMP",
    "NCHWc direct":          "Układ blokowy NCHWc8",
    "1x1 (SGEMM)":           "Splot 1×1 jako SGEMM",
    "Winograd F(2,3)":       "Winograd F(2×2, 3×3)",
    "Zen3 dispatch OMP":     "Dyspozytor zależny od kształtu jądra",
    "im2col + OpenBLAS":     "im2col + OpenBLAS",
    "im2col + AOCL-BLAS":         "im2col + AOCL-BLAS",

    # klucze ogolne (krotkie, wspoldzielone przez kilka jader) -- na koncu
    "OMP":                   "Zrównoleglenie OpenMP",
    "SIMD":                  "Wektoryzacja AVX2",
    "Blocked":               "Blokowanie pamięci podręcznej",
    "Naive":                 "Implementacja naiwna",
}


def polish_label(name: str) -> str:
    """Translate a C-side impl name to its Polish descriptive label.
    Strips any trailing thread-count suffix like "(6T)" before matching.
    Prefers an exact match; otherwise falls back to the longest matching
    substring key, so a short generic key (e.g. "OMP") never captures a
    more specific name ("OMP Packed", "AVX+FMA V3_OMP"). Vendor names with
    no matching key are returned unchanged (with their suffix)."""
    stripped = name
    for sep in (" (", "("):
        idx = stripped.rfind(sep)
        if idx != -1 and ("T)" in stripped[idx:] or "T " in stripped[idx:]):
            stripped = stripped[:idx].rstrip()
            break
    if stripped in IMPL_LABELS:
        return IMPL_LABELS[stripped]
    best_key = None
    for key in IMPL_LABELS:
        if key in stripped and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return IMPL_LABELS[best_key] if best_key else name


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
    # Splot ma 12 ustawien -- pelne wymiary nie zmiescilyby sie czytelnie na osi
    # X (znajduja sie w tabeli ustawien w pracy), wiec dla conv zostaje sama
    # nazwa ustawienia.
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
    xtick_fs = 5 if kernel == "gemm" else 6
    ax.set_xticklabels(labels, fontsize=xtick_fs, rotation=0)
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
        # Etykieta wysrodkowana nad slupkiem autorskim moze siegac sasiedniego
        # slupka (OpenBLAS, tuz po prawej); gdy ten jest wyzszy, podnosimy
        # etykiete nad jego wierzcholek, by nie nachodzila na slupek.
        label_y = r["ours_median"]
        if ref and ref > label_y:
            label_y = ref
        ax.annotate(txt, xy=(x[i] + ours_offset, label_y),
                    ha='center', va='bottom', fontsize=5,
                    fontweight='bold', color=PALETTE[0])
    # Etykiet nad slupkami bibliotek nie rysujemy -- zgodnie z podpisami w pracy
    # liczby podaje sie tylko nad slupkiem autorskim (wyzej). Usuwa to nakladanie
    # sie etykiet sasiednich slupkow widoczne, gdy ich wartosci byly zblizone.

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


# ---------------------------------------------------------------------------
# Przyspieszenie kolejnych implementacji wzgledem wersji naiwnej (1 watek).
# Dla kazdego jadra jeden wykres: poziome slupki, jeden na implementacje
# autorska, dlugosc = srednia geometryczna ilorazu mediana_impl/mediana_naiwna
# po wszystkich presetach. Biblioteki vendorow pomijamy -- wykres ilustruje
# "wage" kolejnych krokow optymalizacji wlasnego kodu.
# ---------------------------------------------------------------------------

def _geomean(values: list) -> float | None:
    """Srednia geometryczna dodatnich wartosci (None, gdy brak danych)."""
    vals = [v for v in values if v and v > 0.0]
    if not vals:
        return None
    return float(np.exp(np.mean(np.log(vals))))


def plot_speedup(kernel: str, presets: list, threads: int, out_path: Path):
    """Przyspieszenie kolejnych implementacji autorskich wzgledem naiwnej.
    Slupek = srednia geometryczna ilorazu mediana_impl/mediana_naiwna po
    presetach. Implementacje w kolejnosci pojawiania sie w benchmarku (czyli
    w kolejnosci rozwoju kodu); naiwna na gorze jako odniesienie (1,0x)."""
    if not presets:
        return

    # Nazwy implementacji w kolejnosci pierwszego wystapienia (= kolejnosc
    # rozwoju kodu w benchmark.c); naiwna trafia na poczatek naturalnie.
    order = []
    seen = set()
    for p in presets:
        for name in p.get("implementations", {}):
            if name not in seen:
                seen.add(name)
                order.append(name)

    # Dla kazdej implementacji: iloraz wzgledem naiwnej w tym samym presecie,
    # usredniony geometrycznie po presetach.
    entries = []
    for name in order:
        if is_vendor(name):
            continue
        ratios = []
        for p in presets:
            impls = p.get("implementations", {})
            naive = next((v.get("median", 0.0) for n, v in impls.items()
                          if is_naive(n)), 0.0)
            cur = (impls.get(name) or {}).get("median", 0.0)
            if naive and naive > 0 and cur and cur > 0:
                ratios.append(cur / naive)
        g = _geomean(ratios)
        if g is None:
            continue
        label = "implementacja naiwna" if is_naive(name) else polish_label(name)
        entries.append((label, g, is_naive(name)))

    if len(entries) < 2:
        return

    n = len(entries)
    fig_h = max(2.6, 0.46 * n + 1.2)
    fig, ax = plt.subplots(figsize=(FIGWIDTH, fig_h))

    y = np.arange(n)
    speeds = [e[1] for e in entries]
    # Naiwna w kolorze neutralnym (szary), pozostale w kolorze autorskim.
    bar_colors = ["#9e9e9e" if e[2] else PALETTE[0] for e in entries]

    ax.barh(y, speeds, 0.66, color=bar_colors,
            edgecolor='black', linewidth=0.5)

    ax.axvline(x=1.0, color="#555555", linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in entries], fontsize=7.5)
    ax.invert_yaxis()  # naiwna na gorze, kolejne kroki nizej
    ax.set_xlabel("Przyspieszenie względem implementacji naiwnej [×]",
                  fontsize=8.5)
    title_thread = "1 wątek" if threads == 1 else f"{threads} {_thread_word(threads)}"
    ax.set_title(f"{kernel.upper()} — przyspieszenie kolejnych implementacji "
                 f"({title_thread})", fontsize=10)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='x', linewidth=0.3, alpha=0.6)
    ax.grid(axis='y', visible=False)

    xmax = max(speeds) * 1.16
    ax.set_xlim(0, xmax)

    for yi, val in zip(y, speeds):
        ax.annotate(f"{val:.1f}×", xy=(val, yi), xytext=(3, 0),
                    textcoords='offset points', va='center',
                    fontsize=7, fontweight='bold', color="#222222")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


# ---------------------------------------------------------------------------
# Model roofline (tylko jadra ograniczone pamiecia: dot, gemv).
# GEMM i splot pomijamy -- sa ograniczone moca obliczeniowa, wiec roofline
# nie wnosi tam nic poza punktem tuz pod pulapem szczytowym.
# ---------------------------------------------------------------------------

# Stale maszyny -- AMD Ryzen 5 5600 (Zen 3), pojedynczy rdzen.
# Szczyt: 2 jednostki FMA * 4 liczby double * 2 (mnozenie+dodawanie) * ~4,4 GHz.
ROOFLINE_PEAK_GFLOPS = 70.4
# Przepustowosc poszczegolnych poziomow pamieci [GB/s].
# L1d i L2 wyznaczone teoretycznie (2x256b/cykl = 64 B/cykl oraz 32 B/cykl,
# przy 4.4 GHz); L3 zmierzone empirycznie (narzedzie bandwidth); DRAM teoretyczne.
ROOFLINE_BW = {
    "L1d":  281.6,
    "L2":   140.8,
    "L3":   106.0,
    "DRAM":  51.2,
}
# Kolejnosc rysowania diagonali / pulapow od najwolniejszego do najszybszego.
ROOFLINE_LEVELS = ("DRAM", "L3", "L2", "L1d")
_CEILING_COLOR = {"L1d": "#9467bd", "L2": "#1f77b4", "L3": "#2ca02c", "DRAM": "#d62728"}

# Rozmiary pamieci podrecznej -- AMD Ryzen 5 5600 (Zen 3), na rdzen.
CACHE_BYTES = {
    "L1d":  32 * 1024,
    "L2":  512 * 1024,
    "L3":   32 * 1024 * 1024,
}
# Tinty pasow poziomow pamieci (jak w wykresie pasma w generatorze artykulu).
_BAND_TINT = {"L1d": "#d0e8ff", "L2": "#c8e6c8", "L3": "#fff3cc", "DRAM": "#ffdddd"}


def _human_bytes(b: float) -> str:
    """Rozmiar w jednostkach binarnych do etykiet osi (B / KiB / MiB / GiB)."""
    for unit, sz in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= sz:
            return f"{b / sz:g} {unit}"
    return f"{int(b)} B"


def _ai_dot(params: dict):
    """Intensywnosc operacyjna iloczynu skalarnego (double, 8 B/element):
    2N dzialan / 8*(2N+1) bajtow (dwa wektory + zapis wyniku)."""
    n = params.get("size")
    if not n:
        return None
    return 2.0 * n / (8.0 * (2.0 * n + 1.0))


def _ai_gemv(params: dict):
    """Intensywnosc operacyjna GEMV (double): 2MN dzialan /
    8*(MN + N + 2M) bajtow (macierz A, wektor x, odczyt i zapis y)."""
    m, n = params.get("rows"), params.get("cols")
    if not m or not n:
        return None
    return 2.0 * m * n / (8.0 * (m * n + n + 2.0 * m))


# kernel -> (funkcja OI, nominalna OI dla pulapow poziomych, etykieta tytulu,
#            zbior nazw presetow do pominiecia, czy pokazac AOCL-BLAS)
# tall/wide GEMV pomijamy -- maja niemal te sama OI co medium/large i tylko
# zageszczaja klaster punktow, nie wnoszac nowej informacji. Dla iloczynu
# skalarnego pomijamy AOCL-BLAS: wszystkie trzy serie pokrywaja sie niemal
# idealnie (operacja ograniczona pasmem), wiec trzeci znacznik tylko zaslania
# pozostale.
ROOFLINE_KERNELS = {
    "dot":  (_ai_dot,  0.125, "iloczyn skalarny", set(),            False),
    "gemv": (_ai_gemv, 0.25,  "GEMV",             {"tall", "wide"}, True),
}


def _place_labels(ax, items, fontsize=6, color="#333333", italic=True,
                  extra_markers=()):
    """Rozmieszcza etykiety punktow tak, by nie nachodzily ani na siebie, ani
    na znaczniki serii. Dla kazdej etykiety wybiera pierwsze z kandydujacych
    przesuniec (gora/dol/boki/dalej), ktore nie koliduje z juz zajetymi
    obszarami; przy wiekszym odsunieciu dodaje cienka linie odniesienia.
    Kolizje liczone w pikselach (transData) z przyblizonym rozmiarem tekstu --
    wystarczajaco dokladnie dla kilku-kilkunastu etykiet. Wymaga ustalonych
    wczesniej granic osi."""
    if not items:
        return
    fig = ax.figure
    ppp = fig.dpi / 72.0
    char_w = 0.60 * fontsize * ppp
    line_h = 1.30 * fontsize * ppp

    def make_box(cx, cy, w, h, ha, va):
        x0 = cx - w / 2.0 if ha == "center" else (cx if ha == "left" else cx - w)
        y0 = cy if va == "bottom" else (cy - h if va == "top" else cy - h / 2.0)
        return [x0, y0, x0 + w, y0 + h]

    def hits(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    placed = []
    for it in items:
        mx, my = ax.transData.transform((it["x"], it["y"]))
        placed.append([mx - 3 * ppp, my - 3 * ppp, mx + 3 * ppp, my + 3 * ppp])
    for mx_, my_ in extra_markers:
        mx, my = ax.transData.transform((mx_, my_))
        placed.append([mx - 3 * ppp, my - 3 * ppp, mx + 3 * ppp, my + 3 * ppp])

    cands = [(0, 6, "center", "bottom"), (0, -6, "center", "top"),
             (6, 2, "left", "bottom"), (-6, 2, "right", "bottom"),
             (6, -2, "left", "top"), (-6, -2, "right", "top"),
             (0, 16, "center", "bottom"), (0, -16, "center", "top"),
             (0, 26, "center", "bottom"), (0, -26, "center", "top")]
    for it in sorted(items, key=lambda d: (d["x"], -d["y"])):
        px, py = ax.transData.transform((it["x"], it["y"]))
        w = max(1, len(it["label"])) * char_w
        chosen = None
        for dx, dy, ha, va in cands:
            b = make_box(px + dx * ppp, py + dy * ppp, w, line_h, ha, va)
            if not any(hits(b, p) for p in placed):
                chosen = (dx, dy, ha, va, b)
                break
        if chosen is None:
            dx, dy, ha, va = 0, 32, "center", "bottom"
            b = make_box(px, py + dy * ppp, w, line_h, ha, va)
            chosen = (dx, dy, ha, va, b)
        dx, dy, ha, va, b = chosen
        placed.append(b)
        far = abs(dx) >= 6 or abs(dy) >= 14
        ax.annotate(it["label"], xy=(it["x"], it["y"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=fontsize, color=color,
                    fontstyle="italic" if italic else "normal",
                    arrowprops=(dict(arrowstyle="-", color="gray", lw=0.4,
                                     alpha=0.5) if far else None))


def plot_roofline(kernel: str, rows: list[dict], out_path: Path):
    """Model roofline dla jadra ograniczonego pamiecia: punkty (OI, GFLOPS)
    najlepszej autorskiej implementacji oraz OpenBLAS na tle diagonali
    przepustowosci kolejnych poziomow pamieci i pulapu szczytowego."""
    spec = ROOFLINE_KERNELS.get(kernel)
    if spec is None or not rows:
        return
    ai_fn, ai_nominal, kernel_label, exclude, show_aocl = spec

    data = []
    for r in rows:
        ai = ai_fn(r.get("params", {}))
        if ai is None or not r.get("ours_median"):
            continue
        base = r["preset"].split("_", 1)[1] if "_" in r["preset"] else r["preset"]
        if base in exclude:
            continue
        blas = next((v[1] for v in r["vendors"] if v[0] == "OpenBLAS"), None)
        aocl = (next((v[1] for v in r["vendors"] if v[0] == "AOCL-BLAS"), None)
                if show_aocl else None)
        data.append({"ai": ai, "ours": r["ours_median"], "blas": blas,
                     "aocl": aocl, "label": base})
    if not data:
        return

    from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator

    ais = [d["ai"] for d in data]
    gflops = ([d["ours"] for d in data] + [d["blas"] for d in data if d["blas"]]
              + [d["aocl"] for d in data if d["aocl"]])
    ai_lo = min(ais) * 0.3
    ai_hi = max(ais) * 6.0
    y_lo = max(1.0, min(gflops) * 0.6)
    y_hi = ROOFLINE_PEAK_GFLOPS * 1.07

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ai_range = np.logspace(np.log10(ai_lo), np.log10(ai_hi), 500)

    # Diagonale przepustowosci -- tlo kontekstowe.
    diag_ls = {"DRAM": "-", "L3": "--", "L2": "-.", "L1d": ":"}
    for level in ROOFLINE_LEVELS:
        bw = ROOFLINE_BW[level]
        perf = np.minimum(ROOFLINE_PEAK_GFLOPS, bw * ai_range)
        ax.loglog(ai_range, perf, diag_ls[level], color="#bbbbbb",
                  alpha=0.6, linewidth=0.8)
        # Etykieta na samej diagonali, w obszarze jeszcze nienasyconym.
        ly = {"L1d": y_hi * 0.50, "L2": y_hi * 0.36,
              "L3": y_hi * 0.26, "DRAM": y_hi * 0.19}[level]
        lx = ly / bw
        if ai_lo < lx < ai_hi and ly < ROOFLINE_PEAK_GFLOPS:
            ax.text(lx, ly * 0.9, level, fontsize=6, color="#999999",
                    ha="center", va="top", rotation=28)

    # Pulap szczytowy.
    ax.axhline(y=ROOFLINE_PEAK_GFLOPS, color="darkred", linewidth=1.2, alpha=0.5)
    ax.text(ai_lo * 1.05, ROOFLINE_PEAK_GFLOPS, f"Szczyt {ROOFLINE_PEAK_GFLOPS:g}",
            fontsize=6, color="darkred", alpha=0.75, va="bottom", ha="left")

    # Poziome pulapy jadra (BW * OI_nominalna); pomijamy te bliskie szczytowi.
    for level in ("L1d", "L2", "L3", "DRAM"):
        ceil = ROOFLINE_BW[level] * ai_nominal
        # Pomijamy pulap zlewajacy sie ze szczytem (np. L1d dla GEMV ~ 66,8).
        if ceil >= ROOFLINE_PEAK_GFLOPS * 0.90 or not (y_lo < ceil < y_hi):
            continue
        c = _CEILING_COLOR[level]
        ax.axhline(y=ceil, color=c, linewidth=1.0, alpha=0.55, dashes=(4, 3))
        ax.text(ai_hi * 0.97, ceil * 1.03, f"pulap {level} ({ceil:.1f})",
                fontsize=6, color=c, alpha=0.85, va="bottom", ha="right")

    # Pionowa linia odniesienia przy nominalnej OI.
    ax.axvline(x=ai_nominal, color="gray", linewidth=0.5, alpha=0.3, linestyle=":")

    # Punkty pomiarowe.
    ax.plot(ais, [d["ours"] for d in data], "o", markersize=5,
            color=PALETTE[0], markeredgecolor="black", markeredgewidth=0.5,
            zorder=10, label="Autorskie")
    blas_xy = [(d["ai"], d["blas"]) for d in data if d["blas"]]
    if blas_xy:
        ax.plot([a for a, _ in blas_xy], [b for _, b in blas_xy], "^",
                markersize=5, color=PALETTE[1], markeredgecolor="black",
                markeredgewidth=0.5, zorder=10, label="OpenBLAS")
    aocl_xy = [(d["ai"], d["aocl"]) for d in data if d["aocl"]]
    if aocl_xy:
        ax.plot([a for a, _ in aocl_xy], [v for _, v in aocl_xy], "s",
                markersize=5, color=PALETTE[2], markeredgecolor="black",
                markeredgewidth=0.5, zorder=10, label="AOCL-BLAS")

    # Etykiety ustawien rozmieszcza _place_labels (po ustaleniu granic osi),
    # tak by nie nachodzily na siebie ani na znaczniki.

    ax.set_xlim(ai_lo, ai_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Intensywność operacyjna [FLOP/bajt]", fontsize=9)
    ax.set_ylabel("Wydajność [GFLOPS]", fontsize=9)
    ax.set_title(f"Model roofline: {kernel_label} — Ryzen 5 5600 (1 rdzeń)",
                 fontsize=11)

    cand_x = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
    cand_y = [2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]
    xticks = [t for t in cand_x if ai_lo <= t <= ai_hi]
    yticks = [t for t in cand_y if y_lo <= t <= y_hi]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    def _fmt(val, _pos):
        return f"{int(val)}" if abs(val - round(val)) < 1e-9 else f"{val:g}"
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9,
              edgecolor="gray", fancybox=False)

    plt.tight_layout()
    _place_labels(ax,
                  [{"x": d["ai"], "y": d["ours"], "label": d["label"]} for d in data],
                  extra_markers=([(d["ai"], d["blas"]) for d in data if d["blas"]]
                                 + aocl_xy))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


# ---------------------------------------------------------------------------
# Wydajnosc a rozmiar danych ("klif cache") -- dot, gemv, gemm.
# Pokazuje, jak wydajnosc zmienia sie, gdy zbior roboczy przekracza pojemnosc
# kolejnych poziomow pamieci podrecznej. Uzupelnia roofline (tam OI jest stala,
# osi rozmiaru brak).
# ---------------------------------------------------------------------------

def _bytes_dot(params: dict):
    """Rozmiar zbioru roboczego iloczynu skalarnego: dwa wektory + wynik."""
    n = params.get("size")
    return (2.0 * n + 1.0) * 8.0 if n else None


def _bytes_gemv(params: dict):
    """Rozmiar zbioru roboczego GEMV: macierz A + wektory x, y (dominuje A)."""
    m, n = params.get("rows"), params.get("cols")
    return (m * n + n + m) * 8.0 if (m and n) else None


def _bytes_gemm(params: dict):
    """Rozmiar zbioru roboczego GEMM: trzy macierze A, B, C."""
    m, n, k = params.get("m"), params.get("n"), params.get("k")
    return (m * k + k * n + m * n) * 8.0 if (m and n and k) else None


# kernel -> (funkcja rozmiaru [bajty], etykieta jadra, czy rysowac linie szczytu,
#            zbior nazw ustawien do pominiecia). Pomijamy ustawienia o skrajnych
#            proporcjach, bo o ich wydajnosci decyduje ksztalt macierzy, a nie
#            rozmiar zbioru roboczego -- na krzywej "wydajnosc vs rozmiar"
#            tworzylyby mylacy dol: dla GEMM rank_k i tall_K (~36 MiB, tuz za L3),
#            dla GEMV wide i tall (analogicznie jak na wykresie roofline).
SIZE_SWEEP_KERNELS = {
    "dot":  (_bytes_dot,  "iloczyn skalarny", False, set()),
    "gemv": (_bytes_gemv, "GEMV",             False, {"tall", "wide"}),
    "gemm": (_bytes_gemm, "GEMM",             True,  {"rank_k", "tall_K"}),
}


def plot_size_sweep(kernel: str, rows: list[dict], out_path: Path):
    """Wydajnosc (GFLOPS) w funkcji rozmiaru danych, z pasami poziomow pamieci.
    Punkty: najlepsza autorska implementacja oraz OpenBLAS (dane 1T)."""
    spec = SIZE_SWEEP_KERNELS.get(kernel)
    if spec is None or not rows:
        return
    bytes_fn, kernel_label, show_peak, exclude = spec

    data = []
    for r in rows:
        nb = bytes_fn(r.get("params", {}))
        if nb is None or not r.get("ours_median"):
            continue
        base = r["preset"].split("_", 1)[1] if "_" in r["preset"] else r["preset"]
        if base in exclude:
            continue
        blas = next((v[1] for v in r["vendors"] if v[0] == "OpenBLAS"), None)
        aocl = next((v[1] for v in r["vendors"] if v[0] == "AOCL-BLAS"), None)
        data.append({"bytes": nb, "ours": r["ours_median"], "blas": blas,
                     "aocl": aocl, "label": base})
    if not data:
        return
    data.sort(key=lambda d: d["bytes"])

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    xs = [d["bytes"] for d in data]
    ours = [d["ours"] for d in data]
    blas_xy = [(d["bytes"], d["blas"]) for d in data if d["blas"]]
    aocl_xy = [(d["bytes"], d["aocl"]) for d in data if d["aocl"]]
    gflops = ours + [b for _, b in blas_xy] + [a for _, a in aocl_xy]

    x_lo = min(xs) / 3.0
    x_hi = max(xs) * 3.0
    if show_peak:
        y_top = max(ROOFLINE_PEAK_GFLOPS * 1.07, max(gflops) * 1.1)
    else:
        y_top = max(gflops) * 1.25

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.set_xscale("log")

    # Pasy poziomow pamieci wg rozmiaru zbioru roboczego.
    bands = [
        (0.0,                 CACHE_BYTES["L1d"], "L1d"),
        (CACHE_BYTES["L1d"],  CACHE_BYTES["L2"],  "L2"),
        (CACHE_BYTES["L2"],   CACHE_BYTES["L3"],  "L3"),
        (CACHE_BYTES["L3"],   x_hi,               "DRAM"),
    ]
    for lo, hi, label in bands:
        lo_c, hi_c = max(lo, x_lo), min(hi, x_hi)
        if hi_c <= lo_c:
            continue
        ax.axvspan(lo_c, hi_c, color=_BAND_TINT[label], alpha=0.35, zorder=0)
        xc = (lo_c * hi_c) ** 0.5  # srodek geometryczny (os log)
        ax.text(xc, y_top * 0.97, label, fontsize=7, color="#555555",
                ha="center", va="top", fontstyle="italic")

    # Linia szczytu -- tylko dla jader compute-bound (GEMM).
    if show_peak:
        ax.axhline(y=ROOFLINE_PEAK_GFLOPS, color="darkred", linewidth=1.0, alpha=0.5)
        ax.text(x_lo * 1.1, ROOFLINE_PEAK_GFLOPS, f"Szczyt {ROOFLINE_PEAK_GFLOPS:g}",
                fontsize=6, color="darkred", alpha=0.75, va="bottom", ha="left")

    # Serie pomiarowe.
    ax.plot(xs, ours, "-o", markersize=5, color=PALETTE[0],
            markeredgecolor="black", markeredgewidth=0.5, linewidth=1.2,
            zorder=10, label="Autorskie")
    if blas_xy:
        ax.plot([x for x, _ in blas_xy], [b for _, b in blas_xy], "-^",
                markersize=5, color=PALETTE[1], markeredgecolor="black",
                markeredgewidth=0.5, linewidth=1.2, zorder=10, label="OpenBLAS")
    if aocl_xy:
        ax.plot([x for x, _ in aocl_xy], [a for _, a in aocl_xy], "-s",
                markersize=5, color=PALETTE[2], markeredgecolor="black",
                markeredgewidth=0.5, linewidth=1.2, zorder=10, label="AOCL-BLAS")

    # Etykiety ustawien rozmieszcza _place_labels (po ustaleniu granic osi),
    # tak by nie nachodzily na siebie ani na znaczniki -- istotne tam, gdzie
    # ustawienia maja zblizony rozmiar zbioru roboczego (np. wide/tall w GEMV).

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Rozmiar zbioru roboczego [bajty]", fontsize=9)
    ax.set_ylabel("Wydajność [GFLOPS]", fontsize=9)
    ax.set_title(f"Wydajność a rozmiar danych: {kernel_label} "
                 f"— Ryzen 5 5600 (1 rdzeń)", fontsize=11)

    # Os X: znaczniki na granicach pojemnosci cache.
    ax.xaxis.set_major_locator(FixedLocator(list(CACHE_BYTES.values())))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _human_bytes(v)))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.6)
    ax.grid(axis="x", visible=False)
    # Dolny-lewy rog jest pusty we wszystkich trzech wariantach (krzywe malejace
    # dla dot/gemv, plateau wysoko dla gemm), wiec tam nie koliduje z danymi.
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9,
              edgecolor="gray", fancybox=False)

    plt.tight_layout()
    _place_labels(ax,
                  [{"x": d["bytes"], "y": d["ours"], "label": d["label"]} for d in data],
                  extra_markers=[(bx, b) for bx, b in blas_xy] + aocl_xy)
    fig.savefig(out_path, bbox_inches="tight")
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

    # Modele roofline -- tylko jadra ograniczone pamiecia (dot, gemv), dane 1T,
    # poniewaz pulap szczytowy i przepustowosci sa wartosciami jednordzeniowymi.
    data_1t = by_threads.get(1)
    if data_1t:
        # Przyspieszenie kolejnych implementacji wzgledem naiwnej (1 watek),
        # po jednym wykresie na jadro (dot, gemv, gemm, conv).
        for kernel, kernel_data in data_1t.get("kernels", {}).items():
            plot_speedup(kernel, kernel_data.get("presets", []), 1,
                         out_dir / f"speedup_{kernel}.pdf")

        for kernel in ROOFLINE_KERNELS:
            kernel_data = data_1t.get("kernels", {}).get(kernel)
            if not kernel_data:
                continue
            plot_roofline(kernel, collect(kernel_data),
                          out_dir / f"roofline_{kernel}.pdf")

        # Wydajnosc a rozmiar danych (klif cache) -- dot, gemv, gemm.
        for kernel in SIZE_SWEEP_KERNELS:
            kernel_data = data_1t.get("kernels", {}).get(kernel)
            if not kernel_data:
                continue
            plot_size_sweep(kernel, collect(kernel_data),
                            out_dir / f"size_{kernel}.pdf")


if __name__ == "__main__":
    main()
