#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE = Path(__file__).resolve().parent

KERNEL_BIN_CANDIDATES = {
    "dot":  ["dot_product/build/benchmark", "build/dot_product/benchmark"],
    "gemv": ["gemv/build/benchmark",        "build/gemv/benchmark"],
    "gemm": ["gemm/build/benchmark",        "build/gemm/benchmark"],
    "conv": ["conv/build/benchmark",        "build/conv/benchmark"],
}


def find_binary(kernel: str) -> Path | None:
    for rel in KERNEL_BIN_CANDIDATES.get(kernel, []):
        p = ROOT / rel
        if p.exists() and os.access(p, os.X_OK):
            return p
    return None

SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
KV_RE      = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*=\s*(.*?)\s*$")


def parse_preset(path: Path):
    sections = []
    cur = None
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            m = SECTION_RE.match(line)
            if m:
                cur = {"_name": m.group(1)}
                sections.append(cur)
                continue
            m = KV_RE.match(line)
            if m and cur is not None:
                cur[m.group(1)] = m.group(2)
    return sections


def detect_physical_cores() -> int:
    try:
        out = subprocess.check_output(["lscpu", "-p=CORE"], text=True)
        cores = {ln for ln in out.splitlines() if ln and not ln.startswith("#")}
        if cores:
            return len(cores)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return os.cpu_count() or 1


def run_kernel(kernel: str, preset_file: Path, json_out: Path,
               threads: int, pcores: int, log_path: Path) -> bool:
    bin_path = find_binary(kernel)
    if bin_path is None:
        print(f"!! binary for kernel '{kernel}' not found in any of: "
              f"{KERNEL_BIN_CANDIDATES.get(kernel)}", file=sys.stderr)
        return False

    if threads == 1:
        taskset = ["taskset", "-c", "0"]
    else:
        taskset = ["taskset", "-c", f"0-{pcores - 1}"]
    cmd = taskset + [str(bin_path),
                     "--preset-file", str(preset_file),
                     "--json", str(json_out)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    extra_lib_dirs = []
    for sub in ("blis_install/lib",):
        d = ROOT / sub
        if d.is_dir():
            extra_lib_dirs.append(str(d))
    if extra_lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        joined = ":".join(extra_lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{joined}:{existing}" if existing else joined

    print(f"   $ OMP_NUM_THREADS={threads} {' '.join(cmd)}")
    with log_path.open("w") as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"!! kernel '{kernel}' returned {proc.returncode} (see {log_path})", file=sys.stderr)
        return False
    return True


def merge_kernel_json(per_kernel_files: dict, out_path: Path, threads: int):
    combined = {"threads": threads, "kernels": {}}
    for kernel, path in per_kernel_files.items():
        if not path.exists():
            continue
        with path.open() as f:
            combined["kernels"][kernel] = json.load(f)
    with out_path.open("w") as f:
        json.dump(combined, f, indent=2)
    return combined


def print_summary(combined: dict):
    print()
    print("=" * 96)
    print(f"  SUMMARY — Median GFLOPS ({combined['threads']} thread(s))")
    print("=" * 96)
    for kernel, payload in combined["kernels"].items():
        for preset in payload["presets"]:
            params = preset.get("params", {})
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"\n  [{kernel}] {preset['name']}  ({param_str})")
            for name, vals in preset["implementations"].items():
                print(f"      {name:<22s} {vals['median']:8.2f}   "
                      f"(min {vals['min']:.2f}, max {vals['max']:.2f}, sd {vals['stddev']:.2f})")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description="Run the multi-kernel benchmark suite.")
    ap.add_argument("--preset", default=str(SUITE / "presets" / "default.preset"),
                    help="path to .preset file (INI)")
    ap.add_argument("--threads", default=None,
                    help="comma-separated list, e.g. '1,6'; default = '1,<physical-cores>'")
    ap.add_argument("--kernels", default=None,
                    help="comma-separated subset (e.g. 'dot,gemv'); default = all referenced in preset")
    ap.add_argument("--out", default=str(SUITE / "results"),
                    help="output directory for JSON results and logs")
    args = ap.parse_args()

    preset_path = Path(args.preset).resolve()
    if not preset_path.exists():
        sys.exit(f"preset file not found: {preset_path}")

    sections = parse_preset(preset_path)
    if not sections:
        sys.exit(f"no sections in {preset_path}")

    kernels_in_preset = sorted({s.get("kernel") for s in sections if s.get("kernel")})
    if args.kernels:
        wanted = set(args.kernels.split(","))
        kernels = [k for k in kernels_in_preset if k in wanted]
    else:
        kernels = kernels_in_preset
    if not kernels:
        sys.exit("no matching kernels to run")

    pcores = detect_physical_cores()
    if args.threads:
        thread_modes = [int(t) for t in args.threads.split(",")]
    else:
        thread_modes = [1, pcores]

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preset:           {preset_path}")
    print(f"Kernels:          {', '.join(kernels)}")
    print(f"Physical cores:   {pcores}")
    print(f"Thread modes:     {thread_modes}")
    print(f"Output directory: {out_dir}")

    for threads in thread_modes:
        print()
        print(f"########  {threads}-THREAD RUN  ########")
        per_kernel_files = {}
        for kernel in kernels:
            print(f"\n--- kernel: {kernel} ---")
            json_out = out_dir / f"results_{kernel}_{threads}t.json"
            log_path = out_dir / f"log_{kernel}_{threads}t.txt"
            if run_kernel(kernel, preset_path, json_out, threads, pcores, log_path):
                per_kernel_files[kernel] = json_out
        combined_path = out_dir / f"combined_{threads}t.json"
        combined = merge_kernel_json(per_kernel_files, combined_path, threads)
        print(f"\nCombined results: {combined_path}")
        print_summary(combined)


if __name__ == "__main__":
    main()
