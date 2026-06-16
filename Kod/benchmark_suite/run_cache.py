#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE = Path(__file__).resolve().parent

NOT_VALUES = ("<not supported>", "<not counted>", "")


def parse_perf_stat(text):
    counts = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(",")
        if len(parts) < 3 or not parts[2]:
            continue
        event = parts[2]
        raw = parts[0]
        if raw in NOT_VALUES:
            counts[event] = None
            continue
        try:
            counts[event] = float(raw)
        except ValueError:
            continue
    return counts


def get_count(counts, event):
    for key in (event, event + ":u", event.split(":")[0]):
        if key in counts:
            return counts[key]
    return None


def median(values):
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    m = len(xs)
    if m % 2:
        return xs[m // 2]
    return 0.5 * (xs[m // 2 - 1] + xs[m // 2])


def per_iteration(count_2n, count_n, n):
    if count_2n is None or count_n is None or n <= 0:
        return None
    diff = count_2n - count_n
    if diff < 0:
        diff = 0.0
    return diff / n


def compute_metrics(per_iter):
    out = {}
    loads = per_iter.get("l1_loads")
    miss = per_iter.get("l1_load_misses")
    out["l1_miss_rate"] = (miss / loads) if (loads is not None and miss is not None and loads > 0) else None
    fl2 = per_iter.get("fill_l2")
    fl3 = per_iter.get("fill_l3")
    fdram = per_iter.get("fill_dram")
    present = [v for v in (fl2, fl3, fdram) if v is not None]
    total = sum(present) if present else 0.0
    for key, val in (("pct_l2", fl2), ("pct_l3", fl3), ("pct_dram", fdram)):
        out[key] = (100.0 * val / total) if (val is not None and total > 0) else None
    out["fills_per_iter"] = total if present else None
    instr = per_iter.get("instructions")
    cyc = per_iter.get("cycles")
    out["ipc"] = (instr / cyc) if (instr is not None and cyc is not None and cyc > 0) else None
    return out


MAX_EVENTS_PER_GROUP = 5

EVENT_CANDIDATES = {
    "l1_loads":        ["l1-dcache-loads", "L1-dcache-loads"],
    "l1_load_misses":  ["l1-dcache-load-misses", "L1-dcache-load-misses"],
    "fill_l2":         ["ls_any_fills_from_sys.lcl_l2",
                        "ls_dmnd_fills_from_sys.lcl_l2"],
    "fill_l3":         ["ls_any_fills_from_sys.int_cache",
                        "ls_dmnd_fills_from_sys.int_cache"],
    "fill_dram":       ["ls_any_fills_from_sys.mem_io_local",
                        "ls_dmnd_fills_from_sys.mem_io_local"],
    "instructions":    ["instructions"],
    "cycles":          ["cycles"],
}


def event_supported(perf_bin, spec):
    p = subprocess.run([perf_bin, "stat", "-x,", "-e", spec, "--", "true"],
                       capture_output=True, text=True)
    counts = parse_perf_stat(p.stderr)
    return get_count(counts, spec) is not None


def resolve_events(perf_bin):
    resolved = {}
    for logical, specs in EVENT_CANDIDATES.items():
        for spec in specs:
            if event_supported(perf_bin, spec):
                resolved[logical] = spec
                break
    return resolved


def chunk_events(resolved, max_per_group=MAX_EVENTS_PER_GROUP):
    items = list(resolved.items())
    return [items[i:i + max_per_group] for i in range(0, len(items), max_per_group)]


def run_perf_once(perf_bin, event_arg, binary, kernel_args, env):
    cmd = [perf_bin, "stat", "-x,", "-e", event_arg, "--",
           "taskset", "-c", "0", str(binary)] + kernel_args
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return parse_perf_stat(p.stderr)


def time_run(binary, kernel_args, env):
    t0 = time.perf_counter()
    subprocess.run([str(binary)] + kernel_args, env=env,
                   capture_output=True, text=True)
    return time.perf_counter() - t0


def calibrate_n(binary, dims_fn, env, target_sec, min_iters, max_iters):
    n_cal = 16
    t = time_run(binary, dims_fn(n_cal), env)
    while t < 0.05 and n_cal < max_iters:
        n_cal *= 4
        t = time_run(binary, dims_fn(n_cal), env)
    per_iter = t / n_cal
    n = int(target_sec / per_iter) if per_iter > 0 else max_iters
    return max(min_iters, min(n, max_iters))


def measure_triple(perf_bin, resolved, binary, dims_fn, n, reps, env):
    per_iter = {logical: None for logical in resolved}
    for group in chunk_events(resolved):
        event_arg = ",".join(spec + ":u" for _, spec in group)
        acc_n = {logical: [] for logical, _ in group}
        acc_2n = {logical: [] for logical, _ in group}
        for _ in range(reps):
            cn = run_perf_once(perf_bin, event_arg, binary, dims_fn(n), env)
            c2 = run_perf_once(perf_bin, event_arg, binary, dims_fn(2 * n), env)
            for logical, spec in group:
                acc_n[logical].append(get_count(cn, spec))
                acc_2n[logical].append(get_count(c2, spec))
        for logical, _ in group:
            per_iter[logical] = per_iteration(median(acc_2n[logical]),
                                              median(acc_n[logical]), n)
    return per_iter


sys.path.insert(0, str(SUITE))
from run_suite import (KERNEL_BIN_CANDIDATES, find_binary, parse_preset)


def make_env():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    extra = []
    d = ROOT / "blis_install" / "lib"
    if d.is_dir():
        extra.append(str(d))
    if extra:
        existing = env.get("LD_LIBRARY_PATH", "")
        joined = ":".join(extra)
        env["LD_LIBRARY_PATH"] = f"{joined}:{existing}" if existing else joined
    return env


def dims_builder(kernel, section):
    def gi(key):
        return int(section[key])
    if kernel == "dot":
        size = gi("size")
        return lambda n, impl: ["--custom", str(n), str(size), "--measure", impl], \
               {"size": size}
    if kernel == "gemv":
        rows, cols = gi("rows"), gi("cols")
        return lambda n, impl: ["--custom", str(n), str(rows), str(cols),
                                "--measure", impl], {"rows": rows, "cols": cols}
    if kernel == "gemm":
        M, N, K = gi("m"), gi("n"), gi("k")
        return lambda n, impl: ["--custom", str(n), str(M), str(N), str(K),
                                "--measure", impl], {"m": M, "n": N, "k": K}
    if kernel == "conv":
        cin, h, w, k, cout = gi("cin"), gi("h"), gi("w"), gi("k"), gi("cout")
        return lambda n, impl: ["--custom", str(n), str(cin), str(h), str(w),
                                str(k), str(cout), "--measure", impl], \
               {"cin": cin, "h": h, "w": w, "k": k, "cout": cout}
    raise ValueError(f"nieznane jadro: {kernel}")


def list_impls(binary, env):
    p = subprocess.run([str(binary), "--list-impls"], env=env,
                       capture_output=True, text=True)
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def measure_kernel(perf_bin, kernel, sections, resolved, reps,
                   target_sec, min_iters, max_iters, env):
    binary = find_binary(kernel)
    if binary is None:
        print(f"!! brak binarki dla '{kernel}'", file=sys.stderr)
        return None
    impls = list_impls(binary, env)
    out = {"kernel": kernel,
           "events": resolved,
           "presets": []}
    for sec in sections:
        if sec.get("kernel") != kernel:
            continue
        dims_fn_full, params = dims_builder(kernel, sec)
        preset_block = {"name": sec["_name"], "params": params,
                        "implementations": {}}
        for impl in impls:
            args1 = dims_fn_full(1, impl)
            probe = subprocess.run([str(binary)] + args1, env=env,
                                   capture_output=True, text=True)
            if probe.returncode != 0:
                preset_block["implementations"][impl] = {
                    "skipped": True, "returncode": probe.returncode}
                continue
            dims_fn = lambda nn, _impl=impl: dims_fn_full(nn, _impl)
            n = calibrate_n(binary, dims_fn, env, target_sec, min_iters, max_iters)
            per_iter = measure_triple(perf_bin, resolved, binary, dims_fn, n, reps, env)
            metrics = compute_metrics(per_iter)
            metrics["raw_per_iter"] = per_iter
            metrics["skipped"] = False
            metrics["n"] = n
            preset_block["implementations"][impl] = metrics
        out["presets"].append(preset_block)
    return out


def print_table(payload):
    def f(x, w, p):
        return (f"{x*100:>{w}.{p}f}" if x is not None else f"{'N/A':>{w}s}")
    def g(x, w, p):
        return (f"{x:>{w}.{p}f}" if x is not None else f"{'N/A':>{w}s}")
    print("\n" + "=" * 120)
    print(f"  CACHE — {payload['kernel']}  (zdarzenia: "
          f"{', '.join(payload['events'].keys())})")
    print("=" * 120)
    for pr in payload["presets"]:
        ps = ", ".join(f"{k}={v}" for k, v in pr["params"].items())
        print(f"\n  [{pr['name']}] {ps}")
        print(f"      {'Implementacja':<26s} {'N':>8s} {'L1 miss%':>9s} {'%L2':>7s} "
              f"{'%L3':>7s} {'%DRAM':>7s} {'fills/it':>10s} {'IPC':>6s}")
        for name, m in pr["implementations"].items():
            if m.get("skipped"):
                print(f"      {name:<26s} {'':>8s} {'POMINIETO':>9s}")
                continue
            n_str = f"{m['n']:>8d}" if m.get("n") is not None else f"{'N/A':>8s}"
            print(f"      {name:<26s} {n_str} {f(m['l1_miss_rate'],9,2)} "
                  f"{g(m['pct_l2'],7,1)} {g(m['pct_l3'],7,1)} {g(m['pct_dram'],7,1)} "
                  f"{g(m['fills_per_iter'],10,1)} {g(m['ipc'],6,2)}")
    print("\n" + "=" * 120)


def main():
    ap = argparse.ArgumentParser(description="Pomiar cache per implementacja (perf stat).")
    ap.add_argument("--preset", default=str(SUITE / "presets" / "default.preset"))
    ap.add_argument("--kernels", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--target-sec", type=float, default=0.25)
    ap.add_argument("--min-iters", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--perf", default="perf")
    ap.add_argument("--out", default=str(SUITE / "results"))
    args = ap.parse_args()

    if subprocess.run([args.perf, "--version"], capture_output=True).returncode != 0:
        sys.exit("perf niedostepny — zainstaluj (sudo pacman -S perf)")

    preset_path = Path(args.preset).resolve()
    if not preset_path.exists():
        sys.exit(f"preset file not found: {preset_path}")
    sections = parse_preset(preset_path)
    kernels_in_preset = sorted({s.get("kernel") for s in sections if s.get("kernel")})
    if args.kernels:
        wanted = set(args.kernels.split(","))
        kernels = [k for k in kernels_in_preset if k in wanted]
    else:
        kernels = kernels_in_preset
    if not kernels:
        sys.exit("no matching kernels to run")

    env = make_env()
    resolved = resolve_events(args.perf)
    print(f"Zdarzenia rozwiazane: {resolved}")
    missing = [k for k in EVENT_CANDIDATES if k not in resolved]
    if missing:
        print(f"!! brak zdarzen (kolumny N/A): {missing}", file=sys.stderr)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = {"reps": args.reps, "target_sec": args.target_sec,
                "events": resolved, "kernels": {}}
    for kernel in kernels:
        print(f"\n######## CACHE: {kernel} ########")
        payload = measure_kernel(args.perf, kernel, sections, resolved,
                                 args.reps, args.target_sec, args.min_iters,
                                 args.max_iters, env)
        if payload is None:
            continue
        with (out_dir / f"results_cache_{kernel}.json").open("w") as f:
            json.dump(payload, f, indent=2)
        combined["kernels"][kernel] = payload
        print_table(payload)
    with (out_dir / "combined_cache.json").open("w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWyniki: {out_dir}/combined_cache.json")


if __name__ == "__main__":
    main()
