# Magisterka — Rafał Lenart

Implementacje jąder BLAS (dot, GEMV, GEMM, konwolucja 2D) dla AMD Zen 3.

## Instalacja

```bash
./install.sh                # buduje OpenBLAS i wszystkie jądra
./install.sh --with-blis    # dodatkowo BLIS (porównanie w benchmarkach)
./install.sh --jobs 12      # liczba wątków kompilacji
```

Skrypt jest idempotentny.

## Uruchamianie

Jedno jądro, jeden wątek:

```bash
cd Kod/gemv
OMP_NUM_THREADS=1 taskset -c 0 ./build/benchmark --preset-file ../benchmark_suite/presets/quick.preset
```

Pełny zestaw (wszystkie jądra, JSON do `Kod/benchmark_suite/results/`):

```bash
python3 Kod/benchmark_suite/run_suite.py
```
