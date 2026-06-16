# Magisterka — Rafał Lenart

Kod źródłowy do pracy magisterskiej „Optymalizacja lokalności danych
i wykorzystania pamięci podręcznej w implementacjach wybranych funkcji BLAS
i operacji splotu" (UMCS, Informatyka).

Repozytorium zawiera własne implementacje czterech jąder obliczeniowych dla
AMD Zen 3 (Ryzen 5 5600, AVX2 256-bit, 2× FMA) wraz z infrastrukturą
benchmarkową, która porównuje je z bibliotekami producentów:

| Jądro | Katalog | Operacja |
| --- | --- | --- |
| `dot`  | `Kod/dot_product` | iloczyn skalarny (BLAS 1, `double`) |
| `gemv` | `Kod/gemv`        | mnożenie macierz-wektor (BLAS 2, `double`) |
| `gemm` | `Kod/gemm`        | mnożenie macierz-macierz (BLAS 3, `float`) |
| `conv` | `Kod/conv`        | splot 2D (`float`) |

Wszystkie pomiary wykonywane są domyślnie jednowątkowo (`OMP_NUM_THREADS=1`,
`taskset -c 0`); dostępne są też warianty równoległe w OpenMP.

---

## 1. Wymagania

**Do kompilacji jąder:**
- `gcc` ze wsparciem dla `-march=znver3` (na innych procesorach zmień flagę na
  `-march=native` w `Kod/common/CMakeLists.txt` i `Kod/<jądro>/CMakeLists.txt`),
- `cmake` (≥ 3.10), `make`, `git`,
- `python3` (skrypt instalacyjny sprawdza obecność tych narzędzi).

**Do narzędzi pomocniczych (Python):**
- `run_suite.py` — tylko biblioteka standardowa,
- `generate_plots.py` — `numpy`, `matplotlib`, `seaborn`,
- `run_cache.py` — narzędzie `perf` (Linux) z dostępem do liczników sprzętowych
  (np. `sudo sysctl kernel.perf_event_paranoid=1`).

Biblioteki producentów budowane są ze źródeł przez skrypt instalacyjny:
**OpenBLAS** (wymagana — jądra linkują się z nią na etapie kompilacji) oraz
opcjonalnie **AOCL-BLAS / BLIS** (ładowana w czasie działania przez `dlopen`,
dostarcza dodatkowe kolumny porównawcze w benchmarkach).

---

## 2. Budowanie

Z katalogu głównego repozytorium:

```bash
./install.sh                 # buduje OpenBLAS i wszystkie cztery jądra
./install.sh --with-blis     # dodatkowo AOCL-BLAS/BLIS (kolumny porównawcze)
./install.sh --jobs 12       # liczba wątków kompilacji (domyślnie nproc)
```

Inne opcje: `--skip-openblas`, `--skip-blis`, `--openblas-tag <tag>`,
`--blis-tag <tag>`, `--help`. Skrypt jest idempotentny — można go uruchamiać
wielokrotnie. (Flaga `--with-libxsmm` jest pozostałością; bieżąca wersja
benchmarku splotu nie korzysta już z libxsmm.)

Po zakończeniu powstają cztery binarki:

```
Kod/dot_product/build/benchmark
Kod/gemv/build/benchmark
Kod/gemm/build/benchmark
Kod/conv/build/benchmark
```

Budowa pojedynczego jądra ręcznie (bez skryptu, OpenBLAS musi już istnieć):

```bash
cmake -S Kod/gemm -B Kod/gemm/build -DCMAKE_BUILD_TYPE=Release
cmake --build Kod/gemm/build -j
```

Katalogi `build/` oraz źródła i instalacje OpenBLAS/BLIS są w `.gitignore`.

---

## 3. Uruchamianie pojedynczego jądra

Każda binarka `benchmark` przyjmuje ten sam zestaw argumentów. Wynik to mediana
GFLOPS z kilku przebiegów, z porównaniem względem wariantu naiwnego i OpenBLAS
oraz maksymalnym błędem względem wyniku referencyjnego.

**Pomiar jednowątkowy (zalecany — przypięty do rdzenia 0):**

```bash
cd Kod/gemv
OMP_NUM_THREADS=1 taskset -c 0 ./build/benchmark medium
```

**Argumenty wspólne dla wszystkich jąder:**

| Argument | Działanie |
| --- | --- |
| *(brak)* | uruchamia domyślny preset wbudowany |
| `<preset>` | konkretny preset wbudowany (np. `large`) |
| `all` | wszystkie presety wbudowane po kolei |
| `--preset-file <plik>` | presety z pliku INI (brane są tylko sekcje danego jądra) |
| `--custom <iter> <wymiary…>` | parametry podane ręcznie (wymiary zależą od jądra) |
| `--json <plik>` | zapis wyników do pliku JSON |
| `--list-impls` | wypisuje nazwy zaimplementowanych wariantów i kończy |
| `--measure <impl>` | uruchamia tylko jeden wariant bez pomiaru czasu (do `perf`/`run_cache.py`) |
| `-h`, `--help` | pomoc i lista presetów wbudowanych |

**Wymiary dla `--custom` i presety wbudowane:**

- **dot** — `--custom <iter> <size>`; presety: `l1_fit` (2048), `l2_fit`
  (32768, *domyślny*), `l3_fit` (2097152), `dram` (67108864).
  Warianty: `Naive`, `SIMD`, `SIMD MultiAcc`, `OMP`, `OpenBLAS`, `AOCL-BLAS`.

- **gemv** — `--custom <iter> <rows> <cols>`; presety: `tiny`, `small`,
  `medium` (*domyślny*), `large`, `wide`, `tall`.
  Warianty: `Naive`, `SIMD`, `SIMD + Prefetch`, `AVX+FMA Blocked`,
  `AVX+FMA V2`, `AVX+FMA V3`, `AVX+FMA V3_OMP`, `OpenBLAS`, `AOCL-BLAS`.

- **gemm** — `--custom <iter> <M> <N> <K>`; presety: `tiny`, `small`
  (*domyślny*), `medium`, `large`, `rank_k`, `tall_K`, `odd`.
  Warianty: m.in. `Naive`, `Loop Reorder ikj`, `Blocked`, `ST 6x8 packed`,
  `ST 4x12 packed`, `ST 4x12 intrinsics`, warianty wielowątkowe `MT …`,
  `MT Strassen`, `OpenBLAS`, `AOCL-BLAS`.

- **conv** — `--custom <iter> <Cin> <H> <W> <K> <Cout>`; presety: `tiny`,
  `small` (*domyślny*), `mid`, `large`, `xlarge`, `pointwise`, `kernel5`,
  `kernel7`, `rgb3x3`, `rgb5x5`, `rgb7x7`, `rgb_fhd`.
  Warianty: `Naive`, `Loop Reorder`, `Blocked`, `Packed Direct`, `OMP Packed`,
  `NCHWc direct`, `1x1 (SGEMM)`, `Winograd F(2,3)`, `Zen3 dispatch OMP`,
  `im2col + OpenBLAS`, `im2col + AOCL-BLAS`.

Pełną aktualną listę presetów i wariantów dla danego jądra zwracają
`./build/benchmark --help` oraz `./build/benchmark --list-impls`.

**Przykłady:**

```bash
# Pojedynczy rozmiar GEMM, własne parametry, 50 iteracji:
OMP_NUM_THREADS=1 taskset -c 0 Kod/gemm/build/benchmark --custom 50 1024 1024 1024

# Splot — wszystkie presety wbudowane, wynik do JSON:
OMP_NUM_THREADS=1 taskset -c 0 Kod/conv/build/benchmark all --json /tmp/conv.json

# Pomiar wielowątkowy (6 rdzeni fizycznych):
OMP_NUM_THREADS=6 taskset -c 0-5 Kod/gemv/build/benchmark large
```

Kolumna `AOCL-BLAS` pojawia się tylko, gdy zbudowano BLIS (`--with-blis`);
w przeciwnym razie jest pomijana. Biblioteka jest odnajdywana automatycznie
względem ścieżki binarki (`Kod/blis_install/lib`), więc nie trzeba ustawiać
`LD_LIBRARY_PATH`.

---

## 4. Zestaw benchmarków (`run_suite.py`)

Uruchamia wszystkie jądra dla wielu rozmiarów i wielu trybów wątkowych naraz,
łącząc wyniki w pliki JSON. Sam dobiera ścieżki binarek i przypina wątki przez
`taskset`.

```bash
python3 Kod/benchmark_suite/run_suite.py
```

Opcje:

| Opcja | Domyślnie | Znaczenie |
| --- | --- | --- |
| `--preset <plik>` | `presets/default.preset` | plik INI z definicjami przypadków |
| `--threads <lista>` | `1,<rdzenie fizyczne>` | np. `1,6` |
| `--kernels <lista>` | wszystkie z presetu | podzbiór, np. `dot,gemv` |
| `--out <katalog>` | `benchmark_suite/results` | katalog na wyniki i logi |

Pliki presetów: `Kod/benchmark_suite/presets/default.preset` (pełny zestaw) oraz
`quick.preset` (szybki, po jednym przypadku na jądro). Format to INI — sekcja na
przypadek, klucz `kernel` wybiera jądro:

```ini
[gemm_medium]
kernel = gemm
m = 1024
n = 1024
k = 1024
iterations = 10
```

Wyniki trafiają do `results/`: `results_<jądro>_<N>t.json`,
`log_<jądro>_<N>t.txt` oraz zbiorcze `combined_<N>t.json` (dla każdego trybu
wątkowego `N`). Na końcu wypisywane jest podsumowanie median GFLOPS.

---

## 5. Pomiar pamięci podręcznej (`run_cache.py`)

Mierzy zachowanie cache (współczynnik chybień L1, udział wypełnień z
L2/L3/DRAM, IPC) per implementacja, metodą różnicową (porównanie przebiegów
o `n` i `2n` iteracjach eliminuje koszty stałe). Wymaga `perf` i działa
jednowątkowo.

```bash
python3 Kod/benchmark_suite/run_cache.py
python3 Kod/benchmark_suite/run_cache.py --kernels gemm --reps 5
```

Opcje: `--preset`, `--kernels`, `--reps` (powtórzenia, domyślnie 3),
`--target-sec` (czas kalibracji `n`, domyślnie 0.25), `--min-iters`,
`--max-iters`, `--perf` (ścieżka do `perf`), `--out`.

Zdarzenia `perf` dobierane są automatycznie (warianty nazw AMD Zen). Wyniki:
`results/results_cache_<jądro>.json` oraz `results/combined_cache.json`,
a tabela jest wypisywana na ekran.

---

## 6. Wykresy (`generate_plots.py`)

Generuje wykresy PDF z plików `combined_<N>t.json` (po `run_suite.py`):

```bash
python3 Kod/benchmark_suite/generate_plots.py
python3 Kod/benchmark_suite/generate_plots.py --results <katalog> --out <katalog>
```

Domyślnie czyta z `benchmark_suite/results` i zapisuje do
`benchmark_suite/figures` (po jednym PDF na jądro i tryb wątkowy, plus zbiorcze
zestawienia wszystkich wariantów).

---

## 7. Testy

Testy jednostkowe parsera i metryk z `run_cache.py`:

```bash
cd Kod/benchmark_suite
python3 -m unittest test_run_cache
```

---

## 8. Typowy przepływ pracy

```bash
./install.sh --with-blis                          # 1. zbuduj wszystko
python3 Kod/benchmark_suite/run_suite.py          # 2. zmierz wydajność
python3 Kod/benchmark_suite/run_cache.py          # 3. (opcjonalnie) zmierz cache
python3 Kod/benchmark_suite/generate_plots.py     # 4. wygeneruj wykresy
```

---

## Uwagi

- Katalogi `Kod/blis_src`, `Kod/blis_install`, `Kod/openblas_src`,
  `Kod/openblas_install` oraz wszystkie `build/` są wytwarzane przez
  `install.sh` i nie są wersjonowane.
- Jądra zakładają wyrównanie danych (`aligned_alloc`) i są strojone pod
  Zen 3 — na innej mikroarchitekturze zmień flagi `-march`/`-mtune`.
- Skrypt `Kod/gemv/run_benchmarks.sh` to pomocniczy starszy runner samego GEMV
  (tryby `single`/`multi`/`both`); pełny, wieloplatformowy przepływ realizują
  skrypty w `Kod/benchmark_suite`.
