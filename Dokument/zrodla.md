# Źródła pracy — do weryfikacji i ręcznego wstawienia

Ten plik to **rejestr wszystkich źródeł** wykorzystanych przy pisaniu pracy.
W treści pracy **nie ma jeszcze żadnych konkretnych cytowań ani bibliografii** —
w miejscach wymagających odwołania do literatury jest **placeholder**:

```latex
\src{etykieta}     % renderuje widoczny znacznik, np. [źródło: etykieta]
```

Każdy taki `\src{etykieta}` w `szablon.tex` odpowiada jednej pozycji poniżej
(po polu **Etykieta**). Dzięki temu możesz zweryfikować źródło i samodzielnie
wstawić właściwe `\cite{...}` w odpowiednim miejscu.

**Zasada: w tym pliku znajdują się wyłącznie pozycje ZWERYFIKOWANE** (metadane
sprawdzone w sieci lub powszechnie znane). Lista rośnie w miarę pisania kolejnych
podrozdziałów — dopisuję pozycję dopiero, gdy faktycznie z niej korzystam.

Status weryfikacji: ✅ = sprawdzone, ⏳ = wymaga Twojego potwierdzenia szczegółu.

---

## drepper-mem ✅
**Ulrich Drepper, „What Every Programmer Should Know About Memory", Red Hat, Inc., 2007.**
- Wersja 1.0 (21 listopada 2007).
- URL: https://www.akkadia.org/drepper/cpumemory.pdf
- Planowane użycie: §1.1 (ściana pamięci, lokalność), §1.2 (DRAM/SRAM), §1.3–1.4 (cache), §2.1–2.4 (lokalność, prefetching).

```bibtex
@misc{drepper2007memory,
  author       = {Ulrich Drepper},
  title        = {What Every Programmer Should Know About Memory},
  howpublished = {Red Hat, Inc.},
  year         = {2007},
  note         = {Wersja 1.0, 21 listopada 2007},
  url          = {https://www.akkadia.org/drepper/cpumemory.pdf}
}
```

## gottlieb-arch ✅
**Allan Gottlieb, „Lecture Notes for Computer Systems Design", wykład 22, New York University (kurs architektury komputerów, sem. jesienny 2001/2002).**
- URL: https://cs.nyu.edu/~gottlieb/courses/2000s/2001-02-fall/arch/lectures/lecture-22.html
- Potwierdzono: wykład obejmuje organizację cache (odwzorowanie bezpośrednie / zbiorowo-asocjacyjne / w pełni asocjacyjne), podział adresu na znacznik/zbiór/przesunięcie oraz politykę LRU.
- Planowane użycie: §1.3 (hierarchia pamięci), §1.4 (organizacja i adresowanie cache).

```bibtex
@misc{gottlieb_arch_lec22,
  author       = {Allan Gottlieb},
  title        = {Lecture Notes for Computer Systems Design (wyk{\l}ad 22)},
  howpublished = {New York University, kurs Computer Architecture, sem. jesienny 2001/2002},
  year         = {2001},
  url          = {https://cs.nyu.edu/~gottlieb/courses/2000s/2001-02-fall/arch/lectures/lecture-22.html}
}
```

## williams-roofline ✅
**S. Williams, A. Waterman, D. Patterson, „Roofline: an insightful visual performance model for multicore architectures", Communications of the ACM, t. 52, nr 4, s. 65–76, 2009.**
- DOI: 10.1145/1498765.1498785
- Planowane użycie: §1.1 (model roofline, intensywność operacyjna), §2.6.
- Użyte: §3.2 (intensywność operacyjna trzech poziomów BLAS; memory-/compute-bound; powiązanie z punktem grzbietowym); §4.2 (położenie splotu względem punktu grzbietowego — compute-bound).

```bibtex
@article{williams2009roofline,
  author  = {Samuel Williams and Andrew Waterman and David Patterson},
  title   = {Roofline: an insightful visual performance model for multicore architectures},
  journal = {Communications of the ACM},
  volume  = {52},
  number  = {4},
  pages   = {65--76},
  year    = {2009},
  doi     = {10.1145/1498765.1498785}
}
```

## evers-zen3 ✅
**M. Evers, L. Barnes, M. Clark, „The AMD Next-Generation »Zen 3« Core", IEEE Micro, t. 42, nr 3, s. 7–12, maj–czerwiec 2022.**
- DOI: 10.1109/MM.2022.3152788
- Planowane użycie: §1.3, §1.6, §1.7 (mikroarchitektura Zen 3 — cache, potok, jednostki FMA).

```bibtex
@article{evers2022zen3,
  author  = {Mark Evers and Leslie Barnes and Mike Clark},
  title   = {The {AMD} Next-Generation ``{Zen}~3'' Core},
  journal = {IEEE Micro},
  volume  = {42},
  number  = {3},
  pages   = {7--12},
  year    = {2022},
  doi     = {10.1109/MM.2022.3152788}
}
```

## goto-anatomy ✅
**K. Goto, R. A. van de Geijn, „Anatomy of high-performance matrix multiplication", ACM Transactions on Mathematical Software, t. 34, nr 3, art. 12, s. 1–25, 2008.**
- DOI: 10.1145/1356052.1356053
- Planowane użycie: §2.3 (blokowanie), §2.6 (cache-aware).
- Użyte: §3.2 (efekt powierzchnia–objętość; reużycie danych poziomu 3 jako podstawa blokowania); §3.3 (algorytm Goto jako podstawa organizacji BLIS/AOCL-BLAS: blokowanie MC/NC/KC, packing, mikrojądro MR×NR); §3.4 (GEMM jako centralne „jądro" wydajności; LAPACK dziedziczy wydajność po GEMM); §3.5 (biblioteki BLAS domyślnie mnożą klasycznie z blokowaniem, nie algorytmem Strassena); §4.2 (analogia efektu powierzchnia–objętość / reużycia danych poziomu 3 jako podstawa wysokiej intensywności splotu); §5.5 (anatomia 5 zagnieżdżonych pętli z packingiem jako organizacja autorskiego jądra GEMM; rola packingu A/B; współdzielony bufor B jako rozwiązanie równoległości po pętli ic — schemat Goto/BLIS).

```bibtex
@article{goto2008anatomy,
  author    = {Kazushige Goto and Robert A. van de Geijn},
  title     = {Anatomy of high-performance matrix multiplication},
  journal   = {ACM Transactions on Mathematical Software},
  volume    = {34},
  number    = {3},
  articleno = {12},
  pages     = {1--25},
  year      = {2008},
  doi       = {10.1145/1356052.1356053}
}
```

## wulf-mckee ✅
**W. A. Wulf, S. A. McKee, „Hitting the Memory Wall: Implications of the Obvious", ACM SIGARCH Computer Architecture News, t. 23, nr 1, s. 20–24, 1995.**
- DOI: 10.1145/216585.216588
- Planowane użycie: §1.1 (geneza pojęcia „ściana pamięci").

```bibtex
@article{wulf1995memorywall,
  author  = {William A. Wulf and Sally A. McKee},
  title   = {Hitting the Memory Wall: Implications of the Obvious},
  journal = {ACM SIGARCH Computer Architecture News},
  volume  = {23},
  number  = {1},
  pages   = {20--24},
  year    = {1995},
  doi     = {10.1145/216585.216588}
}
```

## hennessy-patterson ✅
**J. L. Hennessy, D. A. Patterson, „Computer Architecture: A Quantitative Approach", wyd. 6, Morgan Kaufmann, 2017.**
- ISBN: 978-0-12-811905-1
- Planowane użycie: §1.1, §1.3–1.5 (hierarchia, AMAT), §2.1–2.2 (lokalność, model 3C).

```bibtex
@book{hennessy2017architecture,
  author    = {John L. Hennessy and David A. Patterson},
  title     = {Computer Architecture: A Quantitative Approach},
  edition   = {6},
  publisher = {Morgan Kaufmann},
  year      = {2017},
  isbn      = {978-0-12-811905-1}
}
```

## fog-microarch ✅ (⏳ wersja/data)
**A. Fog, „The microarchitecture of Intel, AMD and VIA CPUs: An optimization guide for assembly programmers and compiler makers", Technical University of Denmark.**
- URL: https://www.agner.org/optimize/microarchitecture.pdf
- ⏳ Podać wersję/datę dostępu (dokument aktualizowany na bieżąco). Użyte: latencje i drożność cache Zen 3.
- Planowane użycie: §1.3 (latencje/drożność cache), §1.6, §1.7.
- Użyte: §5.5 (16 architektonicznych rejestrów YMM jako stała cecha AVX2; czterocyklowe opóźnienie FMA i dwie jednostki FMA jako podstawa wymogu ≥8 niezależnych łańcuchów; przewaga prefetchera sprzętowego nad programowym dla strumieniowych odczytów paneli). Potwierdzone niezależnie: Evers i in. 2022 (Zen 3 — 2× FMA, opóźnienie FMA 4 c) oraz Gepner 2017 (AVX2 — 16 YMM); zgodne z AMD SOG (order# 56665) §2.6/§2.12.

```bibtex
@misc{fog_microarchitecture,
  author       = {Agner Fog},
  title        = {The microarchitecture of {Intel}, {AMD} and {VIA} {CPUs}: An optimization guide for assembly programmers and compiler makers},
  howpublished = {Technical University of Denmark},
  note         = {dokument aktualizowany na bieżąco; podać datę dostępu},
  url          = {https://www.agner.org/optimize/microarchitecture.pdf}
}
```

## amd64-apm ✅ (⏳ rewizja)
**AMD, „AMD64 Architecture Programmer's Manual, Volume 1: Application Programming", publikacja nr 24592.**
- URL: https://docs.amd.com/v/u/en-US/24592
- ⏳ Podać numer rewizji/datę używanej wersji (np. rev. 3.24, 2025).
- Planowane użycie: §1.5 (rejestry ogólnego przeznaczenia x86-64), §1.7 (rejestry wektorowe, AVX2).

```bibtex
@manual{amd64_apm_vol1,
  title        = {AMD64 Architecture Programmer's Manual, Volume 1: Application Programming},
  organization = {Advanced Micro Devices},
  number       = {24592},
  note         = {podać numer rewizji/datę używanej wersji},
  url          = {https://docs.amd.com/v/u/en-US/24592}
}
```

## gepner-avx2 ✅
**P. Gepner, „Using AVX2 Instruction Set to Increase Performance of High Performance Computing Code", Computing and Informatics, t. 36, nr 5, s. 1001–1018, 2017.**
- DOI: 10.4149/cai-2017-5-1001
- Planowane użycie: §1.7 (instrukcje AVX2), §2.6 (wektoryzacja).
- Użyte: §5.5 (AVX2 — 16 rejestrów YMM, instrukcje FMA; podstawa rachunku rejestrów mikrojądra GEMM).

```bibtex
@article{gepner2017avx2,
  author  = {Pawel Gepner},
  title   = {Using {AVX2} Instruction Set to Increase Performance of High Performance Computing Code},
  journal = {Computing and Informatics},
  volume  = {36},
  number  = {5},
  pages   = {1001--1018},
  year    = {2017},
  doi     = {10.4149/cai-2017-5-1001}
}
```

## zhang-fma ✅
**H. Zhang, D. Chen, S. B. Ko, „Efficient Multiple-Precision Floating-Point Fused Multiply-Add with Mixed-Precision Support", IEEE Transactions on Computers, t. 68, nr 7, s. 1035–1048, 2019.**
- DOI: 10.1109/TC.2019.2895031
- Planowane użycie: §1.7 (operacja FMA).

```bibtex
@article{zhang2019fma,
  author  = {Hao Zhang and Dongdong Chen and Seok-Bum Ko},
  title   = {Efficient Multiple-Precision Floating-Point Fused Multiply-Add with Mixed-Precision Support},
  journal = {IEEE Transactions on Computers},
  volume  = {68},
  number  = {7},
  pages   = {1035--1048},
  year    = {2019},
  doi     = {10.1109/TC.2019.2895031}
}
```

## flynn-taxonomy ✅
**M. J. Flynn, „Some Computer Organizations and Their Effectiveness", IEEE Transactions on Computers, t. C-21, nr 9, s. 948–960, 1972.**
- DOI: 10.1109/TC.1972.5009071
- Planowane użycie: §1.7 (model SIMD jako klasa taksonomii Flynna).

```bibtex
@article{flynn1972taxonomy,
  author  = {Michael J. Flynn},
  title   = {Some Computer Organizations and Their Effectiveness},
  journal = {IEEE Transactions on Computers},
  volume  = {C-21},
  number  = {9},
  pages   = {948--960},
  year    = {1972},
  doi     = {10.1109/TC.1972.5009071}
}
```

## amd-sog ✅ (⏳ URL)
**AMD, „Software Optimization Guide for AMD Family 19h Processors", order# 56665, rev. 3.00, listopad 2020.**
- Rodzina 19h = mikroarchitektura Zen 3.
- ⏳ Dodać URL do wersji w portalu AMD (identyfikator: order# 56665).
- Treść zweryfikowana (pdftotext): sekcja 2.12.1 opisuje prefetchery sprzętowe (L1 Stream/Stride/Region, L2 Stream/Up-Down) i zaleca dopasowanie wzorca dostępu do nich; brak zalecenia przeciw prefetchowi programowemu (jedyna wzmianka: `PREFETCHNTA`).
- Planowane użycie: §2.5 (prefetchery sprzętowe Zen 3); ew. §1.3/1.6.

```bibtex
@manual{amd_sog_19h,
  title        = {Software Optimization Guide for AMD Family 19h Processors},
  organization = {Advanced Micro Devices},
  number       = {56665},
  edition      = {Rev. 3.00},
  year         = {2020}
}
```

## openmp-spec ✅
**OpenMP Architecture Review Board, „OpenMP Application Programming Interface, Version 5.2", listopad 2021.**
- URL: https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf
- Planowane użycie: §2.7 (równoległość zrealizowana w pracy za pomocą OpenMP).
- Użyte: §5.3, §5.4 (zrównoleglenie jąder dot/GEMV); §5.5 (zrównoleglenie GEMM — podział wymiaru M, równoległość po pętli ic ze współdzielonym buforem B, zrównoleglone przebiegi dodawania/odejmowania macierzy w algorytmie Strassena).

```bibtex
@manual{openmp52,
  title        = {OpenMP Application Programming Interface, Version 5.2},
  organization = {OpenMP Architecture Review Board},
  year         = {2021},
  url          = {https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf}
}
```

## frigo-cacheoblivious ✅
**M. Frigo, C. E. Leiserson, H. Prokop, S. Ramachandran, „Cache-Oblivious Algorithms", w: Proceedings of the 40th Annual Symposium on Foundations of Computer Science (FOCS), s. 285–297, 1999.**
- DOI: 10.1109/SFFCS.1999.814600
- Planowane użycie: §2.8 (algorytmy cache-oblivious).
- Użyte: §3.5 (rekurencyjny podział Strassena ma charakter zbliżony do strategii cache-oblivious).

```bibtex
@inproceedings{frigo1999cacheoblivious,
  author    = {Matteo Frigo and Charles E. Leiserson and Harald Prokop and Sridhar Ramachandran},
  title     = {Cache-Oblivious Algorithms},
  booktitle = {Proceedings of the 40th Annual Symposium on Foundations of Computer Science (FOCS)},
  pages     = {285--297},
  year      = {1999},
  doi       = {10.1109/SFFCS.1999.814600}
}
```

## bandwidth ✅
**Z. T. Smith, „bandwidth --- a memory bandwidth benchmark", narzędzie open source.**
- URL: https://zsmith.co/bandwidth.html
- Dostęp: 7 czerwca 2026.
- Zweryfikowano (strona narzędzia oraz niezależne kopie/forki i odniesienia: GitHub mmatyas/bandwidth-benchmark, raptor-engineering/bandwidth-benchmark; narzędzie pmbw wskazujące „bandwidth" Z. Smitha jako pierwowzór): benchmark mierzący przepustowość pamięci na procesorach x86 (i innych) --- sekwencyjne odczyty, zapisy, kopiowanie oraz operacje nietymczasowe przy szerokościach 64/128/256 bitów dla rosnących rozmiarów bufora.
- Użyte: §5.1 (pomiar przepustowości poszczególnych poziomów hierarchii pamięci na maszynie testowej; rysunek przepustowości `rys:bandwidth`).

```bibtex
@misc{smith_bandwidth,
  author       = {Zack T. Smith},
  title        = {bandwidth: a memory bandwidth benchmark},
  howpublished = {\url{https://zsmith.co/bandwidth.html}},
  note         = {narz\k{e}dzie open source; dost\k{e}p: 7 czerwca 2026}
}
```

## ramcalc ✅ (⏳ autor/dostęp)
**FinlayDaG33k, „RAM Bandwidth Calculator", narzędzie internetowe (kalkulator przepustowości pamięci).**
- URL: https://edu.finlaydag33k.nl/calculating%20ram%20bandwidth/ ; repozytorium: https://gitlab.com/FinlayDaG33k/ram-calculator
- Dostęp: 8 czerwca 2026.
- Zweryfikowano (repozytorium GitLab; wartość potwierdzona niezależnie): kalkulator stosuje standardową formułę przepustowości pamięci (liczba transferów na sekundę × szerokość kanału w bajtach × liczba kanałów); dla DDR4-3200 w trybie dwukanałowym daje $3200 \times 8 \times 2 = 51{,}2$ GB/s --- wartość potwierdzona w wielu niezależnych źródłach (równoważnie $1600\,\text{MHz} \times 16\,\text{B} \times 2$). ⚠ Strona interaktywna nie renderuje się bez JavaScriptu; sama formuła i wynik są jednak standardowe. Do rozważenia zamiana na bardziej autorytatywne źródło (JEDEC / podręcznik), jeśli promotor zakwestionuje narzędzie internetowe.
- Użyte: §5.1 (teoretyczna przepustowość pamięci operacyjnej DDR4-3200, 51,2 GB/s).

```bibtex
@misc{finlaydag33k_ramcalc,
  author       = {{FinlayDaG33k}},
  title        = {{RAM} Bandwidth Calculator},
  howpublished = {\url{https://gitlab.com/FinlayDaG33k/ram-calculator}},
  note         = {narz\k{e}dzie internetowe; dost\k{e}p: 8 czerwca 2026}
}
```

## Dodatkowe źródła (znalezione na potrzeby przeglądu, do ewentualnego użycia)

### csapp ✅
**R. E. Bryant, D. R. O'Hallaron, „Computer Systems: A Programmer's Perspective", wyd. 3, Pearson, 2015.**
- ISBN: 978-0-13-409266-9
- Możliwe użycie: §1.3–1.4 (hierarchia, organizacja cache), §2.1–2.3 (lokalność, optymalizacje) — uznany podręcznik, alternatywny wobec Hennessy & Patterson.
```bibtex
@book{bryant2015csapp,
  author    = {Randal E. Bryant and David R. O'Hallaron},
  title     = {Computer Systems: A Programmer's Perspective},
  edition   = {3},
  publisher = {Pearson},
  year      = {2015},
  isbn      = {978-0-13-409266-9}
}
```

### smith-cache-memories ✅ (⏳ DOI)
**A. J. Smith, „Cache Memories", ACM Computing Surveys, t. 14, nr 3, s. 473–530, 1982.**
- DOI (⏳ do potwierdzenia): 10.1145/356887.356892
- Możliwe użycie: §1.3 (klasyczny, obszerny przegląd pamięci podręcznych).
```bibtex
@article{smith1982cache,
  author  = {Alan Jay Smith},
  title   = {Cache Memories},
  journal = {ACM Computing Surveys},
  volume  = {14},
  number  = {3},
  pages   = {473--530},
  year    = {1982}
}
```

### hill-associativity ✅ (⏳ DOI)
**M. D. Hill, A. J. Smith, „Evaluating Associativity in CPU Caches", IEEE Transactions on Computers, t. 38, nr 12, s. 1612–1630, 1989.**
- DOI (⏳ do potwierdzenia): 10.1109/12.40842
- Możliwe użycie: §1.4 (drożność), §2.2 (model 3C — pierwotne źródło klasyfikacji chybień, obok H&P).
```bibtex
@article{hill1989associativity,
  author  = {Mark D. Hill and Alan Jay Smith},
  title   = {Evaluating Associativity in {CPU} Caches},
  journal = {IEEE Transactions on Computers},
  volume  = {38},
  number  = {12},
  pages   = {1612--1630},
  year    = {1989}
}
```

### mowry-prefetch ✅
**T. C. Mowry, M. S. Lam, A. Gupta, „Design and Evaluation of a Compiler Algorithm for Prefetching", w: Proc. 5th Int. Conf. on Architectural Support for Programming Languages and Operating Systems (ASPLOS), s. 62–73, 1992.**
- DOI: 10.1145/143371.143488
- Możliwe użycie: §2.5 (prefetching programowy — klasyczne źródło dla prefetchu sterowanego programowo).
```bibtex
@inproceedings{mowry1992prefetch,
  author    = {Todd C. Mowry and Monica S. Lam and Anoop Gupta},
  title     = {Design and Evaluation of a Compiler Algorithm for Prefetching},
  booktitle = {Proceedings of the 5th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
  pages     = {62--73},
  year      = {1992},
  doi       = {10.1145/143371.143488}
}
```

### lawson-blas ✅
**C. L. Lawson, R. J. Hanson, D. R. Kincaid, F. T. Krogh, „Basic Linear Algebra Subprograms for Fortran Usage", ACM Transactions on Mathematical Software, t. 5, nr 3, s. 308–323, 1979.**
- DOI: 10.1145/355841.355847
- Użyte: §1.1 (BLAS) oraz §3.1 (geneza BLAS — poziom 1, 1979; operacje wektor–wektor; konwencja nazw s/d/c/z) — oryginalne źródło BLAS (poziom 1); §3.4 (definicja iloczynu skalarnego dot, poziom 1; axpy jako operacja poziomu 1).
```bibtex
@article{lawson1979blas,
  author  = {C. L. Lawson and R. J. Hanson and D. R. Kincaid and F. T. Krogh},
  title   = {Basic Linear Algebra Subprograms for {Fortran} Usage},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {5},
  number  = {3},
  pages   = {308--323},
  year    = {1979},
  doi     = {10.1145/355841.355847}
}
```

### intel-optimization ✅ (⏳ rewizja)
**Intel, „Intel 64 and IA-32 Architectures Optimization Reference Manual", order# 248966 (rev. 050, 2024).**
- ⏳ Podać używaną rewizję (np. 248966-050).
- Możliwe użycie: §1.7, §2.5–2.6 (techniki SIMD/cache) — niezależne, przemysłowe źródło komplementarne do AMD SOG (uwaga: opisuje procesory Intela, nie AMD — używać do ogólnych technik, nie do liczb Zen 3).
```bibtex
@manual{intel_optimization,
  title        = {Intel 64 and IA-32 Architectures Optimization Reference Manual},
  organization = {Intel Corporation},
  number       = {248966},
  year         = {2024}
}
```

### dongarra-blas2 ✅
**J. J. Dongarra, J. Du Croz, S. Hammarling, R. J. Hanson, „An Extended Set of FORTRAN Basic Linear Algebra Subprograms", ACM Transactions on Mathematical Software, t. 14, nr 1, s. 1–17, 1988.**
- DOI: 10.1145/42288.42291
- Zweryfikowano (ACM Digital Library oraz artykuł Blackford i in. 2002, cytowanie [Dongarra et al. 1988]): poziom 2 BLAS — operacje macierz–wektor, marzec 1988, TOMS 14(1):1–17. Model implementacji opublikowano jako Algorithm 656 (TOMS 14(1):18–32).
- Motywacja (poziom 2 → procesory wektorowe): potwierdzona u Blackford i in. 2002 (odczytane wprost): „With the advent of vector machines, hierarchical memory machines and shared memory parallel machines, specifications for the Level 2 and 3 BLAS [...] were drawn up"; potwierdzają to także przegląd Dongarry oraz hasło encyklopedyczne BLAS.
- Użyte: §3.1 (rys historyczny — poziom 2, 1988; motywacja poziomu 2 — procesory wektorowe); §3.4 (definicja GEMV, poziom 2).
```bibtex
@article{dongarra1988blas2,
  author  = {Jack J. Dongarra and Jeremy Du Croz and Sven Hammarling and Richard J. Hanson},
  title   = {An Extended Set of {FORTRAN} Basic Linear Algebra Subprograms},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {14},
  number  = {1},
  pages   = {1--17},
  year    = {1988},
  doi     = {10.1145/42288.42291}
}
```

### dongarra-blas3 ✅
**J. J. Dongarra, J. Du Croz, I. S. Duff, S. Hammarling, „A Set of Level 3 Basic Linear Algebra Subprograms", ACM Transactions on Mathematical Software, t. 16, nr 1, s. 1–17, 1990.**
- DOI: 10.1145/77626.79170
- Zweryfikowano (ACM Digital Library, Research Explorer Manchester oraz artykuł Blackford i in. 2002, cytowanie [Dongarra et al. 1990]): poziom 3 BLAS — operacje macierz–macierz, marzec 1990, TOMS 16(1):1–17. Model implementacji opublikowano jako Algorithm 679 (TOMS 16(1):18–28).
- Motywacja (poziom 3 → hierarchia pamięci, algorytmy blokowe): potwierdzona u Blackford i in. 2002 (odczytane wprost): „These specifications made it possible to construct new software to more effectively utilize the memory hierarchy of modern computers. In particular, the Level 3 BLAS allowed the construction of software based upon block-partitioned algorithms"; potwierdzają to także przegląd Dongarry oraz notatki wykładowe Demmela (CS267).
- Użyte: §3.1 (rys historyczny — poziom 3, 1990; motywacja poziomu 3 — hierarchia pamięci); §3.4 (definicja GEMM, poziom 3).
```bibtex
@article{dongarra1990blas3,
  author  = {Jack J. Dongarra and Jeremy Du Croz and Iain S. Duff and Sven Hammarling},
  title   = {A Set of Level 3 Basic Linear Algebra Subprograms},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {16},
  number  = {1},
  pages   = {1--17},
  year    = {1990},
  doi     = {10.1145/77626.79170}
}
```

### blackford-blas-standard ✅
**L. S. Blackford, J. Demmel, J. Dongarra, I. Duff, S. Hammarling, G. Henry, M. Heroux, L. Kaufman, A. Lumsdaine, A. Petitet, R. Pozo, K. Remington, R. C. Whaley, „An Updated Set of Basic Linear Algebra Subprograms (BLAS)", ACM Transactions on Mathematical Software, t. 28, nr 2, s. 135–151, 2002.**
- DOI: 10.1145/567806.567807
- Zweryfikowano (ACM Digital Library, NIST oraz tekst PDF artykułu): podsumowuje standard BLAS Technical Forum; potwierdza, że pierwotną specyfikację BLAS podano w Fortranie 66, a następnie 77 (s. 167–168), a standard wprowadza wiązania dla Fortranu 95, Fortranu 77 oraz języka C (CBLAS); implementacje referencyjne dostępne na stronie BLAS Technical Forum. Potwierdza również podział na poziomy: poziom 1 — skalary i wektory [Lawson et al. 1979], poziom 2 — macierz–wektor, poziom 3 — macierz–macierz [Dongarra et al. 1988; 1990].
- Użyte: §3.1 (standard / interfejs C / CBLAS; potwierdzenie podziału na poziomy); §3.4 (algorytmy blokowe LAPACK sprowadzane do operacji poziomu 3).
```bibtex
@article{blackford2002blas,
  author  = {L. Susan Blackford and James Demmel and Jack Dongarra and Iain Duff and Sven Hammarling and Greg Henry and Michael Heroux and Linda Kaufman and Andrew Lumsdaine and Antoine Petitet and Roldan Pozo and Karin Remington and R. Clint Whaley},
  title   = {An Updated Set of Basic Linear Algebra Subprograms ({BLAS})},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {28},
  number  = {2},
  pages   = {135--151},
  year    = {2002},
  doi     = {10.1145/567806.567807}
}
```

### netlib-blas ✅
**Netlib, „BLAS (Basic Linear Algebra Subprograms)", strona projektu Netlib.**
- URL: https://www.netlib.org/blas/
- Dostęp: 6 czerwca 2026 (strona aktualizowana).
- Zweryfikowano (treść strony): Netlib udostępnia referencyjną implementację BLAS w Fortranie 77 („the user can download a Fortran77 reference implementation of the BLAS from netlib"); opisuje trzy poziomy („The Level 1 BLAS perform scalar, vector and vector-vector operations, the Level 2 BLAS perform matrix-vector operations, and the Level 3 BLAS perform matrix-matrix operations"); wskazuje, że BLAS są wykorzystywane w tworzeniu oprogramowania algebry liniowej, na przykład LAPACK; zawiera sekcję CBLAS (interfejs C, obecnie część dystrybucji LAPACK).
- Użyte: §3.1 (implementacja referencyjna Netlib; CBLAS; LAPACK budowany na BLAS); §3.3 (wzorzec poprawności, nie wydajności; potrójne pętle, brak optymalizacji cache).
```bibtex
@misc{netlib_blas,
  author       = {{Netlib}},
  title        = {{BLAS} (Basic Linear Algebra Subprograms)},
  howpublished = {strona projektu Netlib},
  note         = {dost\k{e}p: 6 czerwca 2026},
  url          = {https://www.netlib.org/blas/}
}
```

### demmel-cs267 ✅
**J. Demmel, „CS267: Applications of Parallel Computers — wykład: gęsta algebra liniowa i hierarchie pamięci", University of California, Berkeley, wiosna 2015.**
- URL: https://people.eecs.berkeley.edu/~demmel/cs267_Spr15/Lectures/lecture12_densela_1_jwd15_4pp.pdf
- Dostęp: 6 czerwca 2026.
- Zweryfikowano (pdftotext, odczytane wprost): definicja intensywności obliczeniowej q jako liczby działań na dostęp do wolnej pamięci; poziomy BLAS jako O(n)/O(n^2)/O(n^3) działań na O(n)/O(n^2)/O(n^2) danych; q poziomu 1 = 1, poziomu 2 = 2, poziomu 3 rosnące z n; „Good for machines with caches, other mem. hierarchy levels"; dolne ograniczenie wejścia–wyjścia Ω(n^3/√M) przypisane Hong, Kung 1981. Wartości i definicję potwierdza niezależnie Williams i in. 2009 (intensywność operacyjna) oraz hasło „Arithmetic Intensity" (ScienceDirect).
- Użyte: §3.2 (skalowanie pracy/danych dla trzech poziomów; miara q; efekt powierzchnia–objętość).
```bibtex
@misc{demmel_cs267_memhier,
  author       = {James Demmel},
  title        = {{CS267}: Dense Linear Algebra and Memory Hierarchies (lecture notes)},
  howpublished = {University of California, Berkeley},
  year         = {2015},
  note         = {dost\k{e}p: 6 czerwca 2026},
  url          = {https://people.eecs.berkeley.edu/~demmel/cs267_Spr15/Lectures/lecture12_densela_1_jwd15_4pp.pdf}
}
```

### hong-kung-io ✅
**Hong Jia-Wei, H. T. Kung, „I/O complexity: The red-blue pebble game", w: Proceedings of the 13th Annual ACM Symposium on Theory of Computing (STOC '81), s. 326–333, 1981.**
- DOI: 10.1145/800076.802486
- Zweryfikowano (ACM Digital Library oraz dblp): STOC 1981, s. 326–333; klasyczny wynik o dolnym ograniczeniu ruchu wejścia–wyjścia, w tym Ω(n^3/√M) dla mnożenia macierzy w modelu dwupoziomowej pamięci. Wynik przywoływany także przez Demmela (CS267).
- Użyte: §3.2 (dolne ograniczenie złożoności wejścia–wyjścia mnożenia macierzy; granica reużycia danych w pamięci podręcznej).
```bibtex
@inproceedings{hong1981redblue,
  author    = {Hong Jia-Wei and H. T. Kung},
  title     = {{I/O} complexity: The red-blue pebble game},
  booktitle = {Proceedings of the 13th Annual ACM Symposium on Theory of Computing (STOC)},
  pages     = {326--333},
  year      = {1981},
  doi       = {10.1145/800076.802486}
}
```

### vanzee-blis ✅
**F. G. Van Zee, R. A. van de Geijn, „BLIS: A Framework for Rapidly Instantiating BLAS Functionality", ACM Transactions on Mathematical Software, t. 41, nr 3, art. 14, 2015.**
- DOI: 10.1145/2764454
- Zweryfikowano (dblp, Google Scholar, ACM DL oraz FAQ projektu BLIS flame/blis): framework BLIS sprowadza obliczenia poziomu 2 i 3 BLAS do prostych jąder; mikrojądro = podstawowa, zależna od architektury jednostka obliczeń poziomu 3; makrojądro = przenośny kod dzielący operację wg rozmiarów blokowania pamięci podręcznej.
- Użyte: §3.3 (architektura BLIS: mikrojądro zależne od architektury + przenośne makrojądro; podstawa AOCL-BLAS); §5.5 (kafelek domyślny BLIS 6×8 jako wybór pod przenośność; mikrojądro/makrojądro i blokowanie jako wzorzec autorskiego jądra GEMM; współdzielony bufor B jako rozwiązanie równoległości BLIS; jądra mnożenia macierzy rodziny Zen pisane w ręcznym asemblerze z powodu nasycenia rejestrów). Niezależne potwierdzenie ręcznego asemblera jąder DGEMM: kod źródłowy OpenBLAS (`dgemm_kernel_4x8_haswell.S`, wspólny dla rodziny Haswell/Zen).
```bibtex
@article{vanzee2015blis,
  author    = {Field G. Van Zee and Robert A. van de Geijn},
  title     = {{BLIS}: A Framework for Rapidly Instantiating {BLAS} Functionality},
  journal   = {ACM Transactions on Mathematical Software},
  volume    = {41},
  number    = {3},
  articleno = {14},
  year      = {2015},
  doi       = {10.1145/2764454}
}
```

### openblas ✅ (⏳ strony/data dostępu)
**Zhang Xianyi, Wang Qian, Zhang Yunquan, „Model-driven Level 3 BLAS Performance Optimization on Loongson 3A Processor", w: Proceedings of the 2012 IEEE 18th International Conference on Parallel and Distributed Systems (ICPADS), s. 684–691, 2012.**
- DOI: 10.1109/ICPADS.2012.97
- Zweryfikowano (dblp; dokumentacja „About" projektu OpenBLAS; repozytorium OpenMathLib/OpenBLAS): jeden z dwóch artykułów wskazywanych przez projekt OpenBLAS jako cytowanie. Fakty o OpenBLAS — rodowód z GotoBLAS2 (1.13, licencja BSD), ręcznie strojone jądra dla wielu architektur, DYNAMIC_ARCH (dobór jądra w czasie wykonania), wielowątkowość — potwierdzone w repozytorium projektu oraz niezależnie w hasłach o GotoBLAS i Kazushige Goto (Goto, TACC, ręczny asembler GEMM).
- ⏳ Potwierdzić numery stron (684–691 wg dblp/ScienceDirect) oraz datę dostępu do strony projektu.
- Użyte: §3.3 (OpenBLAS — rodowód z GotoBLAS2 K. Goto; ręcznie strojone jądra; DYNAMIC_ARCH; wielowątkowość; otwarte źródło); §5.3, §5.4, §5.5 (biblioteka referencyjna; wołana przez interfejs C `cblas_dgemm`/`cblas_dgemv`/`cblas_ddot`); §5.5 (jądra DGEMM rodziny Zen pisane w ręcznym asemblerze z powodu nasycenia 16 YMM — potwierdzone w kodzie źródłowym `dgemm_kernel_4x8_haswell.S`, wspólnym dla Haswell/Zen).
```bibtex
@inproceedings{zhang2012openblas,
  author    = {Zhang Xianyi and Wang Qian and Zhang Yunquan},
  title     = {Model-driven {Level} 3 {BLAS} Performance Optimization on {Loongson} 3A Processor},
  booktitle = {Proceedings of the 2012 IEEE 18th International Conference on Parallel and Distributed Systems (ICPADS)},
  pages     = {684--691},
  year      = {2012},
  doi       = {10.1109/ICPADS.2012.97}
}
```

### amd-aocl ✅ (⏳ wersja/data dostępu)
**AMD, „AOCL-BLAS (Basic Linear Algebra Subprograms)", dokumentacja AMD Optimizing CPU Libraries; repozytorium źródłowe amd/blis.**
- URL (portal): https://www.amd.com/en/developer/aocl/blis.html ; repozytorium: https://github.com/amd/blis
- Zweryfikowano (README amd/blis, strona AMD AOCL): „AOCL-BLAS is AMD's optimized version of BLIS targeted for AMD EPYC and Ryzen CPUs. It is developed as a forked version of BLIS"; BLIS pochodzi z University of Texas at Austin; AOCL-BLAS jest składnikiem AOCL (zestaw bibliotek pod architekturę „Zen": EPYC/Ryzen/Threadripper).
- ⏳ Wskazać konkretną wersję/numer publikacji (np. AOCL User Guide) i datę dostępu.
- Użyte: §3.3 (AOCL-BLAS = rozwidlenie BLIS od AMD; składnik AOCL; optymalizacje pod „Zen"); §5.3, §5.4, §5.5 (biblioteka referencyjna; wołana przez interfejs Fortranu `dgemm_`/`dgemv_`/`ddot_` — dla GEMM z argumentami przekazanymi w zamienionej kolejności i z przestawionymi wymiarami, co realizuje mnożenie w układzie wierszowym).
```bibtex
@misc{amd_aocl_blas,
  author       = {{Advanced Micro Devices}},
  title        = {{AOCL-BLAS}: Basic Linear Algebra Subprograms Library},
  howpublished = {AMD Optimizing CPU Libraries (AOCL); repozytorium \url{https://github.com/amd/blis}},
  note         = {dost\k{e}p: podać datę},
  url          = {https://www.amd.com/en/developer/aocl/blis.html}
}
```

### whaley-atlas ✅
**R. C. Whaley, A. Petitet, J. J. Dongarra, „Automated empirical optimizations of software and the ATLAS project", Parallel Computing, t. 27, nr 1–2, s. 3–35, 2001.**
- DOI: 10.1016/S0167-8191(00)00087-9
- Zweryfikowano (Research Explorer Manchester, Google Scholar, ScienceDirect): Parallel Computing 27(1–2):3–35, 2001; wprowadza automatyczne empiryczne strojenie oprogramowania (AEOS) i projekt ATLAS (Automatically Tuned Linear Algebra Software).
- Użyte: §3.3 (ATLAS — automatyczne strojenie empiryczne jako alternatywa wobec ręcznie pisanych jąder).
```bibtex
@article{whaley2001atlas,
  author  = {R. Clint Whaley and Antoine Petitet and Jack J. Dongarra},
  title   = {Automated empirical optimizations of software and the {ATLAS} project},
  journal = {Parallel Computing},
  volume  = {27},
  number  = {1--2},
  pages   = {3--35},
  year    = {2001},
  doi     = {10.1016/S0167-8191(00)00087-9}
}
```

### nvidia-cublas ✅ (⏳ wersja/data dostępu)
**NVIDIA, „cuBLAS Library Documentation", NVIDIA CUDA Toolkit.**
- URL: https://docs.nvidia.com/cuda/cublas/
- Zweryfikowano (dokumentacja NVIDIA): cuBLAS to implementacja BLAS na bazie środowiska CUDA, udostępniająca zasoby obliczeniowe procesorów graficznych NVIDIA; obsługuje poziomy 1, 2 i 3 BLAS.
- ⏳ Podać numer wersji CUDA/cuBLAS i datę dostępu.
- Użyte: §3.3 (cuBLAS — implementacja BLAS na GPU, tło).
```bibtex
@manual{nvidia_cublas,
  title        = {{cuBLAS} Library Documentation},
  organization = {NVIDIA Corporation},
  note         = {część CUDA Toolkit; podać wersję i datę dostępu},
  url          = {https://docs.nvidia.com/cuda/cublas/}
}
```

### intel-onemkl ✅ (⏳ data dostępu)
**Intel, „oneAPI Math Kernel Library (oneMKL)" (dawniej Intel Math Kernel Library, Intel MKL), dokumentacja produktu.**
- URL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
- Zweryfikowano (strona produktu Intel oraz hasło Math Kernel Library): zamknięta, własnościowa biblioteka dostrojona pod procesory x86 Intela; dostarcza BLAS, LAPACK, FFT i inne procedury; dawna nazwa Intel Math Kernel Library (Intel MKL).
- ⏳ Podać datę dostępu.
- Użyte: §3.3 (tło: Intel oneMKL jako zamknięta implementacja BLAS pod x86).
```bibtex
@misc{intel_onemkl,
  author       = {{Intel Corporation}},
  title        = {{oneAPI} Math Kernel Library ({oneMKL})},
  howpublished = {dokumentacja produktu Intel},
  note         = {dawniej Intel Math Kernel Library; dost\k{e}p: podać datę},
  url          = {https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html}
}
```

### golub-vanloan ✅ (⏳ numery stron)
**G. H. Golub, C. F. Van Loan, „Matrix Computations", 4. wyd., Johns Hopkins University Press, Baltimore, 2013.**
- ISBN: 978-1-4214-0794-4; seria: Johns Hopkins Studies in the Mathematical Sciences.
- Zweryfikowano (strona wydawcy JHU Press; SIAM Publications Library — zgodny ISBN i rok): 4. wydanie, 2013. Definicje i wzorce dostępu potwierdzone niezależnie w materiale dydaktycznym C. Van Loana (CS4220, Cornell, rozdz. 5): mnożenie macierz–wektor w ujęciu wierszowym (iloczyny skalarne wierszy z wektorem) oraz kolumnowym (axpy: „vector ← scalar·vector + vector"); uwaga o ciągłości pamięci (wariant kolumnowy dostępuje elementy ciągłe, wierszowy — nieciągłe).
- ⏳ Uzupełnić numery stron, jeśli cytowana będzie konkretna strona.
- Użyte: §3.4 (kanoniczne definicje iloczynu skalarnego, GEMV i GEMM; wzorce dostępu wierszowy/kolumnowy GEMV; axpy jako poziom 1).
```bibtex
@book{golub2013matrix,
  author    = {Gene H. Golub and Charles F. Van Loan},
  title     = {Matrix Computations},
  edition   = {4},
  publisher = {Johns Hopkins University Press},
  address   = {Baltimore},
  year      = {2013},
  isbn      = {978-1-4214-0794-4}
}
```

### kagstrom-gemm ✅
**B. Kågström, P. Ling, C. Van Loan, „GEMM-based level 3 BLAS: high-performance model implementations and performance evaluation benchmark", ACM Transactions on Mathematical Software, t. 24, nr 3, s. 268–302, 1998.**
- DOI: 10.1145/292395.292412
- Zweryfikowano (ACM Digital Library; Semantic Scholar): TOMS 24(3):268–302, 1998; teza — przenośną, wysokowydajną bibliotekę poziomu 3 BLAS można zbudować, opierając pozostałe operacje poziomu 3 na zoptymalizowanym GEMM oraz niewielkiej liczbie działań poziomu 1 i 2. Artykuł towarzyszący: Algorithm 784 (DOI 10.1145/292395.292426).
- Użyte: §3.4 (pozostałe operacje poziomu 3 wyrażane przez GEMM → GEMM jako wspólna podstawa poziomu 3).
```bibtex
@article{kagstrom1998gemm,
  author  = {Bo K{\aa}gstr{\"o}m and Per Ling and Charles {Van Loan}},
  title   = {{GEMM}-based level 3 {BLAS}: high-performance model implementations and performance evaluation benchmark},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {24},
  number  = {3},
  pages   = {268--302},
  year    = {1998},
  doi     = {10.1145/292395.292412}
}
```

### strassen ✅
**V. Strassen, „Gaussian elimination is not optimal", Numerische Mathematik, t. 13, s. 354–356, 1969.**
- DOI: 10.1007/BF02165411
- Zweryfikowano (EuDML, SpringerLink): tom 13 (nie 14), s. 354–356, 1969. Wzory 7 iloczynów M1–M7 i rekonstrukcji bloków C zgodne w dwóch niezależnych źródłach (co do znaku) oraz spójne z kodem autora `Kod/gemm/gemm_strassen.c`. Złożoność O(n^log2(7)), log2(7) ≈ 2,807.
- Użyte: §3.5 (idea Strassena; 7 mnożeń zamiast 8; wzory M1–M7; rekonstrukcja Cij; rekurencja; złożoność O(n^2,807)).
```bibtex
@article{strassen1969gaussian,
  author  = {Volker Strassen},
  title   = {Gaussian elimination is not optimal},
  journal = {Numerische Mathematik},
  volume  = {13},
  pages   = {354--356},
  year    = {1969},
  doi     = {10.1007/BF02165411}
}
```

### coppersmith-winograd ✅
**D. Coppersmith, S. Winograd, „Matrix multiplication via arithmetic progressions", Journal of Symbolic Computation, t. 9, nr 3, s. 251–280, 1990.**
- DOI: 10.1016/S0747-7171(08)80013-2
- Zweryfikowano (IBM Research, ScienceDirect): JSC 9(3):251–280, 1990; wykładnik mnożenia macierzy ω ≈ 2,376.
- Użyte: §3.5 (teoretyczna granica wykładnika ω ≈ 2,376).
```bibtex
@article{coppersmith1990matrix,
  author  = {Don Coppersmith and Shmuel Winograd},
  title   = {Matrix multiplication via arithmetic progressions},
  journal = {Journal of Symbolic Computation},
  volume  = {9},
  number  = {3},
  pages   = {251--280},
  year    = {1990},
  doi     = {10.1016/S0747-7171(08)80013-2}
}
```

### higham-stability ✅
**N. J. Higham, „Exploiting fast matrix multiplication within the Level 3 BLAS", ACM Transactions on Mathematical Software, t. 16, nr 4, s. 352–368, 1990.**
- DOI: 10.1145/98267.98290
- Zweryfikowano (ACM DL, Research Explorer Manchester, kopia autorska nhigham.com): TOMS 16(4):352–368, 1990. Treść: stabilność szybkiego mnożenia w poziomie 3 BLAS (słabsza niż klasyczna, normowa nie składnikowa, pogarsza się z głębokością rekurencji, lecz wystarczająca dla wielu zastosowań); praktyczny próg odcięcia rzędu kilkudziesięciu; wariant Strassena–Winograda (18→15 dodawań).
- Użyte: §3.5 (próg odcięcia; stabilność numeryczna; wariant Strassena–Winograda; złożoność).
```bibtex
@article{higham1990exploiting,
  author  = {Nicholas J. Higham},
  title   = {Exploiting fast matrix multiplication within the {Level} 3 {BLAS}},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {16},
  number  = {4},
  pages   = {352--368},
  year    = {1990},
  doi     = {10.1145/98267.98290}
}
```

### alman-williams ✅ (⏳ strony SODA)
**J. Alman, V. Vassilevska Williams, „A Refined Laser Method and Faster Matrix Multiplication", w: Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms (SODA), s. 522–539, 2021.**
- DOI: 10.1137/1.9781611976465.32 ; arXiv:2010.05846
- Zweryfikowano (arXiv, SIAM Epubs): ω < 2,3728596 (SODA 2021). ⏳ Potwierdzić dokładny zakres stron w materiałach SODA.
- Użyte: §3.5 (ω < 2,373; algorytmy galaktyczne — niepraktyczne).
```bibtex
@inproceedings{alman2021refined,
  author    = {Josh Alman and Virginia Vassilevska Williams},
  title     = {A Refined Laser Method and Faster Matrix Multiplication},
  booktitle = {Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms (SODA)},
  pages     = {522--539},
  year      = {2021},
  doi       = {10.1137/1.9781611976465.32}
}
```

### goodfellow-dl ✅
**I. Goodfellow, Y. Bengio, A. Courville, „Deep Learning", The MIT Press, Cambridge, MA, 2016.**
- ISBN: 978-0-262-03561-3; seria: Adaptive Computation and Machine Learning; pełna wersja: https://www.deeplearningbook.org/
- Zweryfikowano (MIT Press oraz rozdz. 9 „Convolutional Networks" odczytany z PDF): definicja splotu 1D (równ. 9.3), 2D (równ. 9.4) i przemienność (równ. 9.5); korelacja krzyżowa jako splot bez odbicia jądra (równ. 9.6); „Many machine learning libraries implement cross-correlation but call it convolution"; odbicie zbędne, bo wagi są uczone; pojęcia wejście/jądro/mapa cech; parametry krok i wypełnienie. Definicję splot/korelacja potwierdza niezależnie Oppenheim & Schafer oraz hasło encyklopedyczne „Convolution".
- Użyte: §4.1 (definicja splotu 1D/2D; przemienność i odbicie jądra; splot vs korelacja; „splot" w CNN = korelacja; wejście/jądro/mapa cech; krok i wypełnienie; CNN jako zastosowanie).
```bibtex
@book{goodfellow2016deeplearning,
  author    = {Ian Goodfellow and Yoshua Bengio and Aaron Courville},
  title     = {Deep Learning},
  publisher = {The MIT Press},
  address   = {Cambridge, MA},
  year      = {2016},
  series    = {Adaptive Computation and Machine Learning},
  isbn      = {978-0-262-03561-3},
  url       = {https://www.deeplearningbook.org/}
}
```

### oppenheim-dsp ✅ (⏳ numer strony przy cytowaniu konkretnym)
**A. V. Oppenheim, R. W. Schafer, „Discrete-Time Signal Processing", wyd. 3, Pearson, Upper Saddle River, NJ, 2010.**
- ISBN: 978-0-13-198842-2; seria: Prentice Hall Signal Processing.
- Zweryfikowano (Pearson, zgodny ISBN/rok/wydawca): klasyczny podręcznik DSP; splot dyskretny $y[n]=\sum_k x[k]h[n-k]$; splot jako odpowiedź układu liniowego niezmienniczego w czasie (LTI) przez odpowiedź impulsową; korelacja jako operacja pokrewna bez odbicia. Definicję potwierdza niezależnie Goodfellow i in. 2016.
- ⏳ Numer strony definicji splotu (typowo rozdz. 2) do uzupełnienia przy cytowaniu konkretnej strony.
- Użyte: §4.1 (splot dyskretny — postać ogólna; filtracja sygnałów/obrazów jako odpowiedź LTI; rozróżnienie splot/korelacja); §4.4.3 (twierdzenie o splocie — splot przestrzenny = mnożenie punktowe w dziedzinie częstotliwości; schemat FFT → mnożenie → odwrotna FFT).
```bibtex
@book{oppenheim2010dtsp,
  author    = {Alan V. Oppenheim and Ronald W. Schafer},
  title     = {Discrete-Time Signal Processing},
  edition   = {3},
  publisher = {Pearson},
  address   = {Upper Saddle River, NJ},
  year      = {2010},
  isbn      = {978-0-13-198842-2}
}
```

### vasudevan-mcmk ✅
**A. Vasudevan, A. Anderson, D. Gregg, „Parallel Multi Channel Convolution using General Matrix Multiplication", w: Proceedings of the 2017 IEEE 28th International Conference on Application-specific Systems, Architectures and Processors (ASAP), s. 19–24, 2017.**
- DOI: 10.1109/ASAP.2017.7995254; arXiv:1704.04428
- Zweryfikowano (IEEE, Semantic Scholar, arXiv; tekst PDF): splot wielokanałowy jako suma korelacji jednokanałowych po kanałach wejścia; wejście $C\times H\times W$, $M$ jąder o $C$ kanałach, wynik $M\times H'\times W'$; dla kroku 1 i braku wypełnienia wynik mniejszy o rozmiar jądra ($O=$ wymiar $-$ rozmiar jądra $+1$); warstwy splotowe stanowią większość obliczeń w CNN; im2col jako standardowe sprowadzenie splotu do GEMM (przyda się też w §4.4).
- Użyte: §4.1 (wzór splotu wielokanałowego — suma po kanałach; splot bez wypełnienia, krok 1, $O_H=H-K_H+1$; splot jako dominujący koszt obliczeń w CNN); §4.2 (struktura mnożenia z akumulacją; liczba działań splotu $2\cdot Cout\cdot OH\cdot OW\cdot Cin\cdot KH\cdot KW$); §4.4.2 (im2col → GEMM: macierz rozwinięta — w pracy, ang. unrolled; u Vasudevana "column matrix"/"patch matrix" (NIE "lowered") — filtry w wierszach; narzut pamięci do KH·KW razy / powielenie danych; splot 1×1 = GEMM bez im2col).
```bibtex
@inproceedings{vasudevan2017mcmk,
  author    = {Aravind Vasudevan and Andrew Anderson and David Gregg},
  title     = {Parallel Multi Channel Convolution using General Matrix Multiplication},
  booktitle = {Proceedings of the 2017 IEEE 28th International Conference on Application-specific Systems, Architectures and Processors (ASAP)},
  pages     = {19--24},
  year      = {2017},
  doi       = {10.1109/ASAP.2017.7995254}
}
```

### georganas-conv ✅ (⏳ zakres stron)
**E. Georganas, S. Avancha, K. Banerjee, D. Kalamkar, G. Henry, H. Pabst, A. Heinecke, „Anatomy of High-Performance Deep Learning Convolutions on SIMD Architectures", w: Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC '18), IEEE, 2018.**
- DOI: 10.1109/SC.2018.00069 ; arXiv:1808.05567
- Zweryfikowano (ACM DL, dblp, arXiv; tekst PDF): splot bezpośredni na architekturach SIMD (x86); blokowanie rejestrowe i blokowanie pamięci podręcznej dla maksymalizacji reużycia danych; warstwy z oknem 3×3 mają „substantially higher operational intensity […] and therefore achieve close to compute peak performance"; analiza modelem roofline; warstwy 1×1 o niższej intensywności bywają ograniczone przepustowością. Teza o reużyciu potwierdzona niezależnie u Vasudevana i in. 2017 oraz Williamsa i in. 2009.
- ⏳ Potwierdzić zakres stron (podawane 830–841) oraz pisownię imion autorów w oficjalnych materiałach IEEE/ACM.
- Użyte: §4.2 (splot compute-bound; wysoka intensywność przy oknie 3×3 i wielu kanałach; reużycie wejścia/wag; analiza roofline). §4.3 (układ blokowany NCHWc — blokowanie kanałów czynnikiem VLEN równym szerokości rejestru SIMD; najgłębszy ciągły wymiar mapowany na rejestr); §4.4.1 (splot bezpośredni — sześć zagnieżdżonych pętli, obliczenie referencyjne, zależność lokalności od kolejności pętli i wektoryzacji).
```bibtex
@inproceedings{georganas2018conv,
  author    = {Evangelos Georganas and Sasikanth Avancha and Kunal Banerjee and Dhiraj Kalamkar and Greg Henry and Hans Pabst and Alexander Heinecke},
  title     = {Anatomy of High-Performance Deep Learning Convolutions on {SIMD} Architectures},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC)},
  year      = {2018},
  publisher = {IEEE},
  doi       = {10.1109/SC.2018.00069}
}
```

### onednn-formats ✅ (⏳ wersja/data dostępu)
**oneDNN, „Understanding Memory Formats", dokumentacja oneDNN (oneAPI Deep Neural Network Library).**
- URL: https://uxlfoundation.github.io/oneDNN/dev_guide_understanding_memory_formats.html
- Zweryfikowano (treść strony, dwie niezależne kopie): formaty „plain" NCHW i NHWC z jawnymi funkcjami przesunięcia — `offset_nchw(n,c,h,w)=n*CHW+c*HW+h*W+w` (równoważnie `((n*C+c)*H+h)*W+w`), `offset_nhwc(n,c,h,w)=n*HWC+h*WC+w*C+c` (NHWC domyślny w TensorFlow); format blokowany dzieli wymiar(y) na bloki stałego rozmiaru „to achieve better vectorization and cache reuse"; warianty nChw16c (AVX-512) i nChw8c (SSE4.1+/AVX2) blokują wymiar kanałów (blok 16 lub 8), trzymając „blocks of channels contiguously in memory".
- ⏳ Podać numer wersji oneDNN i datę dostępu.
- Użyte: §4.3 (definicje układów NCHW/NHWC; układ blokowany NCHWc — blok kanałów ciągły, dobór rozmiaru bloku pod wektoryzację i reużycie pamięci podręcznej).
```bibtex
@misc{onednn_memory_formats,
  author       = {{oneDNN}},
  title        = {Understanding Memory Formats},
  howpublished = {dokumentacja oneDNN (oneAPI Deep Neural Network Library)},
  note         = {podać wersję i datę dostępu},
  url          = {https://uxlfoundation.github.io/oneDNN/dev_guide_understanding_memory_formats.html}
}
```

### chellapilla-im2col ✅ (⏳ strony)
**K. Chellapilla, S. Puri, P. Simard, „High Performance Convolutional Neural Networks for Document Processing", w: Tenth International Workshop on Frontiers in Handwriting Recognition (IWFHR), Suvisoft, La Baule, Francja, 2006.**
- HAL: inria-00112631 ; URL: https://hal.science/inria-00112631
- Zweryfikowano (HAL/Inria, Semantic Scholar oraz lista referencji u Vasudevana i in. 2017): pierwsze sprowadzenie splotu wielokanałowego do pojedynczego mnożenia macierzy (im2col + GEMM/BLAS); okna wejścia rozwijane (ang. unrolling) w kolumny macierzy rozwiniętej, filtry w wiersze; przyspieszenie dzięki wykorzystaniu zoptymalizowanego BLAS.
- ⏳ Potwierdzić numery stron (materiały warsztatu).
- Użyte: §4.4.2 (im2col — okna → kolumny, filtry → wiersze; splot = jeden GEMM; wykorzystanie zoptymalizowanego BLAS; narzut pamięci powielenia danych).
```bibtex
@inproceedings{chellapilla2006highperf,
  author    = {Kumar Chellapilla and Sidd Puri and Patrice Simard},
  title     = {High Performance Convolutional Neural Networks for Document Processing},
  booktitle = {Tenth International Workshop on Frontiers in Handwriting Recognition (IWFHR)},
  publisher = {Suvisoft},
  address   = {La Baule, France},
  year      = {2006},
  url       = {https://hal.science/inria-00112631}
}
```

### lavin-gray-winograd ✅ (⏳ strony CVPR)
**A. Lavin, S. Gray, „Fast Algorithms for Convolutional Neural Networks", w: Proceedings of the 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), s. 4013–4021, 2016.**
- DOI: 10.1109/CVPR.2016.435 ; arXiv:1509.09308
- Zweryfikowano (CVPR 2016, arXiv): algorytmy minimalnego filtrowania (Winograd) dla małych jąder w CNN; FFT wymaga większych bloków i większego narzutu transformat, przez co dla małych jąder (3×3) jest mniej korzystny niż Winograd; redukcja liczby mnożeń względem splotu bezpośredniego.
- ⏳ Potwierdzić dokładny zakres stron w materiałach CVPR.
- Użyte: §4.4.3 (FFT korzystny dla dużych jąder, nieopłacalny dla małych okien 3×3); §4.5 (algorytm minimalnego filtrowania F(2×2,3×3): wzór Y=Aᵀ[(GgGᵀ)⊙(BᵀdB)]A; macierze Bᵀ/G/Aᵀ; redukcja 36→16 mnożeń = 2,25×; sumowanie po kanałach → pakiet 16 GEMM; kompromisy: dokładność i narzut transformat rosnące z rozmiarem kafelka → ograniczenie do jąder 3×3).
```bibtex
@inproceedings{lavin2016fast,
  author    = {Andrew Lavin and Scott Gray},
  title     = {Fast Algorithms for Convolutional Neural Networks},
  booktitle = {Proceedings of the 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4013--4021},
  year      = {2016},
  doi       = {10.1109/CVPR.2016.435}
}
```

### libxsmm ✅
**A. Heinecke, G. Henry, M. Hutchinson, H. Pabst, „LIBXSMM: Accelerating Small Matrix Multiplications by Runtime Code Generation", w: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '16), s. 981–991, IEEE, 2016.**
- DOI: 10.1109/SC.2016.83
- Zweryfikowano (Semantic Scholar, dblp oraz oficjalne repozytorium libxsmm/libxsmm): autorzy Alexander Heinecke, Greg Henry, Maxwell Hutchinson, Hans Pabst (UWAGA: trzeci autor to **Maxwell Hutchinson**, nie „M. Greutter" jak w roboczych metadanych); SC '16, Salt Lake City, s. 981–991; biblioteka małych, gęstych/rzadkich mnożeń macierzy oraz prymitywów uczenia głębokiego (małe sploty), generująca kod w czasie wykonania (JIT) pod rozszerzenia wektorowe x86 (do AVX-512/AMX); celuje w rozmiary $(M\cdot N\cdot K)^{1/3} \le 64$.
- Użyte: §5.6 (biblioteka referencyjna splotu — splot realizowany przez im2col + SGEMM funkcją \texttt{libxsmm\_sgemm}; biblioteka z generacją kodu w czasie wykonania dla małych mnożeń macierzy).
```bibtex
@inproceedings{heinecke2016libxsmm,
  author    = {Alexander Heinecke and Greg Henry and Maxwell Hutchinson and Hans Pabst},
  title     = {{LIBXSMM}: Accelerating Small Matrix Multiplications by Runtime Code Generation},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)},
  pages     = {981--991},
  year      = {2016},
  publisher = {IEEE},
  doi       = {10.1109/SC.2016.83}
}
```

### winograd-arithmetic ✅
**S. Winograd, „Arithmetic Complexity of Computations", CBMS-NSF Regional Conference Series in Applied Mathematics, nr 33, Society for Industrial and Applied Mathematics (SIAM), Filadelfia, 1980.**
- DOI: 10.1137/1.9781611970364 ; ISBN: 978-0-89871-163-9.
- Zweryfikowano (SIAM Publications Library, Internet Archive, rekord biblioteczny UPenn, Semantic Scholar): seria CBMS-NSF nr 33, SIAM, 1980; rozdz. 5 o filtrach FIR (teoria minimalnego filtrowania). Źródło pierwotne dla minimalnej liczby mnożeń $m+r-1$ oraz algorytmu F(2,3) (4 zamiast 6 mnożeń); przywoływane przez Lavina i Graya 2016.
- Użyte: §4.5 (teoria minimalnego filtrowania; minimalna liczba mnożeń F(m,r)=m+r−1; F(2,3) z 4 mnożeniami zamiast 6; oszczędność mnożeń kosztem dodawań).
```bibtex
@book{winograd1980arithmetic,
  author    = {Shmuel Winograd},
  title     = {Arithmetic Complexity of Computations},
  series    = {CBMS-NSF Regional Conference Series in Applied Mathematics},
  number    = {33},
  publisher = {Society for Industrial and Applied Mathematics (SIAM)},
  address   = {Philadelphia, PA},
  year      = {1980},
  isbn      = {978-0-89871-163-9},
  doi       = {10.1137/1.9781611970364}
}
```

---

## Kandydaci do dopisania (po weryfikacji, przy użyciu w tekście)
Z listy referencji Twojego artykułu — dopiszę je tutaj dopiero, gdy faktycznie
ich użyję i zweryfikuję metadane: Agner Fog (tablice instrukcji; mikroarchitektura),
AMD Software Optimization Guide (rodzina 19h / Zen 3), P. Gepner (AVX2),
H. Zhang i in. (FMA), Hennessy & Patterson (architektura komputerów),
Frigo i in. (algorytmy cache-oblivious).
