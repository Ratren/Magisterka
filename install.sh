#!/usr/bin/env bash
# Build script for the Magisterka kernel benchmarks.
#
# What it does:
#   1. Checks host tooling (gcc, cmake, make, python3, git).
#   2. Builds OpenBLAS from source into Kod/openblas_install (required — the
#      benchmark executables link against it at compile time).
#   3. Optionally builds BLIS from source into Kod/blis_install (loaded at
#      runtime via dlopen; only needed for the BLIS comparison columns).
#   4. Optionally builds libxsmm from source into Kod/libxsmm_install (loaded
#      at runtime via dlopen; used by conv as the correctness reference and
#      as the "libxsmm" benchmark row).
#   5. Configures and builds each kernel module (dot_product, gemv, gemm, conv)
#      with CMake into Kod/<module>/build/benchmark.
#
# Usage:
#   ./install.sh [--with-blis] [--with-libxsmm]
#                [--skip-openblas] [--skip-blis] [--skip-libxsmm]
#                [--jobs N] [--openblas-tag TAG] [--blis-tag TAG] [--libxsmm-tag TAG]
#
# Defaults: build OpenBLAS, skip BLIS, skip libxsmm, use $(nproc) jobs.

set -euo pipefail

OPENBLAS_TAG="v0.3.31"
BLIS_TAG="5.2.0"
LIBXSMM_TAG="main"
WITH_BLIS=0
WITH_LIBXSMM=0
SKIP_OPENBLAS=0
SKIP_BLIS=0
SKIP_LIBXSMM=0
JOBS="$(nproc 2>/dev/null || echo 4)"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
KOD_DIR="$ROOT_DIR/Kod"
OPENBLAS_SRC="$KOD_DIR/openblas_src"
OPENBLAS_INSTALL="$KOD_DIR/openblas_install"
BLIS_SRC="$KOD_DIR/blis_src"
BLIS_INSTALL="$KOD_DIR/blis_install"
LIBXSMM_SRC="$KOD_DIR/libxsmm_src"
LIBXSMM_INSTALL="$KOD_DIR/libxsmm_install"

KERNELS=(dot_product gemv gemm conv)

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,18p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-blis)       WITH_BLIS=1 ;;
        --with-libxsmm)    WITH_LIBXSMM=1 ;;
        --skip-openblas)   SKIP_OPENBLAS=1 ;;
        --skip-blis)       SKIP_BLIS=1 ;;
        --skip-libxsmm)    SKIP_LIBXSMM=1 ;;
        --jobs)            JOBS="${2:?--jobs needs a value}"; shift ;;
        --jobs=*)          JOBS="${1#*=}" ;;
        --openblas-tag)    OPENBLAS_TAG="${2:?}"; shift ;;
        --openblas-tag=*)  OPENBLAS_TAG="${1#*=}" ;;
        --blis-tag)        BLIS_TAG="${2:?}"; shift ;;
        --blis-tag=*)      BLIS_TAG="${1#*=}" ;;
        --libxsmm-tag)     LIBXSMM_TAG="${2:?}"; shift ;;
        --libxsmm-tag=*)   LIBXSMM_TAG="${1#*=}" ;;
        -h|--help)         usage ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
    shift
done

check_tools() {
    log "Checking host tools…"
    local missing=()
    for t in gcc cmake make git python3; do
        command -v "$t" >/dev/null 2>&1 || missing+=("$t")
    done
    if (( ${#missing[@]} )); then
        die "missing required tools: ${missing[*]} (install with apt/dnf/pacman)"
    fi
    log "  gcc:    $(gcc --version | head -n1)"
    log "  cmake:  $(cmake --version | head -n1)"
    log "  python: $(python3 --version)"

    if ! gcc -march=znver3 -E -x c /dev/null >/dev/null 2>&1; then
        warn "this gcc does not understand -march=znver3."
        warn "edit Kod/<module>/CMakeLists.txt and Kod/common/CMakeLists.txt to use -march=native"
        warn "(or your CPU's arch) if the build fails."
    fi
}

clone_or_update() {
    local url="$1" dest="$2" tag="$3"
    if [[ -d "$dest/.git" ]]; then
        log "  refreshing $(basename "$dest") (existing checkout)"
        git -C "$dest" fetch --tags --quiet
    else
        log "  cloning $url -> $dest"
        git clone --quiet "$url" "$dest"
    fi
    git -C "$dest" checkout --quiet "$tag"
}

build_openblas() {
    if (( SKIP_OPENBLAS )); then
        log "Skipping OpenBLAS build (--skip-openblas)."
        [[ -f "$OPENBLAS_INSTALL/lib/libopenblas.so" || \
           -f "$OPENBLAS_INSTALL/lib/libopenblas.a" ]] || \
            die "OpenBLAS install missing at $OPENBLAS_INSTALL/lib — drop --skip-openblas"
        return
    fi

    log "Building OpenBLAS $OPENBLAS_TAG -> $OPENBLAS_INSTALL"
    mkdir -p "$KOD_DIR"
    clone_or_update https://github.com/OpenMathLib/OpenBLAS.git "$OPENBLAS_SRC" "$OPENBLAS_TAG"

    # Wipe stale .o/.a from a previous build — LTO bytecode is GCC-version
    # specific, so any pre-existing artefact will break the final link when
    # the host gcc has been upgraded since.
    make -C "$OPENBLAS_SRC" clean >/dev/null 2>&1 || true

    make -C "$OPENBLAS_SRC" -j"$JOBS" \
         USE_OPENMP=1 NUM_THREADS=64 DYNAMIC_ARCH=0 TARGET=ZEN \
         CFLAGS="-O3" >/dev/null
    make -C "$OPENBLAS_SRC" PREFIX="$OPENBLAS_INSTALL" install >/dev/null
    log "  OpenBLAS installed."
}

build_blis() {
    if (( ! WITH_BLIS )) || (( SKIP_BLIS )); then
        log "Skipping BLIS (pass --with-blis to include it)."
        return
    fi

    log "Building BLIS $BLIS_TAG -> $BLIS_INSTALL"
    mkdir -p "$KOD_DIR"
    clone_or_update https://github.com/amd/blis.git "$BLIS_SRC" "$BLIS_TAG"

    pushd "$BLIS_SRC" >/dev/null
    make clean >/dev/null 2>&1 || true
    ./configure --prefix="$BLIS_INSTALL" --enable-threading=openmp \
                --enable-cblas --enable-shared zen3 >/dev/null
    make -j"$JOBS" >/dev/null
    make install >/dev/null
    popd >/dev/null
    log "  BLIS installed (libblis-mt.so at $BLIS_INSTALL/lib)."
}

build_libxsmm() {
    if (( ! WITH_LIBXSMM )) || (( SKIP_LIBXSMM )); then
        log "Skipping libxsmm (pass --with-libxsmm to include it)."
        return
    fi

    log "Building libxsmm $LIBXSMM_TAG -> $LIBXSMM_INSTALL"
    mkdir -p "$KOD_DIR"
    clone_or_update https://github.com/libxsmm/libxsmm.git "$LIBXSMM_SRC" "$LIBXSMM_TAG"

    # libxsmm has no configure step -- AVX2 codegen is selected at JIT time.
    # STATIC=0 builds the shared object; the conv benchmark dlopens it.
    make -C "$LIBXSMM_SRC" clean >/dev/null 2>&1 || true
    make -C "$LIBXSMM_SRC" -j"$JOBS" STATIC=0 >/dev/null
    mkdir -p "$LIBXSMM_INSTALL/lib"
    cp -p "$LIBXSMM_SRC"/lib/libxsmm.so* "$LIBXSMM_INSTALL/lib/"
    log "  libxsmm installed (libxsmm.so at $LIBXSMM_INSTALL/lib)."
}

build_kernels() {
    for k in "${KERNELS[@]}"; do
        local src="$KOD_DIR/$k"
        local build="$src/build"
        log "Configuring $k"
        cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release >/dev/null
        log "Building $k"
        cmake --build "$build" -j"$JOBS" >/dev/null
        [[ -x "$build/benchmark" ]] || die "$k benchmark binary not produced"
    done
    log "All kernels built."
}

main() {
    check_tools
    build_openblas
    build_blis
    build_libxsmm
    build_kernels

    cat <<EOF

==========================================================================
 Build complete.

   Per-kernel binaries:
     Kod/dot_product/build/benchmark
     Kod/gemv/build/benchmark
     Kod/gemm/build/benchmark
     Kod/conv/build/benchmark

   Quick run:
     cd Kod/gemv && OMP_NUM_THREADS=1 taskset -c 0 ./build/benchmark

   Full suite (Python harness, both 1-thread and N-thread sweeps):
     python3 Kod/benchmark_suite/run_suite.py

EOF
    if (( WITH_BLIS )); then
        cat <<EOF
   BLIS was installed. The benchmark binaries discover it via dlopen from
   Kod/blis_install/lib at runtime — no extra LD_LIBRARY_PATH is needed
   when launching from a kernel build/ directory.

EOF
    fi
    if (( WITH_LIBXSMM )); then
        cat <<EOF
   libxsmm was installed. The conv benchmark dlopens libxsmm.so from
   Kod/libxsmm_install/lib at runtime and uses it both as the
   "libxsmm" benchmark entry and as the correctness reference.

EOF
    fi
}

main "$@"
