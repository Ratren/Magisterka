#include "common.h"
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void* blis_handle = NULL;

typedef void (*blis_set_threads_func)(int);

static void exe_dir(char* out, size_t out_size) {
    ssize_t n = readlink("/proc/self/exe", out, out_size - 1);
    if (n <= 0) { out[0] = 0; return; }
    out[n] = 0;
    char* slash = strrchr(out, '/');
    if (slash) *slash = 0;
}

int blis_loader_init(int nthreads) {
    char dir[PATH_MAX];
    char buf[PATH_MAX];
    exe_dir(dir, sizeof(dir));

    const char* rel_from_exe[] = {
        "/../../blis_install/lib/libblis-mt.so",
        "/../../../blis_install/lib/libblis-mt.so",
        NULL
    };
    for (int i = 0; rel_from_exe[i] && dir[0]; i++) {
        snprintf(buf, sizeof(buf), "%s%s", dir, rel_from_exe[i]);
        blis_handle = dlopen(buf, RTLD_NOW | RTLD_LOCAL);
        if (blis_handle) break;
    }

    if (!blis_handle) {
        const char* fallback[] = {
            "libblis-mt.so",
            "/usr/local/lib/libblis-mt.so",
            NULL
        };
        for (int i = 0; fallback[i]; i++) {
            blis_handle = dlopen(fallback[i], RTLD_NOW | RTLD_LOCAL);
            if (blis_handle) break;
        }
    }

    if (!blis_handle) {
        fprintf(stderr, "AOCL-BLAS not found: %s\n", dlerror());
        return 0;
    }
    blis_set_threads_func set_threads =
        (blis_set_threads_func)dlsym(blis_handle, "bli_thread_set_num_threads");
    if (set_threads) set_threads(nthreads);
    return 1;
}

void blis_loader_shutdown(void) {
    if (blis_handle) { dlclose(blis_handle); blis_handle = NULL; }
}

void* blis_loader_sym(const char* name) {
    if (!blis_handle) return NULL;
    return dlsym(blis_handle, name);
}

int blis_loader_ok(void) {
    return blis_handle != NULL;
}
