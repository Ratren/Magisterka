#include "common.h"
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void* xsmm_handle = NULL;

static void exe_dir(char* out, size_t out_size) {
    ssize_t n = readlink("/proc/self/exe", out, out_size - 1);
    if (n <= 0) { out[0] = 0; return; }
    out[n] = 0;
    char* slash = strrchr(out, '/');
    if (slash) *slash = 0;
}

int libxsmm_loader_init(void) {
    if (xsmm_handle) return 1;

    char dir[PATH_MAX];
    char buf[PATH_MAX];
    exe_dir(dir, sizeof(dir));

    /* First try the install.sh location relative to the binary. */
    const char* rel_from_exe[] = {
        "/../../libxsmm_install/lib/libxsmm.so",
        "/../../../libxsmm_install/lib/libxsmm.so",
        NULL
    };
    for (int i = 0; rel_from_exe[i] && dir[0]; i++) {
        snprintf(buf, sizeof(buf), "%s%s", dir, rel_from_exe[i]);
        xsmm_handle = dlopen(buf, RTLD_NOW | RTLD_LOCAL);
        if (xsmm_handle) break;
    }

    if (!xsmm_handle) {
        const char* fallback[] = {
            "libxsmm.so",
            "libxsmm.so.1",
            "/usr/local/lib/libxsmm.so",
            "/usr/lib/libxsmm.so",
            NULL
        };
        for (int i = 0; fallback[i]; i++) {
            xsmm_handle = dlopen(fallback[i], RTLD_NOW | RTLD_LOCAL);
            if (xsmm_handle) break;
        }
    }

    if (!xsmm_handle) return 0;

    /* libxsmm_init() initialises the JIT engine; safe to call multiple times. */
    typedef void (*init_fn)(void);
    init_fn init = (init_fn)dlsym(xsmm_handle, "libxsmm_init");
    if (init) init();
    return 1;
}

void libxsmm_loader_shutdown(void) {
    if (xsmm_handle) {
        typedef void (*fini_fn)(void);
        fini_fn fini = (fini_fn)dlsym(xsmm_handle, "libxsmm_finalize");
        if (fini) fini();
        dlclose(xsmm_handle);
        xsmm_handle = NULL;
    }
}

void* libxsmm_loader_sym(const char* name) {
    if (!xsmm_handle) return NULL;
    return dlsym(xsmm_handle, name);
}

int libxsmm_loader_ok(void) {
    return xsmm_handle != NULL;
}
