#include <stdio.h>
#include <stdlib.h>

static void read_cache_size(const char* level_desc, int index) {
    char path[128], size_str[32];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    FILE* fp = fopen(path, "r");
    if (!fp) { perror("Failed to read cache size"); return; }
    if (fgets(size_str, sizeof(size_str), fp)) {
        printf("%s Cache Size: %s", level_desc, size_str);
    }
    fclose(fp);
}

int main(void) {
    // index 0/1 = L1d/L1i, 2 = L2, 3 = L3 on this CPU layout.
    read_cache_size("L1d", 0);
    read_cache_size("L1i", 1);
    read_cache_size("L2",  2);
    read_cache_size("L3",  3);
    return 0;
}
