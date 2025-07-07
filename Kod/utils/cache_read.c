#include <stdio.h>
#include <stdlib.h>

void read_cache_size(const char *level_desc, int index) {
    char path[128];
    char size_str[32];

    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    FILE *fp = fopen(path, "r");

    if (fp == NULL) {
        perror("Failed to read cache size");
        return;
    }

    if (fgets(size_str, sizeof(size_str), fp)) {
        printf("%s Cache Size: %s", level_desc, size_str);
    }

    fclose(fp);
}

int main() {
    read_cache_size("L1d", 0); // index0 usually = L1d
    read_cache_size("L1i", 1); // index1 usually = L1i
    read_cache_size("L2", 2);
    read_cache_size("L3", 3);
    return 0;
}
