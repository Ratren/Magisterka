#include "common.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void read_cpu_name(char* buf, size_t buf_size) {
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (!fp) { strncpy(buf, "Unknown CPU", buf_size); return; }
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "model name", 10) == 0) {
            char* colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ' || *colon == '\t') colon++;
                colon[strcspn(colon, "\n")] = 0;
                strncpy(buf, colon, buf_size - 1);
                buf[buf_size - 1] = 0;
                fclose(fp);
                return;
            }
        }
    }
    fclose(fp);
    strncpy(buf, "Unknown CPU", buf_size);
}

int read_cache_size_bytes(int index) {
    char path[128], size_str[32];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    FILE* fp = fopen(path, "r");
    if (!fp) return -1;
    if (!fgets(size_str, sizeof(size_str), fp)) { fclose(fp); return -1; }
    fclose(fp);
    size_str[strcspn(size_str, "\n")] = 0;
    long size = strtol(size_str, NULL, 10);
    char unit = size_str[strlen(size_str) - 1];
    switch (toupper(unit)) {
        case 'K': size *= 1024; break;
        case 'M': size *= 1024 * 1024; break;
    }
    return (int)size;
}

double read_cpu_freq_mhz(void) {
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
    if (fp) {
        char buf[32];
        if (fgets(buf, sizeof(buf), fp)) { fclose(fp); return atof(buf) / 1000.0; }
        fclose(fp);
    }
    fp = fopen("/proc/cpuinfo", "r");
    if (!fp) return -1;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "cpu MHz", 7) == 0) {
            char* colon = strchr(line, ':');
            if (colon) { fclose(fp); return atof(colon + 1); }
        }
    }
    fclose(fp);
    return -1;
}

int read_boost_state(void) {
    FILE* fp = fopen("/sys/devices/system/cpu/cpufreq/boost", "r");
    if (!fp) return -1;
    char buf[8];
    int v = -1;
    if (fgets(buf, sizeof(buf), fp)) v = atoi(buf);
    fclose(fp);
    return v;
}

void print_system_header(const char* title) {
    char cpu_name[256];
    read_cpu_name(cpu_name, sizeof(cpu_name));
    int l1 = read_cache_size_bytes(0);
    int l2 = read_cache_size_bytes(2);
    int l3 = read_cache_size_bytes(3);
    double freq = read_cpu_freq_mhz();
    int boost = read_boost_state();

    printf("================================================================\n");
    printf("              %s - %s\n", title, cpu_name);
    printf("================================================================\n");
    if (l1 > 0 && l2 > 0 && l3 > 0) {
        printf("L1: %d KB | L2: %d KB | L3: %d MB\n",
               l1 / 1024, l2 / 1024, l3 / (1024 * 1024));
    }
    if (freq > 0) {
        printf("CPU freq: %.0f MHz | Boost: %s\n", freq,
               boost < 0 ? "unknown" : (boost ? "ON" : "OFF"));
    }
    printf("Clock: CLOCK_MONOTONIC_RAW | Flags: -O3 -march=znver3 -mavx2 -mfma -flto\n");
    printf("================================================================\n\n");
}
