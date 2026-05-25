#include "common.h"
#include <stdio.h>
#include <string.h>

static void escape_json(const char* in, char* out, size_t out_size) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 2 < out_size; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"' || c == '\\') {
            if (j + 3 >= out_size) break;
            out[j++] = '\\'; out[j++] = c;
        } else if (c < 0x20) {
            continue;
        } else {
            out[j++] = c;
        }
    }
    out[j] = 0;
}

void json_write_results(const char* path, const char* kernel,
                        int threads, int runs,
                        const PresetResult* presets, int num_presets) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", path); return; }

    char cpu_name[256], cpu_esc[512];
    read_cpu_name(cpu_name, sizeof(cpu_name));
    escape_json(cpu_name, cpu_esc, sizeof(cpu_esc));

    fprintf(f, "{\n");
    fprintf(f, "  \"kernel\": \"%s\",\n", kernel);
    fprintf(f, "  \"cpu\": \"%s\",\n", cpu_esc);
    fprintf(f, "  \"threads\": %d,\n", threads);
    fprintf(f, "  \"runs\": %d,\n", runs);
    fprintf(f, "  \"presets\": [\n");

    for (int p = 0; p < num_presets; p++) {
        const PresetResult* pr = &presets[p];
        fprintf(f, "    {\n");
        fprintf(f, "      \"name\": \"%s\",\n", pr->name);
        if (pr->params_json[0]) {
            fprintf(f, "      \"params\": {%s},\n", pr->params_json);
        }
        fprintf(f, "      \"implementations\": {\n");
        for (int i = 0; i < pr->num_impls; i++) {
            const ImplResult* ir = &pr->impls[i];
            fprintf(f,
                "        \"%s\": {\"median\": %.4f, \"min\": %.4f, \"max\": %.4f, \"stddev\": %.4f, \"max_error\": %.3e}%s\n",
                ir->name, ir->median, ir->min, ir->max, ir->stddev, ir->max_error,
                i < pr->num_impls - 1 ? "," : "");
        }
        fprintf(f, "      }\n");
        fprintf(f, "    }%s\n", p < num_presets - 1 ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}
