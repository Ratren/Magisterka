#include "common.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* dup_trim(const char* s, size_t n) {
    while (n && isspace((unsigned char)*s)) { s++; n--; }
    while (n && isspace((unsigned char)s[n - 1])) n--;
    char* r = (char*)malloc(n + 1);
    memcpy(r, s, n);
    r[n] = 0;
    return r;
}

Preset* preset_load(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "preset: cannot open %s\n", path); return NULL; }

    Preset *head = NULL, *tail = NULL, *cur = NULL;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        char* p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == 0 || *p == '#' || *p == ';') continue;

        if (*p == '[') {
            char* end = strchr(p, ']');
            if (!end) continue;
            Preset* np = (Preset*)calloc(1, sizeof(Preset));
            np->name = dup_trim(p + 1, end - p - 1);
            if (!head) head = np; else tail->next = np;
            tail = np;
            cur = np;
            continue;
        }

        if (!cur) continue;
        char* eq = strchr(p, '=');
        if (!eq) continue;
        PresetKV* kv = (PresetKV*)calloc(1, sizeof(PresetKV));
        kv->key = dup_trim(p, eq - p);
        char* v = eq + 1;
        kv->value = dup_trim(v, strlen(v));
        kv->next = cur->kvs;
        cur->kvs = kv;
    }
    fclose(fp);
    return head;
}

void preset_free(Preset* head) {
    while (head) {
        Preset* next = head->next;
        PresetKV* kv = head->kvs;
        while (kv) {
            PresetKV* knext = kv->next;
            free(kv->key); free(kv->value); free(kv);
            kv = knext;
        }
        free(head->name); free(head);
        head = next;
    }
}

const char* preset_get(const Preset* p, const char* key) {
    for (PresetKV* kv = p->kvs; kv; kv = kv->next) {
        if (strcmp(kv->key, key) == 0) return kv->value;
    }
    return NULL;
}

long preset_get_long(const Preset* p, const char* key, long def) {
    const char* v = preset_get(p, key);
    return v ? strtol(v, NULL, 10) : def;
}

int preset_get_int(const Preset* p, const char* key, int def) {
    return (int)preset_get_long(p, key, def);
}

int preset_count(const Preset* head) {
    int n = 0;
    for (const Preset* p = head; p; p = p->next) n++;
    return n;
}
