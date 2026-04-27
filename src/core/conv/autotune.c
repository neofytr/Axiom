/* conv/autotune.c — first-call path calibration for conv forward.

   on the first forward for a new shape, trials each eligible path
   (winograd, im2col+gemm, implicit gemm, etc.) and caches the
   fastest. subsequent calls skip calibration and dispatch directly.

   calibration overhead: 1 extra path trial per shape (~3-5 ms).
   results persist in memory (process lifetime) and on disk
   (XDG_CACHE_HOME/axiom/conv_calib.bin) across runs.

   disabled under AX_NO_AUTOTUNE=1. overridable per-path with
   AX_CONV_FORCE_PATH={winograd,im2col,implicit,direct,smallcin}. */

#include "internal.h"
#include "axiom/compute.h"
#include "axiom/internal/calib_cache.h"
#include "../../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}

static uint64_t fnv1a64(const void *data, size_t len)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* shape key for cache lookup */
typedef struct {
    int64_t N, C_in, C_out, H, W;
    int     kh, kw, sh, sw, ph, pw;
    int     _pad;
} conv_shape_key_t;

static uint64_t shape_hash(const conv_shape_key_t *k)
{
    return fnv1a64(k, sizeof(*k));
}

/* in-memory cache: small static array with linear scan. number of
   distinct conv shapes per network is typically <20. */
#define MEM_CACHE_CAP 128
typedef struct { uint64_t hash; ax_conv_path_t path; } mem_entry_t;
static mem_entry_t g_mem[MEM_CACHE_CAP];
static int g_mem_n = 0;

static bool mem_lookup(uint64_t h, ax_conv_path_t *out)
{
    for (int i = 0; i < g_mem_n; i++) {
        if (g_mem[i].hash == h) { *out = g_mem[i].path; return true; }
    }
    return false;
}

static void mem_insert(uint64_t h, ax_conv_path_t p)
{
    for (int i = 0; i < g_mem_n; i++) {
        if (g_mem[i].hash == h) { g_mem[i].path = p; return; }
    }
    if (g_mem_n < MEM_CACHE_CAP)
        g_mem[g_mem_n++] = (mem_entry_t){h, p};
}

/* disk cache: conv_calib.bin alongside calib.bin.
   format: "AXCONVCL" + schema(u32) + host_key(u64) + n(u32) + entries */
#define CONV_MAGIC     "AXCONVCL"
#define CONV_MAGIC_LEN 8
#define CONV_SCHEMA    1u
#define DISK_CAP       256

typedef struct { uint64_t hash; uint32_t path; uint32_t _pad; } disk_entry_t;

static uint64_t host_key_cached = 0;
static bool     host_key_ready  = false;

static uint64_t get_host_key(void)
{
    if (!host_key_ready) {
        char fp[64];
        ax_calib_cache_fingerprint(fp, sizeof(fp));
        host_key_cached = fnv1a64(fp, strlen(fp));
        host_key_ready = true;
    }
    return host_key_cached;
}

static bool resolve_conv_path(char *file_path, size_t fp_sz,
                               char *dir_path, size_t dir_sz)
{
    const char *override = getenv("AX_CALIB_CACHE_PATH");
    const char *xdg = getenv("XDG_CACHE_HOME");
    const char *home = getenv("HOME");

    if (override && override[0]) {
        const char *slash = strrchr(override, '/');
        if (slash) {
            size_t dn = (size_t)(slash - override);
            if (dn + 1 > dir_sz) return false;
            memcpy(dir_path, override, dn);
            dir_path[dn] = '\0';
        } else {
            dir_path[0] = '.'; dir_path[1] = '\0';
        }
        return snprintf(file_path, fp_sz, "%s/conv_calib.bin", dir_path) < (int)fp_sz;
    }
    if (xdg && xdg[0]) {
        if (snprintf(dir_path, dir_sz, "%s/axiom", xdg) >= (int)dir_sz) return false;
    } else if (home && home[0]) {
        if (snprintf(dir_path, dir_sz, "%s/.cache/axiom", home) >= (int)dir_sz) return false;
    } else {
        return false;
    }
    return snprintf(file_path, fp_sz, "%s/conv_calib.bin", dir_path) < (int)fp_sz;
}

static int mkdir_p(const char *path)
{
    if (!path || !*path) return -1;
    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return -1;
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i < n; i++) {
        if (buf[i] == '/') {
            buf[i] = '\0';
            if (mkdir(buf, 0700) != 0 && errno != EEXIST) return -1;
            buf[i] = '/';
        }
    }
    if (mkdir(buf, 0700) != 0 && errno != EEXIST) return -1;
    return 0;
}

static bool disk_disabled(void)
{
    const char *e = getenv("AX_NO_CALIB_CACHE");
    return (e && e[0] == '1');
}

static bool disk_lookup(uint64_t sh, ax_conv_path_t *out)
{
    if (disk_disabled()) return false;
    char fp[1024], dp[1024];
    if (!resolve_conv_path(fp, sizeof(fp), dp, sizeof(dp))) return false;

    FILE *f = fopen(fp, "rb");
    if (!f) return false;

    char magic[CONV_MAGIC_LEN];
    uint32_t schema; uint64_t key; uint32_t n;
    bool ok = fread(magic, 1, CONV_MAGIC_LEN, f) == CONV_MAGIC_LEN
           && memcmp(magic, CONV_MAGIC, CONV_MAGIC_LEN) == 0
           && fread(&schema, 4, 1, f) == 1 && schema == CONV_SCHEMA
           && fread(&key, 8, 1, f) == 1 && key == get_host_key()
           && fread(&n, 4, 1, f) == 1 && n <= DISK_CAP;
    if (!ok) { fclose(f); return false; }

    disk_entry_t e;
    for (uint32_t i = 0; i < n; i++) {
        if (fread(&e, sizeof(e), 1, f) != 1) break;
        if (e.hash == sh && e.path < AX_CONV_PATH_COUNT) {
            *out = (ax_conv_path_t)e.path;
            fclose(f);
            return true;
        }
    }
    fclose(f);
    return false;
}

static void disk_save(uint64_t sh, ax_conv_path_t path)
{
    if (disk_disabled()) return;
    char fp[1024], dp[1024];
    if (!resolve_conv_path(fp, sizeof(fp), dp, sizeof(dp))) return;

    /* read existing entries */
    disk_entry_t entries[DISK_CAP];
    uint32_t n = 0;

    FILE *f = fopen(fp, "rb");
    if (f) {
        char magic[CONV_MAGIC_LEN]; uint32_t schema; uint64_t key; uint32_t cnt;
        if (fread(magic, 1, CONV_MAGIC_LEN, f) == CONV_MAGIC_LEN
            && memcmp(magic, CONV_MAGIC, CONV_MAGIC_LEN) == 0
            && fread(&schema, 4, 1, f) == 1 && schema == CONV_SCHEMA
            && fread(&key, 8, 1, f) == 1 && key == get_host_key()
            && fread(&cnt, 4, 1, f) == 1 && cnt <= DISK_CAP)
        {
            for (uint32_t i = 0; i < cnt; i++) {
                if (fread(&entries[n], sizeof(disk_entry_t), 1, f) != 1) break;
                if (entries[n].hash == sh) continue; /* skip old entry for this shape */
                n++;
            }
        }
        fclose(f);
    }

    if (n < DISK_CAP)
        entries[n++] = (disk_entry_t){sh, (uint32_t)path, 0};

    if (mkdir_p(dp) != 0) return;

    char tmp[1100];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", fp) >= (int)sizeof(tmp)) return;
    f = fopen(tmp, "wb");
    if (!f) return;

    uint32_t schema = CONV_SCHEMA;
    uint64_t key = get_host_key();
    int ok = 1;
    if (fwrite(CONV_MAGIC, 1, CONV_MAGIC_LEN, f) != CONV_MAGIC_LEN) ok = 0;
    if (ok && fwrite(&schema, 4, 1, f) != 1) ok = 0;
    if (ok && fwrite(&key, 8, 1, f) != 1) ok = 0;
    if (ok && fwrite(&n, 4, 1, f) != 1) ok = 0;
    if (ok && fwrite(entries, sizeof(disk_entry_t), n, f) != n) ok = 0;
    fclose(f);

    if (ok) rename(tmp, fp);
    else    unlink(tmp);
}

static bool autotune_disabled(void)
{
    const char *e = getenv("AX_NO_AUTOTUNE");
    return (e && e[0] == '1');
}

static ax_conv_path_t forced_path(void)
{
    static int checked = 0;
    static ax_conv_path_t fp = AX_CONV_PATH_AUTO;
    if (!checked) {
        checked = 1;
        const char *e = getenv("AX_CONV_FORCE_PATH");
        if (!e) return fp;
        if (strcmp(e, "winograd") == 0)  fp = AX_CONV_PATH_WINOGRAD;
        else if (strcmp(e, "im2col") == 0)    fp = AX_CONV_PATH_IM2COL;
        else if (strcmp(e, "implicit") == 0)   fp = AX_CONV_PATH_IMPLICIT;
        else if (strcmp(e, "direct") == 0)     fp = AX_CONV_PATH_DIRECT_3X3;
        else if (strcmp(e, "smallcin") == 0)   fp = AX_CONV_PATH_SMALLCIN;
        else if (strcmp(e, "winograd_f43") == 0) fp = AX_CONV_PATH_WINOGRAD_F43;
    }
    return fp;
}

static const char *path_label(ax_conv_path_t p)
{
    switch (p) {
    case AX_CONV_PATH_WINOGRAD:   return "winograd";
    case AX_CONV_PATH_DIRECT_3X3: return "direct_3x3";
    case AX_CONV_PATH_SMALLCIN:   return "smallcin";
    case AX_CONV_PATH_IMPLICIT:   return "implicit";
    case AX_CONV_PATH_IM2COL:     return "im2col";
    case AX_CONV_PATH_BATCHED:    return "batched";
    case AX_CONV_PATH_1X1_FAST:       return "1x1_fast";
    case AX_CONV_PATH_WINOGRAD_F43: return "winograd_f43";
    default:                         return "?";
    }
}

/* enumerate which paths are eligible for a shape */
static int enumerate_paths(
    int kh, int kw, int sh, int sw, int ph, int pw,
    int64_t N, int64_t C_in, int64_t C_out,
    int64_t out_h, int64_t out_w,
    bool has_wino, bool has_wino43, bool has_conv_gemm,
    ax_conv_path_t *out, int cap)
{
    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;
    int n = 0;

    if (has_wino43 && ax_conv_prefer_winograd_f43(kh, kw, sh, sw, ph, pw,
                                                    N, C_in, C_out, out_h, out_w))
        out[n++] = AX_CONV_PATH_WINOGRAD_F43;

    if (n < cap && has_wino && ax_conv_prefer_winograd_f23(kh, kw, sh, sw, ph, pw,
                                                            N, C_in, C_out, out_h, out_w))
        out[n++] = AX_CONV_PATH_WINOGRAD;

    if (n < cap && ax_conv_can_direct_3x3(kh, kw, sh, sw, ph, pw, C_in))
        out[n++] = AX_CONV_PATH_DIRECT_3X3;

    if (n < cap && ax_conv_can_direct_smallcin(kh, kw, sh, sw, C_in, out_w))
        out[n++] = AX_CONV_PATH_SMALLCIN;

    if (n < cap && has_conv_gemm && ax_conv_prefer_implicit_gemm(K, M))
        out[n++] = AX_CONV_PATH_IMPLICIT;

    /* im2col+GEMM is always available as fallback */
    if (n < cap) out[n++] = AX_CONV_PATH_IM2COL;

    return n;
}

/* run a single conv path into od, return elapsed us */
static double trial_one(
    ax_conv_path_t path,
    struct ax_conv_scratch *s,
    const float *id, const float *wd, const float *bias, float *od,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t out_h, int64_t out_w,
    int kh, int kw, int sh, int sw, int ph, int pw, int T)
{
    int64_t M = out_h * out_w;
    double t0 = time_us();

    switch (path) {
    case AX_CONV_PATH_WINOGRAD:
        ax_conv_winograd_f23_forward(s, id, wd, bias, od,
                                      N, C_in, H, W, C_out, out_h, out_w, T);
        break;

    case AX_CONV_PATH_WINOGRAD_F43:
        ax_conv_winograd_f43_forward(s, id, wd, bias, od,
                                      N, C_in, H, W, C_out, out_h, out_w, T);
        break;

    case AX_CONV_PATH_DIRECT_3X3:
        for (int64_t n = 0; n < N; n++)
            ax_conv_direct_3x3_sample(id + n * C_in * H * W, wd, bias,
                                       od + n * C_out * M, C_in, C_out, H, W);
        break;

    case AX_CONV_PATH_SMALLCIN:
        for (int64_t n = 0; n < N; n++)
            ax_conv_direct_smallcin_sample(id + n * C_in * H * W, wd, bias,
                                            od + n * C_out * M,
                                            C_in, C_out, H, W,
                                            kh, kw, sh, sw, ph, pw, out_h, out_w);
        break;

    case AX_CONV_PATH_IMPLICIT:
        for (int64_t n = 0; n < N; n++) {
            ax_conv_params_t cp = {
                .input = id + n * C_in * H * W,
                .C_in = C_in, .H = H, .W = W,
                .kh = kh, .kw = kw, .sh = sh, .sw = sw,
                .ph = ph, .pw = pw,
                .out_h = out_h, .out_w = out_w,
            };
            ax_compute_conv_gemm(s->w2d, &cp, s->res_bufs[0]);
            float *rd = (float *)s->res_bufs[0]->storage->data;
            for (int64_t co = 0; co < C_out; co++) {
                float bv = bias ? bias[co] : 0.0f;
                ax_vf32 vb = ax_vf32_set1(bv);
                float *dst = od + (n * C_out + co) * M;
                const float *src = rd + co * M;
                int64_t m = 0, ve = M - (M % AX_VF32_WIDTH);
                for (; m < ve; m += AX_VF32_WIDTH)
                    ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), vb));
                for (; m < M; m++) dst[m] = src[m] + bv;
            }
        }
        break;

    case AX_CONV_PATH_IM2COL:
    default: {
        for (int64_t n = 0; n < N; n++) {
            float *cd = (float *)s->col_bufs[0]->storage->data;
            ax_conv_im2col_into(id + n * C_in * H * W, C_in, H, W,
                                kh, kw, sh, sw, ph, pw, out_h, out_w, cd);
            ax_storage_touch(s->col_bufs[0]->storage);
            ax_compute_gemm(s->w2d, s->col_bufs[0], s->res_bufs[0]);
            float *rd = (float *)s->res_bufs[0]->storage->data;
            for (int64_t co = 0; co < C_out; co++) {
                float bv = bias ? bias[co] : 0.0f;
                ax_vf32 vb = ax_vf32_set1(bv);
                float *dst = od + (n * C_out + co) * M;
                const float *src = rd + co * M;
                int64_t m = 0, ve = M - (M % AX_VF32_WIDTH);
                for (; m < ve; m += AX_VF32_WIDTH)
                    ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), vb));
                for (; m < M; m++) dst[m] = src[m] + bv;
            }
        }
        break;
    }
    }

    return time_us() - t0;
}

ax_conv_path_t ax_conv_autotune_select(
    struct ax_conv_scratch *s,
    const float *id, const float *wd, const float *bias, float *od,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t out_h, int64_t out_w,
    int kh, int kw, int sh, int sw, int ph, int pw, int T,
    ax_conv_path_t static_pick)
{
    conv_shape_key_t key;
    memset(&key, 0, sizeof(key));
    key.N = N; key.C_in = C_in; key.C_out = C_out;
    key.H = H; key.W = W;
    key.kh = kh; key.kw = kw; key.sh = sh; key.sw = sw;
    key.ph = ph; key.pw = pw;
    uint64_t h = shape_hash(&key);

    /* forced override: skip all calibration */
    ax_conv_path_t fp = forced_path();
    if (fp != AX_CONV_PATH_AUTO) {
        trial_one(fp, s, id, wd, bias, od,
                  N, C_in, H, W, C_out, out_h, out_w,
                  kh, kw, sh, sw, ph, pw, T);
        return fp;
    }

    /* autotune disabled: use static cascade's pick */
    if (autotune_disabled()) {
        trial_one(static_pick, s, id, wd, bias, od,
                  N, C_in, H, W, C_out, out_h, out_w,
                  kh, kw, sh, sw, ph, pw, T);
        return static_pick;
    }

    /* check caches */
    ax_conv_path_t cached;
    if (mem_lookup(h, &cached)) {
        trial_one(cached, s, id, wd, bias, od,
                  N, C_in, H, W, C_out, out_h, out_w,
                  kh, kw, sh, sw, ph, pw, T);
        return cached;
    }
    if (disk_lookup(h, &cached)) {
        mem_insert(h, cached);
        trial_one(cached, s, id, wd, bias, od,
                  N, C_in, H, W, C_out, out_h, out_w,
                  kh, kw, sh, sw, ph, pw, T);
        return cached;
    }

    /* enumerate eligible paths */
    bool has_wino = (s->wino_U && s->wino_V && s->wino_M);
    bool has_wino43 = (s->wino43_U && s->wino43_V && s->wino43_M);
    bool has_cg = ax_compute_has_conv_gemm();
    ax_conv_path_t paths[8];
    int n_paths = enumerate_paths(kh, kw, sh, sw, ph, pw,
                                   N, C_in, C_out, out_h, out_w,
                                   has_wino, has_wino43, has_cg, paths, 8);

    /* single eligible path: no trial needed */
    if (n_paths <= 1) {
        ax_conv_path_t pick = n_paths == 1 ? paths[0] : AX_CONV_PATH_IM2COL;
        trial_one(pick, s, id, wd, bias, od,
                  N, C_in, H, W, C_out, out_h, out_w,
                  kh, kw, sh, sw, ph, pw, T);
        mem_insert(h, pick);
        disk_save(h, pick);
        return pick;
    }

    /* trial all eligible paths.
       paths[0] writes into od (reference output for cross-check).
       subsequent paths write into tmp_buf and are spot-checked. */
    int64_t out_elems = N * C_out * out_h * out_w;
    float *tmp_buf = (float *)malloc((size_t)out_elems * sizeof(float));
    if (!tmp_buf) {
        trial_one(static_pick, s, id, wd, bias, od,
                  N, C_in, H, W, C_out, out_h, out_w,
                  kh, kw, sh, sw, ph, pw, T);
        return static_pick;
    }

    double times[8];
    times[0] = trial_one(paths[0], s, id, wd, bias, od,
                          N, C_in, H, W, C_out, out_h, out_w,
                          kh, kw, sh, sw, ph, pw, T);

    ax_conv_path_t winner = paths[0];
    double best_t = times[0];

    for (int pi = 1; pi < n_paths; pi++) {
        times[pi] = trial_one(paths[pi], s, id, wd, bias, tmp_buf,
                               N, C_in, H, W, C_out, out_h, out_w,
                               kh, kw, sh, sw, ph, pw, T);
        if (times[pi] < best_t) {
            bool agree = true;
            int64_t step = out_elems > 64 ? out_elems / 32 : 1;
            for (int64_t i = 0; i < out_elems; i += step) {
                float d = od[i] - tmp_buf[i];
                if (d > 0.01f || d < -0.01f) { agree = false; break; }
            }
            if (agree) {
                memcpy(od, tmp_buf, (size_t)out_elems * sizeof(float));
                winner = paths[pi];
                best_t = times[pi];
            }
        }
    }
    free(tmp_buf);

#ifndef AX_NO_STDIO
    static int verbose = -1;
    if (verbose < 0) {
        const char *e = getenv("AX_CONV_AUTOTUNE_VERBOSE");
        verbose = (e && e[0] == '1') ? 1 : 0;
    }
    if (verbose) {
        fprintf(stderr,
            "axiom: conv autotune N=%ld Cin=%ld Cout=%ld %ldx%ld K%d:",
            (long)N, (long)C_in, (long)C_out, (long)H, (long)W, kh);
        for (int pi = 0; pi < n_paths; pi++)
            fprintf(stderr, " %s=%.0fus", path_label(paths[pi]), times[pi]);
        fprintf(stderr, " -> %s\n", path_label(winner));
    }
#endif

    mem_insert(h, winner);
    disk_save(h, winner);
    return winner;
}
