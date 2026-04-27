/* calib_cache.c — disk persistence for the autotuner.

   stores ax_calib_payload_t under ${XDG_CACHE_HOME}/axiom/calib.bin
   (or fallbacks). file format:

     [magic]            8 bytes : "AXIOMCAL"
     [schema_version]   4 bytes : u32 little-endian
     [key_hash]         8 bytes : u64 little-endian — host fingerprint
     [payload_size]     4 bytes : u32 little-endian (sizeof payload)
     [payload]          N bytes : ax_calib_payload_t struct dump

   keep the format simple — host-endian struct dumps are fine because
   we only read on the same host that wrote. cross-host caches are
   distinguished by the key_hash; mismatch = treat as miss.

   the file is written atomically: write to "calib.bin.tmp" then rename
   on top. concurrent writers from multiple processes will race but
   the rename is atomic so we never see a half-written file. */

#include "axiom/internal/calib_cache.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* bump when the payload struct layout changes incompatibly.
   v2: A3 added per-regime gemm tiles (gemm_{mc,nc,kc}_{small,med,large}).
   v3: added gemm_unpacked_a_max_k. */
#define AX_CALIB_SCHEMA_VERSION 3u

#define AX_CALIB_MAGIC "AXIOMCAL"
#define AX_CALIB_MAGIC_LEN 8

/* fnv-1a 64-bit — small, cheap, good enough for fingerprinting. */
static uint64_t fnv1a64(const void *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* read the cpu brand string. on x86 we use cpuid leafs 0x80000002-4
   (48 bytes total). on other arches or if cpuid fails, fall back to
   /proc/cpuinfo "model name" line. anything goes — this is fingerprint
   material, not security-sensitive. */
static void read_cpu_brand(char *buf, size_t buf_size) {
    if (buf_size == 0) return;
    buf[0] = '\0';

#if defined(__x86_64__) || defined(__i386__)
    if (buf_size >= 49) {
        uint32_t regs[12];
        __asm__ __volatile__ (
            "cpuid"
            : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
            : "a"(0x80000002), "c"(0)
        );
        __asm__ __volatile__ (
            "cpuid"
            : "=a"(regs[4]), "=b"(regs[5]), "=c"(regs[6]), "=d"(regs[7])
            : "a"(0x80000003), "c"(0)
        );
        __asm__ __volatile__ (
            "cpuid"
            : "=a"(regs[8]), "=b"(regs[9]), "=c"(regs[10]), "=d"(regs[11])
            : "a"(0x80000004), "c"(0)
        );
        memcpy(buf, regs, 48);
        buf[48] = '\0';
        if (buf[0] != '\0') return;
    }
#endif

    /* /proc/cpuinfo fallback */
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        snprintf(buf, buf_size, "unknown");
        return;
    }
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        const char *prefix = "model name";
        if (strncmp(line, prefix, strlen(prefix)) == 0) {
            const char *colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ' || *colon == '\t') colon++;
                size_t len = strlen(colon);
                while (len > 0 && (colon[len-1] == '\n' || colon[len-1] == ' '))
                    len--;
                size_t copy = len < buf_size - 1 ? len : buf_size - 1;
                memcpy(buf, colon, copy);
                buf[copy] = '\0';
                fclose(fp);
                return;
            }
        }
    }
    fclose(fp);
    if (buf[0] == '\0') snprintf(buf, buf_size, "unknown");
}

static const char *active_isa_label(void) {
#if defined(__AVX512F__) && defined(__FMA__)
    return "avx512";
#elif defined(__AVX2__) && defined(__FMA__)
    return "avx2";
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return "neon";
#else
    return "scalar";
#endif
}

static int active_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* compute the host fingerprint hash. covers cpu brand + isa + thread
   count + a build identifier (compile date string captured via __DATE__ +
   __TIME__ macros at compile time). that last field bumps the cache
   whenever the binary is rebuilt — safest default for a project still
   in heavy development. once the calibrator stabilises, the build
   identifier can be downgraded to a release tag. */
static const char ax_calib_build_id[] = __DATE__ " " __TIME__;

static uint64_t compute_host_key(void) {
    char brand[80] = {0};
    read_cpu_brand(brand, sizeof(brand));

    /* hash all key components into one u64 */
    uint64_t h = fnv1a64(brand, strlen(brand));
    const char *isa = active_isa_label();
    h ^= fnv1a64(isa, strlen(isa));
    int nt = active_threads();
    h ^= fnv1a64(&nt, sizeof(nt));
    h ^= fnv1a64(ax_calib_build_id, sizeof(ax_calib_build_id) - 1);
    h ^= (uint64_t)AX_CALIB_SCHEMA_VERSION;
    return h;
}

/* mkdir -p: create directory and any missing parents. ignores EEXIST.
   returns 0 on success, -1 on real failure. */
static int mkdir_p(const char *path) {
    if (!path || !*path) return -1;
    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return -1;
    memcpy(buf, path, n + 1);
    /* walk through each '/' and mkdir along the way. skip empty roots. */
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

/* resolve the cache directory and file path.
   returns true on success, false if no path can be built (no HOME). */
static bool resolve_cache_path(char *file_path, size_t fp_size,
                                char *dir_path, size_t dir_size) {
    const char *override = getenv("AX_CALIB_CACHE_PATH");
    if (override && override[0] != '\0') {
        size_t n = strlen(override);
        if (n + 1 > fp_size) return false;
        memcpy(file_path, override, n + 1);
        /* derive dir from file path */
        const char *slash = strrchr(file_path, '/');
        if (slash) {
            size_t dn = (size_t)(slash - file_path);
            if (dn + 1 > dir_size) return false;
            memcpy(dir_path, file_path, dn);
            dir_path[dn] = '\0';
        } else {
            dir_path[0] = '.';
            dir_path[1] = '\0';
        }
        return true;
    }

    const char *xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0] != '\0') {
        if (snprintf(dir_path, dir_size, "%s/axiom", xdg) >= (int)dir_size) return false;
    } else {
        const char *home = getenv("HOME");
        if (!home || home[0] == '\0') return false;
        if (snprintf(dir_path, dir_size, "%s/.cache/axiom", home) >= (int)dir_size) return false;
    }
    if (snprintf(file_path, fp_size, "%s/calib.bin", dir_path) >= (int)fp_size) return false;
    return true;
}

static bool calib_cache_disabled(void) {
    const char *e = getenv("AX_NO_CALIB_CACHE");
    return (e && e[0] == '1');
}

bool ax_calib_cache_load(ax_calib_payload_t *out) {
    if (!out) return false;
    if (calib_cache_disabled()) return false;

    char file_path[1024], dir_path[1024];
    if (!resolve_cache_path(file_path, sizeof(file_path), dir_path, sizeof(dir_path)))
        return false;

    FILE *fp = fopen(file_path, "rb");
    if (!fp) return false;

    /* header parse: magic + version + key_hash + payload_size */
    char magic[AX_CALIB_MAGIC_LEN];
    uint32_t version = 0;
    uint64_t key = 0;
    uint32_t payload_size = 0;

    if (fread(magic, 1, AX_CALIB_MAGIC_LEN, fp) != AX_CALIB_MAGIC_LEN ||
        memcmp(magic, AX_CALIB_MAGIC, AX_CALIB_MAGIC_LEN) != 0) {
        fclose(fp); return false;
    }
    if (fread(&version, sizeof(version), 1, fp) != 1 ||
        version != AX_CALIB_SCHEMA_VERSION) {
        fclose(fp); return false;
    }
    if (fread(&key, sizeof(key), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fread(&payload_size, sizeof(payload_size), 1, fp) != 1 ||
        payload_size != sizeof(ax_calib_payload_t)) {
        fclose(fp); return false;
    }

    uint64_t want_key = compute_host_key();
    if (key != want_key) {
        fclose(fp); return false;
    }

    if (fread(out, sizeof(*out), 1, fp) != 1) {
        fclose(fp); return false;
    }
    fclose(fp);
    return true;
}

bool ax_calib_cache_save(const ax_calib_payload_t *in) {
    if (!in) return false;
    if (calib_cache_disabled()) return false;

    char file_path[1024], dir_path[1024];
    if (!resolve_cache_path(file_path, sizeof(file_path), dir_path, sizeof(dir_path)))
        return false;

    if (mkdir_p(dir_path) != 0) return false;

    /* write to .tmp then rename for atomicity */
    char tmp_path[1100];
    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", file_path) >= (int)sizeof(tmp_path))
        return false;

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp) return false;

    uint32_t version = AX_CALIB_SCHEMA_VERSION;
    uint64_t key = compute_host_key();
    uint32_t payload_size = (uint32_t)sizeof(ax_calib_payload_t);
    int ok = 1;
    if (fwrite(AX_CALIB_MAGIC, 1, AX_CALIB_MAGIC_LEN, fp) != AX_CALIB_MAGIC_LEN) ok = 0;
    if (ok && fwrite(&version, sizeof(version), 1, fp) != 1) ok = 0;
    if (ok && fwrite(&key, sizeof(key), 1, fp) != 1) ok = 0;
    if (ok && fwrite(&payload_size, sizeof(payload_size), 1, fp) != 1) ok = 0;
    if (ok && fwrite(in, sizeof(*in), 1, fp) != 1) ok = 0;
    fclose(fp);

    if (!ok) {
        unlink(tmp_path);
        return false;
    }

    if (rename(tmp_path, file_path) != 0) {
        unlink(tmp_path);
        return false;
    }
    return true;
}

void ax_calib_cache_fingerprint(char *buf, size_t buf_size) {
    if (!buf || buf_size == 0) return;
    char brand[80] = {0};
    read_cpu_brand(brand, sizeof(brand));
    uint64_t cpu_hash = fnv1a64(brand, strlen(brand));
    snprintf(buf, buf_size, "v%u-%016llx-%s-nt%d",
             AX_CALIB_SCHEMA_VERSION,
             (unsigned long long)cpu_hash,
             active_isa_label(),
             active_threads());
}
