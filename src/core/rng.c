/* rng.c — xoshiro256** prng implementation.
   thread-local state, auto-seeds on first use. */

#include "axiom/rng.h"
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <process.h>
#define ax_getpid() _getpid()
#else
#include <unistd.h>
#define ax_getpid() getpid()
#endif

/* per-thread xoshiro256** state */
static _Thread_local uint64_t s[4];
static _Thread_local int seeded = 0;

/* splitmix64 for seeding xoshiro from a single uint64 */
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

void ax_rng_seed(uint64_t seed) {
    uint64_t sm = seed;
    s[0] = splitmix64(&sm);
    s[1] = splitmix64(&sm);
    s[2] = splitmix64(&sm);
    s[3] = splitmix64(&sm);
    seeded = 1;
}

bool ax_rng_is_seeded(void) {
    return seeded != 0;
}

/* auto-seed from time + pid if not yet seeded */
static void ensure_seeded(void) {
    if (seeded) return;
    uint64_t t = (uint64_t)time(NULL);
    uint64_t p = (uint64_t)ax_getpid();
    ax_rng_seed(t ^ (p << 32) ^ (p * 0x517cc1b727220a95ULL));
}

/* xoshiro256** core */
uint64_t ax_rng_uint64(void) {
    ensure_seeded();
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

float ax_rng_float(void) {
    /* use top 24 bits for float mantissa precision */
    return (float)(ax_rng_uint64() >> 40) * 0x1.0p-24f;
}

float ax_rng_uniform(float low, float high) {
    return low + ax_rng_float() * (high - low);
}

/* box-muller for normal distribution.
   generates pairs; caches one for next call. */
static _Thread_local int has_spare = 0;
static _Thread_local float spare_val = 0.0f;

float ax_rng_normal(void) {
    if (has_spare) {
        has_spare = 0;
        return spare_val;
    }

    float u, v, s_val;
    do {
        u = ax_rng_float() * 2.0f - 1.0f;
        v = ax_rng_float() * 2.0f - 1.0f;
        s_val = u * u + v * v;
    } while (s_val >= 1.0f || s_val == 0.0f);

    float mul = sqrtf(-2.0f * logf(s_val) / s_val);
    spare_val = v * mul;
    has_spare = 1;
    return u * mul;
}

float ax_rng_normal_params(float mean, float std) {
    return mean + std * ax_rng_normal();
}

/* debiased bounded random: rejection method.
   uniform in [0, n) with no modulo bias. */
uint64_t ax_rng_bounded(uint64_t n) {
    if (n <= 1) return 0;
    /* compute threshold to reject values that cause bias */
    uint64_t threshold = (-n) % n; /* = (2^64 - n) % n */
    for (;;) {
        uint64_t r = ax_rng_uint64();
        if (r >= threshold)
            return r % n;
    }
}
