/* axiom/rng.h — high-quality pseudorandom number generator.

   xoshiro256** with per-thread state. replaces rand()/srand() which
   has poor quality and is not thread-safe.

   thread-safety: every thread keeps its own xoshiro256** state
   initialised by mixing the global seed with the thread id. each
   ax_rng_* call is concurrent-safe across threads but sequential within
   one thread.

   ownership: no allocations. all functions return values by value.

   error handling: no failure modes. ax_rng_seed mutates the per-thread
   seed source; subsequent rng_* calls produce a deterministic sequence
   per thread. */

#ifndef AX_RNG_H
#define AX_RNG_H

#include <stdint.h>
#include <stdbool.h>

/* seed the global rng. call once at startup.
   if never called, first use auto-seeds from time+pid. */
void ax_rng_seed(uint64_t seed);

/* returns true if ax_rng_seed has been called */
bool ax_rng_is_seeded(void);

/* generate a uniform uint64 in [0, UINT64_MAX] */
uint64_t ax_rng_uint64(void);

/* generate a uniform float in [0, 1) */
float ax_rng_float(void);

/* generate a uniform float in [low, high) */
float ax_rng_uniform(float low, float high);

/* generate a normally distributed float (mean=0, std=1)
   using the ziggurat method for speed. */
float ax_rng_normal(void);

/* generate a normal float with given mean and std */
float ax_rng_normal_params(float mean, float std);

/* generate a uniform uint64 in [0, n) without modulo bias */
uint64_t ax_rng_bounded(uint64_t n);

#endif /* AX_RNG_H */
