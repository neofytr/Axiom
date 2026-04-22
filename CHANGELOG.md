# Changelog

All notable changes to **axiom** are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version constants in `include/axiom/types.h`
(`AX_VERSION_MAJOR`/`MINOR`/`PATCH`, `AX_VERSION_STRING`) and the cmake
project version are the single sources of truth and must change together.

## [Unreleased]

## [0.10.0] — 2026-04-23

First tagged pre-1.0 release. Establishes the abi baseline that future
minor releases must remain backward-compatible with. See `axiom/axiom.h`
for the full conventions block.

### Added

- **Public/internal header split.** Public headers stay under
  `include/axiom/`. Implementation-only contracts (vtable struct,
  parallelism thresholds, calibration entry points) moved to
  `include/axiom/internal/` and are not part of the published abi.
- **Centralised conventions doc** in `axiom/axiom.h` covering error
  handling, tensor ownership patterns, thread safety, naming, and the
  abi-stability promise.
- **`AX_VERSION_*` constants** in `axiom/types.h` plus `AX_DEPRECATED`
  and `AX_ABI_STABLE_SINCE` macros for the deprecation cycle.
- **`ax_compute_backend_name()`** public accessor — replaces direct
  vtable->name access from external benchmarks/examples.
- **`AX_RETURN_NULL_IF_ALLOC_FAIL`** convenience macro in `error.h`,
  applied to every public `ax_*_create` constructor so allocation
  failures always set `ax_err_last_status() == AX_ERR_ALLOC` instead
  of returning NULL silently.
- **Install target** (`cmake --install`): public headers, libraries,
  `axiomConfig.cmake` for `find_package(axiom)`, and `axiom.pc` for
  `pkg-config`. Internal headers are excluded.
- **CTest fast-startup** via `AX_NO_AUTOTUNE=1` + `AX_GEMM_CALIBRATE=0`
  in the test environment. Full suite runtime dropped from ~270 s to
  ~2 s.

### Changed

- All public functions documented for ownership, thread safety, error
  reporting, and failure modes (per-header notes + per-symbol notes
  for non-trivial cases).
- `ax_compute_get_ops`, `ax_backend_for_device`, and
  `ax_compute_register_backend` moved from public `compute.h` to
  `internal/compute_internal.h`.
- Header guards normalised to the `AX_<NAME>_H` convention.

### Notes on stability

Every symbol declared in any header under `include/axiom/` (excluding
`internal/`) is implicitly tagged abi-stable as of 0.10.0. Removals
or signature changes for these symbols require:

1. one minor release (e.g. 0.11.0) with the symbol marked
   `AX_DEPRECATED` and a hint to its replacement;
2. removal at the next major bump (1.0.0).

Internal headers carry no stability guarantee.
