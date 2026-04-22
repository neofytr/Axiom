#!/usr/bin/env python3
"""check_portability.py — flag x86-only intrinsics in portable code.

axiom is built for x86_64, aarch64, and embedded targets. portable code
(everything outside src/compute/backends/jit_x64.c, jit_gemm_avx2.c,
jit_gemm_avx512.c, the simd_defs ISA branches, and explicitly arch-
gated regions) must NOT call x86-only intrinsics directly without an
`#if defined(__x86_64__)` guard surrounding the use.

three regressions in a row today (rdtsc in cpu_opt.c::ax_prof_tick,
attention.c::PF_TICK, and earlier conv_gemm profile) all had the same
shape: x86 intrinsic call survived release-mode dead-code elimination
because gating was a runtime flag, then arm64 link failed with
"undefined reference to __builtin_ia32_*". this script catches that
class before merge.

algorithm: for every .c / .h file under src/ and include/, find any
line containing a known x86-only token (`__builtin_ia32_`, `__rdtsc`,
`_mm_`, `_mm256_`, `_mm512_`). check whether that line falls inside a
`#if defined(__x86_64__)` / `#elif defined(AX_SIMD_AVX...)` /
`#ifdef __x86_64__` block. files explicitly listed as x86-native are
skipped entirely.

exit 0 = no issues, exit 1 = at least one unguarded use found. """

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# files that are permitted to use x86 intrinsics without per-line guards
# (they're already inside a single x86 #if block at file-or-config level).
SKIP_FILES = {
    # JIT encoders are x86-only by construction; cmake only links them on x86.
    "src/compute/backends/jit_x64.c",
    "src/compute/backends/jit_x64.h",
    "src/compute/backends/jit_gemm_avx2.c",
    "src/compute/backends/jit_gemm_avx2.h",
    "src/compute/backends/jit_gemm_avx512.c",
    "src/compute/backends/jit_gemm_avx512.h",
    # SIMD wrapper provides per-ISA intrinsic blocks; the file's whole
    # structure is one big #if defined(AX_SIMD_AVX...) chain.
    "src/compute/backends/simd_defs.h",
}

# tokens that appear ONLY in x86 intrinsics. _mm matches MMX/SSE; _mm256
# is AVX; _mm512 is AVX-512. __builtin_ia32_* is the gcc family.
X86_TOKEN = re.compile(
    r"(?:__builtin_ia32_|__rdtsc\b|\b_mm_|\b_mm256_|\b_mm512_)"
)

# patterns that open a region where x86 intrinsics are OK
ARCH_GUARD_OPEN = re.compile(
    r"#\s*(?:if|elif|ifdef)\s.*"
    r"(?:__x86_64__|__amd64__|_M_X64|AX_SIMD_AVX|__AVX|__SSE)"
)
PREP_OTHER = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b")


def file_lines_in_x86_guard(path):
    """Yield (line_num, line, in_x86, in_block_comment) for each line of `path`."""
    stack = []  # stack of bools: True = current open region is x86-guarded
    in_block_comment = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            # track multi-line /* ... */ comments — line is "in comment" if
            # we're still in one when reading it. ignore intra-line opens
            # that close on the same line (handled by single-line strip below).
            line_starts_in_comment = in_block_comment
            scan = line
            while True:
                if in_block_comment:
                    end = scan.find("*/")
                    if end == -1:
                        break  # whole rest of line is comment
                    scan = scan[end + 2:]
                    in_block_comment = False
                else:
                    start = scan.find("/*")
                    if start == -1:
                        break
                    end = scan.find("*/", start + 2)
                    if end == -1:
                        in_block_comment = True
                        break
                    scan = scan[end + 2:]
            stripped = line.strip()
            if PREP_OTHER.match(stripped):
                d = stripped.split()[0][1:].strip()
                # nested form: "# if" → take the second word
                if d.startswith("#"):
                    d = d[1:].strip()
                if d in ("if", "ifdef", "ifndef"):
                    stack.append(bool(ARCH_GUARD_OPEN.match(stripped)))
                elif d == "elif":
                    if stack:
                        stack[-1] = bool(ARCH_GUARD_OPEN.match(stripped))
                elif d == "else":
                    # entering the negated branch: invert top
                    if stack:
                        stack[-1] = False  # conservative: outside x86 branch
                elif d == "endif":
                    if stack:
                        stack.pop()
            in_x86 = any(stack)
            yield lineno, line, in_x86, line_starts_in_comment


def scan_file(path):
    """Return list of (lineno, line) where an x86 token appears unguarded."""
    issues = []
    for lineno, line, in_x86, in_comment in file_lines_in_x86_guard(path):
        if in_x86:
            continue
        if in_comment:
            continue  # whole line is inside a /* ... */ block
        # strip single-line C / C++ comments before token-matching
        no_block = re.sub(r"/\*.*?\*/", "", line)
        no_line = re.sub(r"//.*$", "", no_block)
        if X86_TOKEN.search(no_line):
            issues.append((lineno, line.rstrip()))
    return issues


def main():
    bad = 0
    for root, dirs, files in os.walk(ROOT):
        # skip out-of-tree build artefacts, vendored deps, agent worktrees
        dirs[:] = [d for d in dirs if d not in {".git", ".claude", "build", "build-asan",
                                                 "build-embedded", "build-embedded2",
                                                 ".venv", "node_modules", "legacy"}]
        rel_root = os.path.relpath(root, ROOT)
        if rel_root == ".":
            rel_root = ""
        for name in files:
            if not name.endswith((".c", ".h", ".cpp", ".cu")):
                continue
            rel = os.path.join(rel_root, name) if rel_root else name
            if rel in SKIP_FILES:
                continue
            path = os.path.join(root, name)
            issues = scan_file(path)
            for lineno, line in issues:
                print(f"{rel}:{lineno}: x86-only intrinsic outside __x86_64__ guard:")
                print(f"  {line}")
                bad += 1
    if bad:
        print(f"\nFAIL: {bad} unguarded x86 intrinsic use(s) found.", file=sys.stderr)
        print("wrap each in `#if defined(__x86_64__) || defined(_M_X64)` ... `#endif`,",
              file=sys.stderr)
        print("or add the file to scripts/check_portability.py SKIP_FILES if it is",
              file=sys.stderr)
        print("intentionally x86-only and never linked on other architectures.",
              file=sys.stderr)
        return 1
    print("ok: no unguarded x86 intrinsics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
