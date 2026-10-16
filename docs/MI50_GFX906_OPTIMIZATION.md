# MI50 / gfx906 optimization fork

This branch turns the tested AMD Instinct MI50 work into a reviewable source branch instead of a collection of
out-of-tree patches. It is intended for MI50, MI60, and Radeon VII owners who can build the existing opt-in gfx906
backend and want the measured Q2_0 improvement and wave64 correctness fix.

## Baseline and scope

| Item | Value |
| --- | --- |
| Upstream | [`Niko1221/Strata`](https://github.com/Niko1221/Strata) |
| Base commit | `296b59d2f73c8f04a805df29c2fa97655adace78` |
| Release identity | v0.1.44.1 |
| Tested GPU | AMD Instinct MI50 32 GB, `gfx906`, wave64 |
| Tested runtime | ROCm 6.2.4 / HIP 6.2.41134, AMD clang 18, Linux 6.8 amdgpu |
| Tested model path | Strata v0.1.44 Q2_0 expert weights |

The branch contains five focused source deltas:

1. Gate gfx906-incompatible optional targets and NVIDIA-only FP4 compilation in CMake.
2. Use bounded reusable pinned staging for queued HIP expert uploads.
3. Enable the existing LDS grouped gate/up kernel for Q2_0 (`gu_type == 42`).
4. Bound register-resident Top-K to the proven 33-key capacity on gfx906 wave64.
5. Add public reproduction notes, compact evidence, and standalone gfx906 dot-product probes.

No model files, generated packs, benchmark responses, service configuration, credentials, host names, or private
infrastructure paths are included.

## What was measured

The retained Q2_0 change extends the existing LDS grouped gate/up path rather than introducing a new approximate
quantization contract.

| Measurement | Unmodified base | This branch | Result |
| --- | ---: | ---: | ---: |
| Fixed 256-token generation, median | 30.503 tok/s | 34.294 tok/s | +12.43% |
| Independent steady repeat | 30.503 tok/s reference | 33.930 tok/s | +11.23% |
| Native expert, layer 0/23/47 | 231.8–244.9 us | 149.2–155.8 us | bitwise equal |
| Gate/up portion | 115.6–119.5 us | 88.7–91.7 us | bitwise equal |

Quality was repeated because temperature-zero output varied sequentially on both binaries. The unmodified runs
scored 25/33 and 22/33; optimized runs scored 23/33 and 24/33. Both sides therefore had a median of 23.5/33, and
all four runs passed the exact-token 2K, 8K, and 16K retrieval cases.

The Top-K change is retained for correctness, not as a speed claim. Before the fix, the 262,144-context case could
produce non-identical selected IDs because the generic HIP path assumed a wave32-sized register budget. The gfx906
path now uses the proven 33-key capacity and passed all 20 active parity cases, including 64/64 identical IDs at
262,144 context.

Compact machine-readable results are in [`mi50-evidence/`](mi50-evidence/).

## Build

The upstream gfx906 build is opt-in. The following is the reference CMake shape; use an environment that actually
ships gfx906 rocBLAS/hipBLAS libraries. Current official ROCm releases no longer support gfx906, so do not replace a
working host runtime merely to reproduce this branch.

```sh
git submodule update --init --recursive

cmake -S . -B build-906 -G Ninja \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++

ninja -C build-906 strata native_expert_bench qsa_topk_parity qsa_topk_active_parity gr_parity
```

The upstream documentation describes a community ROCm 7.14 container with gfx906 libraries. The measurements in
this branch used ROCm 6.2.4 on the test host. Treat those as two supported reproduction routes, not as evidence that
their generated code or performance is identical.

## Safe runtime defaults

The validated Q2_0 configuration used:

```sh
export STRATA_TOPK_OLD=1
export STRATA_GR_SPLIT=0
export STRATA_QFUSE=0
```

`STRATA_TOPK_OLD=1` is a conservative runtime selection used in the measured configuration. The source-level wave64
capacity fix is still required because parity tests and future dispatches must not assume a wave32 register budget.

## Validation order

Run gates in this order when porting the branch or changing the toolchain:

1. Configure and build `strata`, `native_expert_bench`, `qsa_topk_parity`, `qsa_topk_active_parity`, and `gr_parity`.
2. Run file-expert, expert-profile, platform-memory, and device self-tests available in the checkout.
3. Compare Q2 expert mode 0 versus the LDS mode on layers 0, 23, and 47; require zero differing outputs.
4. Run all Top-K parity cases and require bitwise selected IDs at 262,144 context.
5. Run one warm-up followed by at least three fixed 256-token generations for each arm; compare medians.
6. Retain a performance change only when the improvement is reproducible, exceeds 3%, and has no deterministic
   quality loss.
7. Run the deterministic quality suite, including exact-token 2K, 8K, and 16K retrieval, before deployment.

The accepted source snapshot was compared against the tested machine copy after normalizing line endings. The file
hashes are recorded in [`mi50-evidence/source-manifest.json`](mi50-evidence/source-manifest.json).

## Rejected experiments

These results are preserved so other MI50 owners do not repeat locally attractive but end-to-end negative paths:

| Experiment | Result | Decision |
| --- | --- | --- |
| Existing QFUSE | 14.217 tok/s off vs 11.489 on, -19.19% | keep disabled |
| QFUSE q32-owner repair | parity passed; 15.814 vs 33.076 tok/s, -52.2% | rejected |
| GR split | 33.744 off vs 33.076 on, -1.98% | keep disabled |
| Per-dot Q2 two-plane `v_dot8_i32_i4` | exact CPU mapping; realistic packed path 2–12% slower | rejected |
| Forced `launch_bounds(256,3)` | 166.3 us to 209.7 us, +26.1% latency | rejected |
| LDS row/thread and aligned-copy sweeps | slower or below 1% | rejected |

The dot8 instruction is real and the pre-expanded microkernel was about 5–9% faster. The complete path lost because
unpack, popcount, and plane construction were paid inside each dot product. A future attempt should build activation
planes once and reuse them across selected experts; it should not reintroduce the rejected per-dot conversion. The
standalone probes are under [`tools/mi50/`](../tools/mi50/).

## Updating from upstream

Do not blindly replay the historical complete patch on a new release.

1. Create a fresh branch from the desired upstream commit.
2. Try an unmodified gfx906 configure/build first.
3. Port the CMake compatibility pieces only if the corresponding targets still fail on gfx906.
4. Inspect expert-cache staging; skip the local staging delta if upstream now provides equivalent reusable pinned
   buffers for `STRATA_HIP_GFX906`.
5. Port the Q2_0 LDS dispatch and wave64 Top-K guard separately.
6. Re-run the full validation order above and record new evidence against the new base.

Suggested commit order is build compatibility, expert staging, Q2 performance, Top-K correctness, then documentation.
This keeps each change independently reviewable and makes future rebases much less painful.

## Contribution boundaries

- Keep gfx906-specific behavior under `STRATA_HIP_GFX906`; do not alter CUDA or wave32 defaults without their own
  tests.
- Do not commit models, build products, logs, raw API responses, service files, credentials, tokens, private keys, or
  host-specific configuration.
- Preserve negative evidence. A microbenchmark win is not an end-to-end claim.
- Do not open an upstream pull request until the branch has been rebuilt and validated on real gfx906 hardware.
