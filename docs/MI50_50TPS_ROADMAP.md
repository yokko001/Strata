# MI50 50 tok/s investigation and experiment roadmap

This document tracks the work required to make Strata a first-class AMD
Instinct MI50 (`gfx906`) target.  The immediate target is repeatable steady
single-request generation above **50 tokens/s**, measured over fixed 256-token
responses after warm-up.  Throughput from concurrent requests is reported
separately and must not be substituted for the single-stream target.

Last inventory: 2026-10-17.  The tested branch is based directly on upstream
commit `296b59d2f73c8f04a805df29c2fa97655adace78`; at the inventory date upstream
`main` had no commits beyond that base.  The MI50 branch contains 34 focused
commits on top.

## Primary sources and hardware constraints

| Subject | Primary source | Constraint for this project |
|---|---|---|
| MI50 specification | [AMD accelerator specifications](https://www.amd.com/en/products/specifications/accelerators.html) | Vega 20 / gfx906, 60 CUs, 3,840 stream processors, up to 1 TB/s HBM2, 26.5 TFLOP/s FP16, 13.3 TFLOP/s FP32, 300 W peak |
| ROCm 6.2 support | [ROCm 6.2 Linux requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.1/reference/system-requirements.html) | MI50/gfx906 was already deprecated; retain the validated ROCm 6.2.4 runtime until a compiler-only A/B proves a safer replacement |
| Current ROCm support | [Current ROCm Linux requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) | MI50 is unsupported by current prebuilt ROCm libraries; a routine in-place ROCm upgrade is not an optimization experiment |
| gfx906 ISA and ABI | [LLVM AMDGPU backend guide](https://llvm.org/docs/AMDGPUUsage.html) | wave64, GFX9 VGPR allocation granularity, LDS, `v_dot4_i32_i8` and `v_dot8_i32_i4`; register/LDS pressure must be checked in generated code |
| Model formats | [ISTA-DASLab GSQ-RCO model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | Q2_0 deliberately avoids lookup-heavy i-quant decode; IQ3_S is the quality operating point, not the speed operating point |
| Strata model sizing | [Strata model guide](MODELS.md) | Resident transformer data: Q2_0 37.6 GB, IQ2_XS 39.2 GB, IQ3_XXS 47.0 GB, IQ3_S 54.8 GB; the second 28.8 GB PLE shard may remain on SSD |
| Request batching | [Strata batching guide](BATCHING.md) | Every slot consumes VRAM otherwise available to the expert cache; aggregate and per-request rates must both be reported |

The specific card used for validation is a 32 GB MI50 with ECC enabled,
`gfx906:sramecc+:xnack-`, 60 CUs and wave64.  The validated userspace is ROCm
6.2.4 / HIP 6.2.41134 with clang 18.  The guest's old AVX-only Xeon CPU makes
CPU expert fallback, pack conversion and cache-miss recovery relatively weak,
but it is not the dominant cost after the Q2_0 expert cache is warm.

## Measured starting points

| Model / path | Steady result | Interpretation |
|---|---:|---|
| Qwen3.8-Flash-Next Q2_0, unmodified base | 30.503 tok/s | fixed tg256 baseline |
| Qwen3.8-Flash-Next Q2_0, retained gfx906 path | 33.930-34.294 tok/s | accepted LDS gate/up path, +11.2-12.4% |
| Qwen3.8-Flash-Next Q2_0, latest experiment control | 32.867 tok/s | same-binary control used for recent A/B experiments |
| Qwen3.8-Flash-Next IQ3_S, v0.1.39 | 19.6 tok/s | 13,397 cached experts, 97.6-98.4% hit rate |
| Qwen3.8-Flash-Next IQ3_S, calibrated v0.1.40.1 | 26.819 tok/s | best recorded IQ3_S operating point |
| Qwen3.8-Flash-Next Coder IQ1_M, full cache | 37.2-37.9 tok/s | historical full-cache result; must be repeated on the current branch |

The Q2_0 target therefore needs about **1.46x** over the best retained median.
IQ3_S needs about **1.86x** over its best calibrated median.  Increasing RAM or
SSD speed alone cannot supply either multiplier.

The detailed Q2_0 profile measured a 58.97 ms median verify window.  Reaching
50 tok/s at the same draft acceptance requires about 38.8 ms per equivalent
window, roughly 20 ms less.  The GPU-reachable portion was 42.14 ms; CPU expert
time was 0.01 ms and measured PCIe expert traffic rounded to 0.00 ms.  The
largest reported stages were VRAM expert hits (6.39 ms), waitA (3.62 ms), Q8
QKV work (3.31 ms), router/plan/doorbell (2.98 ms), head (2.33 ms), `z`
(2.09 ms), shared/quantize (2.08 ms), and output projection (1.80 ms).

## What the RAM and NVMe upgrade changes

Adding 64 GB of RAM is valuable for IQ3_S and other larger formats because the
54.8 GB transformer shard, resident expert pool, KV state and filesystem cache
can coexist without paging.  It also permits a fair current-branch Coder and
IQ3 comparison.  It does not enlarge MI50 VRAM or reduce the dense and routed
GPU kernels in a warm Q2_0 window.

A low-latency PCIe-passthrough SSD should hold active packs, the PLE file,
generated model data and builds.  It improves cold readiness, mmap faults, PLE
tail latency, pack conversion and cache-miss recovery.  The canonical raw model
may remain on slower archival storage.  Once the needed pages and experts are
resident, SSD bandwidth is not part of the measured steady Q2_0 critical path.

After installation, record physical root-complex placement outside the guest,
guest block topology, direct-read bandwidth and latency, model cold-start time,
warm-restart time, major faults, and steady decode.  A virtual PCIe link width
alone is not evidence of physical locality or achieved bandwidth.

## Model decision

1. **Speed track: Q2_0.**  Keep it as the 50 tok/s target.  It is explicitly
   designed to avoid the lookup-heavy i-quant decode overhead and is the only
   full model format with a plausible path from the current result.
2. **Quality track: IQ3_S.**  Install it after the RAM/SSD upgrade and benchmark
   it, but do not forecast 50 tok/s from the hardware upgrade.  The first goal
   is to beat the calibrated 26.819 tok/s result without cache or storage
   stalls.
3. **Code-specialized track: Coder IQ1_M.**  Repeat the historical 37.9 tok/s
   full-cache measurement on the current branch.  It is a credible intermediate
   speed candidate, but its reduced expert set is weaker for general chat and
   CJK, so it cannot replace the full model silently.
4. **IQ2_XS / IQ3_XXS.**  Use these only when their quality/latency trade-off is
   wanted.  They are not higher-probability 50 tok/s candidates than Q2_0.

## Ranked experiment queue

Every runtime experiment uses the same prompt, greedy sampling, context, KV
format, MTP files, warm-up, and ten measured 256-token responses.  Report
median, mean, individual rates, draft offered/accepted counts, verify-window
time, clocks, temperature, power, expert-cache occupancy, CPU expert time and
PCIe bytes.  A performance change is retained only when the median improvement
is repeatable and greater than 3% with no parity or quality regression.

| Rank | Experiment | Why it is next | Expected scope |
|---:|---|---|---|
| 1 | Current-branch Coder IQ1_M full-residency retest after RAM upgrade | Historical 37.9 tok/s is the closest unverified operating point to 50; proves whether capacity removes all host-tier stalls | model/runtime, potentially 5-15% from current Q2 reference but model-specific |
| 2 | MTP matrix: default vs `--mtp-q4 proj` / `all`, `--mtp-hnorm pooled` / `stream`, with `STRATA_MTP_PROFILE=1` | Useful tokens per verify window can matter more than another small expert microkernel gain; measure draft latency and acceptance together | potentially several percent; rounding/acceptance may change |
| 3 | `STRATA_LFUSE=1` exact layer-fusion path | It removes launches around the shared expert and KV append, directly targeting the measured small-stage overhead; it has not been measured on MI50 | 0-8%, no format change |
| 4 | `STRATA_SH_STREAM=1` on gfx906 | Other wave64 AMD cards gained from overlapping shared and routed experts, while other HIP systems regressed; MI50 needs its own A/B | 0-10%, architecture/driver-dependent |
| 5 | Compiler-only matrix with the runtime/driver fixed | Separates code-generation gains from unsupported driver changes; compare clang scheduling, VGPR/SGPR/LDS and spills | 0-5% expected, low operational risk when binaries are isolated |
| 6 | Fused wave64 router/plan/doorbell and shared/quantize micrographs | These small stages total about 5 ms per window and are dominated by launch/synchronization latency | 0-8% if barriers are actually removed |
| 7 | Persistent wave64 expert work queue across the verify window | Amortizes dispatch while retaining the compact Q2_0 representation | 0-10%, high implementation risk |
| 8 | Offline MI50 Q2 bit-plane pack prototype | Decode activation planes once and reuse them; the prior per-dot conversion was slower | proceed only if isolated packed-path gain exceeds 10% and storage remains close to Q2_0 |
| 9 | Request batching after residency is protected | The tested two-slot path reduced cache residency and aggregate throughput to 20.454 tok/s | throughput-only; do not use for the 50 tok/s single-stream claim |

The following are already rejected and should not be repeated without a new
mechanism: generic QFUSE, GR split, forced four-wave occupancy, shallower LDS
staging, one-entry expert specialization, forced one-row Q8 geometry, the
current two-slot batch implementation, `--spec 3`, `--spec 6`, and a wider
pre-expanded Q2_0 representation.

## Upgrade acceptance sequence

1. Capture a final pre-upgrade Q2_0 tg256 baseline and hardware telemetry.
2. Install RAM and the passthrough SSD; verify ECC, NUMA/root-complex placement,
   block-device identity, sustained cooling and no swapping.
3. Copy only generated packs and PLE data to the SSD, verify hashes, and retain
   the archival model copy.
4. Re-measure cold start, warm restart and the unchanged Q2_0 tg256 protocol.
5. Run the ranked runtime A/B matrix above before changing kernels.
6. Install IQ3_S only when enough RAM and disk headroom remain; record Q2_0 and
   IQ3_S as separate speed/quality operating points.
7. Commit only source, scripts, compact summaries and reproducible commands.
   Never commit GGUF files, generated packs, raw responses, credentials,
   service configuration, host names or private infrastructure paths.

The 50 tok/s goal is a stretch target, not a forecast.  The evidence says it
requires a change that removes cross-stage latency or increases accepted tokens
per verify window; RAM and SSD upgrades are enabling infrastructure for larger
models and stable residency, not the missing 46-86% warm-decode speedup.
