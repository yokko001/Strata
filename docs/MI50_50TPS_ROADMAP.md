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

The current guest exposes 8 CPU cores across two virtual sockets, 21 GiB RAM
and 3.8 GiB swap.  Strata automatically uses seven expert-pool workers plus the
host thread, so all eight CPUs participate.  The current Q2_0 run keeps 8.68
GiB of the GPU-cache complement in pageable RAM and reports zero file reads
after warm-up.  The GGUF shards are still on CIFS (about 127 MB/s in a direct
2 GiB read); the generated pack is on a 300 GB virtual SCSI disk reported as
rotational.  These facts make cold start storage-bound but do not make warm
decode storage-bound.

## Measured starting points

| Model / path | Steady result | Interpretation |
|---|---:|---|
| Qwen3.8-Flash-Next Q2_0, unmodified base | 30.503 tok/s | fixed tg256 baseline |
| Qwen3.8-Flash-Next Q2_0, retained gfx906 path | 33.930-34.294 tok/s | accepted LDS gate/up path, +11.2-12.4% |
| Qwen3.8-Flash-Next Q2_0, latest experiment control | 32.867 tok/s | same-binary control used for recent A/B experiments |
| Qwen3.8-Flash-Next Q2_0, fully warmed 2026-10-17 control | 38.051 tok/s | ten-run median after adaptive-cache convergence; current comparison baseline |
| Qwen3.8-Flash-Next IQ3_S, v0.1.39 | 19.6 tok/s | 13,397 cached experts, 97.6-98.4% hit rate |
| Qwen3.8-Flash-Next IQ3_S, calibrated v0.1.40.1 | 26.819 tok/s | best recorded IQ3_S operating point |
| Qwen3.8-Flash-Next Coder IQ1_M, full cache | 37.2-37.9 tok/s | historical full-cache result; must be repeated on the current branch |

The Q2_0 target therefore needs about **1.31x** over the newest fully warmed
control (or about **1.46x** over the earlier retained-kernel median).
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

If the whole 64 GB addition is assigned to the VM, it takes this guest from
about 21 GiB to about 85 GiB.  That is valuable for IQ3_S and other larger
formats because the host expert complement, dense state and KV allocation can
coexist without swap.  It does **not** make every IQ3_S file page resident:
the 54.8 GB transformer data plus the 28.8 GB PLE shard already total 83.6 GB
before the OS, server, KV state and working buffers.  The practical design is
to keep the active expert complement in RAM and serve the PLE mmap working set
from the passthrough SSD.  This permits a fair current-branch Coder and IQ3
comparison, but does not enlarge MI50 VRAM or reduce the dense and routed GPU
kernels in a warm Q2_0 window.

A low-latency PCIe-passthrough SSD should hold active packs, the PLE file,
generated model data and builds.  It improves cold readiness, mmap faults, PLE
tail latency, pack conversion and cache-miss recovery.  The canonical raw model
may remain on slower archival storage.  Once the needed pages and experts are
resident, SSD bandwidth is not part of the measured steady Q2_0 critical path.

After installation, record physical root-complex placement outside the guest,
guest block topology, direct-read bandwidth and latency, model cold-start time,
warm-restart time, major faults, and steady decode.  A virtual PCIe link width
alone is not evidence of physical locality or achieved bandwidth.

`tools/mi50/host_inventory.py` captures the comparable guest-side subset without
publishing host names, mount points, device names or source paths.  For example:

```bash
python3 tools/mi50/host_inventory.py \
  --path pack=/path/to/packs/q2_0 \
  --path model=/path/to/model.gguf > host-inventory.json
```

Run it before and after the upgrade.  The labels are retained, but the supplied
paths are not written to the JSON.

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
| 1 | Compiler-only matrix with the runtime/driver fixed | Separates code-generation gains from unsupported driver changes; compare clang scheduling, VGPR/SGPR/LDS and spills | 0-5% expected, low operational risk when binaries are isolated |
| 2 | Fused wave64 router/plan/doorbell and shared/quantize micrographs | These small stages total about 5 ms per window and are dominated by launch/synchronization latency | 0-8% if barriers are actually removed |
| 3 | Persistent wave64 expert work queue across the verify window | Amortizes dispatch while retaining the compact Q2_0 representation | 0-10%, high implementation risk |
| 4 | Current-branch Coder IQ1_M full-residency retest after RAM upgrade | Historical 37.9 tok/s is the closest unverified operating point to 50; proves whether capacity removes all host-tier stalls | model/runtime, potentially 5-15% from the earlier Q2 reference but model-specific |
| 5 | CPU placement sweep: 4-7 workers, host first/last, host participation, task count | The VM spans two four-core virtual sockets; measure rather than assume that cross-socket workers help | likely small in warm Q2 because CPU expert time is already near zero; may reduce tail latency |
| 6 | RAM/SSD acceptance matrix after installation | Compare swap disabled for the service, resident-page coverage, cold/warm start, major faults and direct I/O latency with identical decode settings | startup and larger-model stability; little expected warm-Q2 gain |
| 7 | Offline MI50 Q2 bit-plane pack prototype | Decode activation planes once and reuse them; the prior per-dot conversion was slower | proceed only if isolated packed-path gain exceeds 10% and storage remains close to Q2_0 |
| 8 | Request batching after residency is protected | The tested two-slot path reduced cache residency and aggregate throughput to 20.454 tok/s | throughput-only; do not use for the 50 tok/s single-stream claim |

The following are already rejected and should not be repeated without a new
mechanism: generic QFUSE, GR split, forced four-wave occupancy, shallower LDS
staging, one-entry expert specialization, forced one-row Q8 geometry, the
current two-slot batch implementation, `--spec 3`, `--spec 6`, a wider
pre-expanded Q2_0 representation, `STRATA_LFUSE=1`, `STRATA_SH_STREAM=1`,
`--mtp-q4 all`, and `--mtp-hnorm stream`.  `--mtp-q4 proj` produced only a
1.4-2.1% median gain and remains an opt-in research candidate below the 3%
retention threshold.

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
