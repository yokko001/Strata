# MI50-native backend experiments

This document records the second-stage work that treats `gfx906` as a target in
its own right rather than as a compatibility build. It deliberately includes
negative results. A kernel that is faster in isolation is not enabled by
default unless it also improves steady-state text generation on a real MI50.

The stable, accepted changes remain documented in
[MI50_GFX906_OPTIMIZATION.md](MI50_GFX906_OPTIMIZATION.md). Machine-readable
results for this round are in
[`mi50-evidence/gfx906-native-backend-v1-summary.json`](mi50-evidence/gfx906-native-backend-v1-summary.json).

## Test target and acceptance rule

- AMD Instinct MI50 32 GB (`gfx906`, 60 CUs, wave64)
- ROCm 6.2.4 toolchain with clang 18
- Q2_0 MoE model, 20,402 experts in a 26.27 GiB GPU expert cache
- Identical model, cache policy, speculative settings, prompt, and executable
  for each A/B pair
- 32-token warm-up followed by fixed 256-token generation runs
- Bitwise equality for isolated expert-kernel output
- Default enablement requires a repeatable median improvement greater than 3%
  with no correctness loss

The server was fully unloaded between environment-controlled variants so that
only one model occupied the GPU at a time.

## Single-token Q2_0 layout experiment

Autoregressive decode commonly enters the grouped expert kernel with one token
row and one entry per selected expert. A candidate passed the explicit token-row
count into the native expert API and selected layout mode 5 for that shape.

An initial process-to-process measurement appeared to favor mode 5 by 5.26-6.78%
on three layers. That result did not reproduce after a clean final build. A
same-process, 120-iteration comparison at one token and 18 routed groups instead
favored the existing mode 7 on every layer; output remained bitwise identical:

| Layer | mode 7 | mode 5 | mode 5 change |
|---:|---:|---:|---:|
| 0 | 133.5 us | 141.2 us | +5.77% slower |
| 23 | 135.4 us | 141.9 us | +4.80% slower |
| 47 | 132.1 us | 138.2 us | +4.62% slower |

The end-to-end candidate also failed the acceptance threshold. After a fresh
load and a discarded five-run warm-up, ten 256-token runs produced:

| Variant | Median | Mean | Range |
|---|---:|---:|---:|
| mode 7 | 30.927 tok/s | 30.869 tok/s | 29.022-32.137 |
| mode 5 for one token | 31.269 tok/s | 30.582 tok/s | 23.851-32.133 |

The median change was only **+1.11%**, while the mean was lower because of an
outlier. An earlier five-run sample appeared to show +13.55%; the longer repeat
demonstrated that this was warm-cache and run-to-run variance, not a defensible
backend gain. The dispatch and API changes were therefore fully reverted; mode
7 remains the only default path and no experimental runtime switch is shipped.

## Aggressive designs evaluated

### Pre-expanded MI50-only Q2_0 pack

MI50 has no native INT2 matrix instruction. Eliminating bit unpacking entirely
therefore requires storing wider values. Expanding Q2_0 weights to byte lanes
would multiply the expert payload and reduce the effective expert-cache and PCIe
bandwidth by roughly the same factor. The tested model already occupies about
34 GiB before runtime working space, while its GPU expert cache uses 26.27 GiB
on a 32 GiB card. A permanently expanded pack is therefore rejected for this
streamed-MoE workload: it trades a small ALU cost for an unaffordable memory and
transfer cost.

A density-preserving bit-plane reorder remains possible, but it cannot remove
the unpack step; it can only change its instruction pattern. That work should
start with a pack converter plus a microbenchmark and must not replace the
portable Q2_0 files until end-to-end results pass the same gate.

### Decode reuse across grouped entries

Reusing decoded Q2_0 blocks across grouped gate/up entries was correct, but
improved the isolated kernel by only 0.6-1.5% and was inconsistent across
layers. Extending the reuse to the down projection was about 1% faster at two
tokens and slower at four and eight tokens. Both changes were reverted.

### Fused gate/up, activation, and requantization

A dedicated mode fused gate/up output, activation, and activation quantization.
It was bitwise correct but 8-11% slower than the accepted mode 7 path. Increased
register pressure and synchronization outweighed the removed launch and
temporary traffic, so it was reverted.

### Packed multi-token dot8 path

A gfx906 path staged and repacked activations for eight-lane integer dot work.
It compiled and was bitwise correct, but was slower at two tokens and about 20%
slower at four and eight tokens. The repack/correction overhead dominated at
the batch sizes relevant to interactive generation, so it was reverted.

### Routing and compiler specialization

The earlier MI50 branch already carries the wave64 Top-K correctness bound and
the accepted LDS Q2_0 gate/up kernel. More aggressive routing/Q-fusion remains
disabled because its measured end-to-end result was negative. The build now
also corrects the pinned llama.cpp FP8 feature guard for ROCm 6.2: FP8 types are
only referenced when the HIP version actually provides them. This makes a
clean `gfx906` build reproducible without relying on a dirty dependency tree.

### LDS depth and forced occupancy

Reducing the grouped-expert LDS staging depth from four token rows to one was
bitwise correct but slowed the Q2_0 down projection from roughly 59-62 us to
70-73 us. The candidate was reverted.

Forcing four waves per SIMD with `amdgpu_waves_per_eu(4)` reduced the reported
VGPR allocation from about 80 to 64, but the compiler introduced 18-19 VGPR
spills. The one-token grouped-expert microbenchmark regressed from 132-135 us
to 183-188 us, about 40%, so occupancy was not forced in the retained source.

### Specialized one-entry Q2_0 kernels

A second candidate compiled dedicated one-entry gate/up and down kernels for
the common single-token decode shape. It removed the generic one-to-four-entry
switch and reduced register use without spills:

| Kernel | Generic | One-entry specialized |
|---|---:|---:|
| gate/up | 74 SGPR, 81 VGPR | 57 SGPR, 58 VGPR |
| down | 58 SGPR, 80 VGPR | 49 SGPR, 54 VGPR |

Five interleaved layer-23 runs improved the isolated grouped-expert median from
138.0 us to 133.9 us (**+3.06%**); gate/up improved by about 8.3% and down by
about 13.1%. Outputs were bitwise equal, and two-, four-, and eight-token inputs
fell back to the generic path with bitwise equality.

The gain did not reach model throughput. With the same executable, five
discarded warm-up runs and ten measured 256-token runs, both the generic mode
and the specialized mode produced a median of **32.867 tok/s**. The code and
the temporary native API extension were therefore reverted.

### Continuous batching and speculative decode

The existing two-slot continuous-batching path was also tested before adding a
new persistent queue. It reduced the expert cache from 20,402 entries
(26.27 GiB) to 19,531 entries (25.15 GiB). Five pairs of concurrent 256-token
requests achieved only 20.454 tok/s median aggregate throughput, below the
single-request baseline even before considering per-request latency. A
four-slot test was skipped because its additional KV state would reduce expert
residency further.

The retained speculative settings were checked around the current operating
point with ten measured 256-token generations per candidate:

| Candidate | Median | Change from 32.867 tok/s baseline |
|---|---:|---:|
| `--spec 3 --spec-min-p 0.5` | 31.656 tok/s | -3.7% |
| `--spec 4 --spec-min-p 0.7` | 32.379 tok/s | -1.5% |
| `--spec 6 --spec-min-p 0.5` | 26.207 tok/s | -20.3% |

The current `--spec 4 --spec-min-p 0.5` remains the best tested point. A
`--pipeline-windows 2` probe was rejected at startup because this target is a
single 32 GB MI50 rather than a two-GPU layer split; the engine correctly left
the feature off and no timing was attributed to it.

### Decode profile and Q8_0 dense GEMV geometry

A refreshed GPU-stage profile confirmed that warm decode is GPU-bound. CPU
experts and PCIe expert traffic were effectively zero. Representative GDN
stage costs per verify window included about 3.0 ms for Q8 QKV GEMV, 2.8 ms for
router/plan/doorbell, 2.1 ms for the head, 1.9 ms for `z`, 1.9 ms for
shared+quant, and 1.7 ms for the output projection. This explains why a 3%
isolated expert-kernel gain can disappear in end-to-end generation.

The documented `STRATA_NO_MMVQ_ROWS2=1` override was found not to force the
single-output-row Q8_0 geometry on unrecognized architectures: it still applied
the gfx12/sm86 allow-list. The override now honors its documented meaning. Both
MMVQ multi-column parity tests pass with the override off and on; default device
selection is unchanged.

The corrected override was used to measure the real model shapes on gfx906. At
the common two- and three-column sizes, one-row blocks improved GDN QKV by
3.1-4.8% and GDN gate by 6.0-8.3%. The result was not general: QSA `q` changed
by +1.0%/-2.3%, the vocabulary head slowed 1.3-1.8%, and several small shapes
were noisy or materially slower. The estimated model-level contribution is
below the 3% acceptance threshold, so gfx906 keeps the two-row default and the
override remains an explicit diagnostic control.

## 2026-10-17 runtime and host-resource matrix

A fresh same-binary control was allowed to converge through the adaptive expert
cache before measurement.  Every retained number below is a ten-run median of
256 generated tokens.  The compact, path-free evidence is stored in
[`mi50-evidence/2026-10-17-runtime-matrix.json`](mi50-evidence/2026-10-17-runtime-matrix.json).

| Candidate | Median | Change from 38.051 control | Decision |
|---|---:|---:|---|
| Control | 38.051 tok/s | - | baseline |
| `STRATA_LFUSE=1` | 37.508 tok/s | -1.43% | reject |
| `STRATA_SH_STREAM=1` | 31.834 tok/s | -16.34% | reject |
| `--mtp-q4 proj`, first / repeat | 38.859 / 38.588 tok/s | +2.12% / +1.41% | opt-in research candidate; below gate |
| `--mtp-q4 all` | 38.413 tok/s | +0.95% | reject |

`--mtp-hnorm stream` was stopped before a scored final set: two twelve-run
adaptation series had medians of 30.032 and 37.475 tok/s and still ranged from
16.650 to 40.567 tok/s after 24 runs.  Its unstable cache behavior is worse
than the pooled default.  Q4 projection remains worth revisiting only together
with draft acceptance/profile data; its observed gain is too small to change a
public default.

The host inventory explains why RAM and storage changes must be evaluated
separately from warm decode.  The guest currently has eight available AVX-only
CPU cores across two virtual sockets, 21 GiB RAM and 3.8 GiB swap.  Automatic
placement uses seven pool workers plus the host thread and 24 expert tasks per
phase.  In the warmed Q2_0 run, the 8.68 GiB host complement was resident in
ordinary pageable RAM and Strata reported zero expert-file reads.  The raw GGUF
was read from CIFS at about 127 MB/s, while the pack lived on a virtual
rotational SCSI disk.  Consequently:

- a 64 GiB RAM addition should remove swap pressure and permit IQ3_S/model page
  residency, but cannot remove the measured warm GPU stages;
- a passthrough SSD should materially reduce cold load, conversion and page
  fault latency, but must not be credited as warm tok/s without measurement;
- the next CPU sweep should compare worker count, host participation, task
  count and first/last host core because the VM spans two virtual sockets;
- model and pack placement on the new SSD must be hash-verified, while service
  credentials and machine-specific paths remain outside the repository.

## 2026-10-18 72 GiB IQ3_S and v0.1.45 validation

The guest memory increase made the full IQ3_S expert arena practical: the
process resident set was about 50.5 GiB and the 46.84 GiB expert arena remained
resident without process swap.  This changes the preferred quality/speed
operating point from Q2_0 to IQ3_S, but it does not make warm decode a storage
workload.  No SSD claim is made here.

The upstream v0.1.45 tree was rebuilt for gfx906 after adding the omitted
`src/prefill/moe_f16_kq.cu` translation unit to `strata_mmq`.  The build and the
three selected file-source, cache-slot and native-multi parity tests passed.
End-to-end single-request IQ3_S performance did not pass the deployment gate:

| Candidate | Median | Change / decision |
|---|---:|---|
| v0.1.40.1 reference | 29.297 tok/s | deployed reference |
| v0.1.45 default, fully warm | 25.046 tok/s | -14.51%; reject |
| v0.1.45 with old grouped path | 22.545 tok/s | reject |
| v0.1.45 with route-tail skip disabled | 13.883 tok/s | reject |
| v0.1.45 with the new gfx906 Top-K path | 21.377 tok/s | reject for deployment |

The new Top-K implementation remains a correctness fix, but forcing it in this
workload was not a speed win.  The service therefore stays on v0.1.40.1 with
the conservative old Top-K runtime selection while v0.1.45's gfx906
single-stream regression is investigated.  Compact rates and test metadata are
in
[`mi50-evidence/2026-10-18-iq3s-72g-v0145.json`](mi50-evidence/2026-10-18-iq3s-72g-v0145.json).

This rollout also exposed an operational benchmark hazard: two independently
enabled model services can consume most guest RAM and make both decode and SSH
unreliable.  Before every retained measurement, verify that exactly one model
engine is active.  Do not infer a kernel regression from a run made while a
second engine is loading or resident.

## Reproduction guidance

Build with the native gfx906 target and avoid compiling unsupported optional
targets:

```bash
cmake -S . -B build-mi50 -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-mi50 --target strata native_expert_bench
```

For generation measurements, use a fixed prompt and settings, discard the first
post-load series, then report both median and mean over at least ten 256-token
runs. Record individual values as well: MI50 expert-cache warming can make a
small sample look more than ten percentage points better than the repeat.

## Next high-value directions

The remaining work should focus on changes that increase useful residency or
amortize expert dispatch without inflating the streamed weight representation:

1. persistent wave64 work queues spanning multiple concurrent requests;
2. request-level continuous batching with latency bounds, measured at 2, 4,
   and 8 active decode rows;
3. a density-preserving Q2_0 bit-plane pack prototype with an offline converter;
4. LDS/register-pressure tuning driven by generated gfx906 ISA and occupancy;
5. expert-cache admission/prefetch driven by observed router locality.

Each should land behind an opt-in switch first and graduate only after the
bitwise and repeated end-to-end gates above pass.
