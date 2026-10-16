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
row and one entry per selected expert. For that shape, layout mode 5 avoids
some of mode 7's token-in-K-loop bookkeeping. The native expert API now carries
the explicit token-row count so this shape can be selected without guessing
from the number of routed entries.

On three representative layers, 120-iteration isolated measurements at one
token and 18 routed groups were bitwise identical and favored mode 5:

| Layer | mode 7 | mode 5 | Improvement |
|---:|---:|---:|---:|
| 0 | 148.0 us | 140.6 us | 5.26% |
| 23 | 148.1 us | 138.7 us | 6.78% |
| 47 | 147.0 us | 139.5 us | 5.38% |

Two-token input retained mode 7 and showed no regression. End-to-end results,
however, did not clear the acceptance threshold. After a fresh load and a
discarded five-run warm-up, ten 256-token runs produced:

| Variant | Median | Mean | Range |
|---|---:|---:|---:|
| mode 7 | 30.927 tok/s | 30.869 tok/s | 29.022-32.137 |
| mode 5 for one token | 31.269 tok/s | 30.582 tok/s | 23.851-32.133 |

The median change was only **+1.11%**, while the mean was lower because of an
outlier. An earlier five-run sample appeared to show +13.55%; the longer repeat
demonstrated that this was warm-cache and run-to-run variance, not a defensible
backend gain.

Consequently mode 7 remains the default. Developers can reproduce the kernel
experiment with:

```bash
STRATA_Q2_SINGLE_MODE=5 ./build/strata ...
```

Unset the variable, or set it to `7`, for the accepted default.

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
