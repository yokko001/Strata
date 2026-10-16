#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kGroups = 1 << 18;
constexpr int kThreads = 256;

__device__ __forceinline__ int dot8_i4(uint32_t a, uint32_t b, int acc) {
    return amd_mixed_dot(static_cast<int>(a), static_cast<int>(b), acc, false);
}

__device__ __forceinline__ int dot4_i8(uint32_t a, uint32_t b, int acc) {
    union Packed4 {
        uint32_t word;
        char4 lanes;
    } pa{a}, pb{b};
    return amd_mixed_dot(pa.lanes, pb.lanes, acc, false);
}

__global__ void twoplane_kernel(const uint32_t* weights, const uint32_t* low,
                                const uint32_t* high, const uint32_t* packed_weight_sum,
                                int32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int lo = dot8_i4(weights[i], low[i], 0);
    const int hi = dot8_i4(weights[i], high[i], 0);
    const uint32_t sums = packed_weight_sum[i >> 2];
    const int sum = int(int8_t(sums >> (8 * (i & 3))));
    out[i] = lo + 16 * hi + 8 * sum;
}

__global__ void dp4a_kernel(const uint32_t* weights0, const uint32_t* weights1,
                            const uint32_t* activations0, const uint32_t* activations1,
                            int32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int sum = dot4_i8(weights0[i], activations0[i], 0);
    out[i] = dot4_i8(weights1[i], activations1[i], sum);
}

__device__ __forceinline__ uint32_t spread_q2_to_i4(uint32_t q) {
    q = (q | (q << 8)) & 0x00ff00ffu;
    q = (q | (q << 4)) & 0x0f0f0f0fu;
    q = (q | (q << 2)) & 0x33333333u;
    return ((q | 0x88888888u) - 0x11111111u) ^ 0x88888888u;
}

__device__ __forceinline__ uint32_t compress_nibbles(uint32_t x) {
    x &= 0x0f0f0f0fu;
    x = (x | (x >> 4)) & 0x00ff00ffu;
    return (x | (x >> 8)) & 0x0000ffffu;
}

__global__ void raw_q2_dot8_kernel(const uint16_t* raw_weights, const uint32_t* activations0,
                                   const uint32_t* activations1, int32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t raw = raw_weights[i];
    const uint32_t w = spread_q2_to_i4(raw);
    uint32_t lo = compress_nibbles(activations0[i]) | (compress_nibbles(activations1[i]) << 16);
    const uint32_t hi = compress_nibbles(activations0[i] >> 4) |
                        (compress_nibbles(activations1[i] >> 4) << 16);
    lo ^= 0x88888888u;
    const int sumw = __popc(raw & 0x5555u) + 2 * __popc(raw & 0xaaaau) - 8;
    out[i] = dot8_i4(w, lo, 0) + 16 * dot8_i4(w, hi, 0) + 8 * sumw;
}

__global__ void raw_q2_dot4_kernel(const uint16_t* raw_weights, const uint32_t* activations0,
                                   const uint32_t* activations1, int32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t q = raw_weights[i];
    const uint32_t qx_idx = (q & 0x03u) | ((q & 0x0cu) << 6) | ((q & 0x30u) << 12) | ((q & 0xc0u) << 18);
    const uint32_t qy = q >> 8;
    const uint32_t qy_idx = (qy & 0x03u) | ((qy & 0x0cu) << 6) | ((qy & 0x30u) << 12) | ((qy & 0xc0u) << 18);
    const uint32_t wx = __builtin_amdgcn_perm(0x020100ffu, 0x020100ffu, qx_idx);
    const uint32_t wy = __builtin_amdgcn_perm(0x020100ffu, 0x020100ffu, qy_idx);
    out[i] = dot4_i8(wy, activations1[i], dot4_i8(wx, activations0[i], 0));
}

uint32_t pack_i4(const int8_t* values) {
    uint32_t packed = 0;
    for (int i = 0; i < 8; ++i) packed |= (uint32_t(values[i]) & 0x0fu) << (4 * i);
    return packed;
}

uint32_t pack_i8(const int8_t* values) {
    uint32_t packed = 0;
    for (int i = 0; i < 4; ++i) packed |= uint32_t(uint8_t(values[i])) << (8 * i);
    return packed;
}

void hip_check(hipError_t e, const char* where) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(e));
        std::exit(2);
    }
}

}  // namespace

int main() {
    std::vector<uint16_t> rawq(kGroups);
    std::vector<uint32_t> w4(kGroups), lo4(kGroups), hi4(kGroups);
    std::vector<uint32_t> w8a(kGroups), w8b(kGroups), x8a(kGroups), x8b(kGroups);
    std::vector<uint32_t> packed_sumw((kGroups + 3) / 4, 0);
    std::vector<int32_t> reference(kGroups), result(kGroups);
    uint32_t state = 0x90650u;
    for (int g = 0; g < kGroups; ++g) {
        int8_t w[8], x[8], lo[8], hi[8];
        int32_t sw = 0, ref = 0;
        for (int j = 0; j < 8; ++j) {
            state = state * 1664525u + 1013904223u;
            const uint32_t q = (state >> 30) & 3u;
            w[j] = int8_t(int(q) - 1);
            rawq[g] |= uint16_t(q << (2 * j));
            state = state * 1664525u + 1013904223u;
            x[j] = int8_t(state >> 24);
            const int u = uint8_t(x[j]) & 15;
            lo[j] = int8_t(u - 8);
            hi[j] = int8_t((int(x[j]) - u) / 16);
            sw += w[j];
            ref += int(w[j]) * int(x[j]);
        }
        w4[g] = pack_i4(w);
        lo4[g] = pack_i4(lo);
        hi4[g] = pack_i4(hi);
        w8a[g] = pack_i8(w);
        w8b[g] = pack_i8(w + 4);
        x8a[g] = pack_i8(x);
        x8b[g] = pack_i8(x + 4);
        packed_sumw[g >> 2] |= uint32_t(uint8_t(int8_t(sw))) << (8 * (g & 3));
        reference[g] = ref;
    }

    uint16_t* drawq;
    uint32_t *dw4, *dlo4, *dhi4, *dw8a, *dw8b, *dx8a, *dx8b;
    uint32_t* dsumw;
    int32_t* dout;
    const size_t words = size_t(kGroups) * sizeof(uint32_t);
    const size_t ints = size_t(kGroups) * sizeof(int32_t);
    hip_check(hipMalloc(&drawq, rawq.size() * sizeof(uint16_t)), "hipMalloc drawq");
    hip_check(hipMalloc(&dw4, words), "hipMalloc dw4");
    hip_check(hipMalloc(&dlo4, words), "hipMalloc dlo4");
    hip_check(hipMalloc(&dhi4, words), "hipMalloc dhi4");
    hip_check(hipMalloc(&dw8a, words), "hipMalloc dw8a");
    hip_check(hipMalloc(&dw8b, words), "hipMalloc dw8b");
    hip_check(hipMalloc(&dx8a, words), "hipMalloc dx8a");
    hip_check(hipMalloc(&dx8b, words), "hipMalloc dx8b");
    hip_check(hipMalloc(&dsumw, packed_sumw.size() * sizeof(uint32_t)), "hipMalloc dsumw");
    hip_check(hipMalloc(&dout, ints), "hipMalloc dout");
    hip_check(hipMemcpy(drawq, rawq.data(), rawq.size() * sizeof(uint16_t), hipMemcpyHostToDevice), "copy drawq");
    hip_check(hipMemcpy(dw4, w4.data(), words, hipMemcpyHostToDevice), "copy dw4");
    hip_check(hipMemcpy(dlo4, lo4.data(), words, hipMemcpyHostToDevice), "copy dlo4");
    hip_check(hipMemcpy(dhi4, hi4.data(), words, hipMemcpyHostToDevice), "copy dhi4");
    hip_check(hipMemcpy(dw8a, w8a.data(), words, hipMemcpyHostToDevice), "copy dw8a");
    hip_check(hipMemcpy(dw8b, w8b.data(), words, hipMemcpyHostToDevice), "copy dw8b");
    hip_check(hipMemcpy(dx8a, x8a.data(), words, hipMemcpyHostToDevice), "copy dx8a");
    hip_check(hipMemcpy(dx8b, x8b.data(), words, hipMemcpyHostToDevice), "copy dx8b");
    hip_check(hipMemcpy(dsumw, packed_sumw.data(), packed_sumw.size() * sizeof(uint32_t), hipMemcpyHostToDevice),
              "copy dsumw");

    const dim3 block(kThreads), grid((kGroups + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(twoplane_kernel, grid, block, 0, 0, dw4, dlo4, dhi4, dsumw, dout, kGroups);
    hip_check(hipDeviceSynchronize(), "twoplane sync");
    hip_check(hipMemcpy(result.data(), dout, ints, hipMemcpyDeviceToHost), "copy result");
    int mismatches = 0;
    for (int i = 0; i < kGroups; ++i) {
        if (result[i] != reference[i] && mismatches++ < 4)
            std::fprintf(stderr, "mismatch %d: gpu=%d cpu=%d\n", i, result[i], reference[i]);
    }
    if (mismatches) {
        std::fprintf(stderr, "FAIL: %d/%d mismatches\n", mismatches, kGroups);
        return 1;
    }
    hipLaunchKernelGGL(raw_q2_dot8_kernel, grid, block, 0, 0, drawq, dx8a, dx8b, dout, kGroups);
    hip_check(hipDeviceSynchronize(), "raw dot8 sync");
    hip_check(hipMemcpy(result.data(), dout, ints, hipMemcpyDeviceToHost), "copy raw result");
    mismatches = 0;
    for (int i = 0; i < kGroups; ++i) {
        if (result[i] != reference[i] && mismatches++ < 4)
            std::fprintf(stderr, "raw mismatch %d: gpu=%d cpu=%d\n", i, result[i], reference[i]);
    }
    if (mismatches) {
        std::fprintf(stderr, "FAIL: %d/%d raw mismatches\n", mismatches, kGroups);
        return 1;
    }

    hipEvent_t begin, end;
    hip_check(hipEventCreate(&begin), "event begin");
    hip_check(hipEventCreate(&end), "event end");
    auto benchmark = [&](bool twoplane) {
        constexpr int kIters = 400;
        hip_check(hipEventRecord(begin), "record begin");
        for (int i = 0; i < kIters; ++i) {
            if (twoplane)
                hipLaunchKernelGGL(twoplane_kernel, grid, block, 0, 0, dw4, dlo4, dhi4, dsumw, dout, kGroups);
            else
                hipLaunchKernelGGL(dp4a_kernel, grid, block, 0, 0, dw8a, dw8b, dx8a, dx8b, dout, kGroups);
        }
        hip_check(hipEventRecord(end), "record end");
        hip_check(hipEventSynchronize(end), "event sync");
        float milliseconds = 0;
        hip_check(hipEventElapsedTime(&milliseconds, begin, end), "elapsed");
        return milliseconds / kIters;
    };
    const float dot8_ms = benchmark(true);
    const float dot4_ms = benchmark(false);
    auto benchmark_raw = [&](bool dot8) {
        constexpr int kIters = 400;
        hip_check(hipEventRecord(begin), "record raw begin");
        for (int i = 0; i < kIters; ++i) {
            if (dot8)
                hipLaunchKernelGGL(raw_q2_dot8_kernel, grid, block, 0, 0, drawq, dx8a, dx8b, dout, kGroups);
            else
                hipLaunchKernelGGL(raw_q2_dot4_kernel, grid, block, 0, 0, drawq, dx8a, dx8b, dout, kGroups);
        }
        hip_check(hipEventRecord(end), "record raw end");
        hip_check(hipEventSynchronize(end), "event raw sync");
        float milliseconds = 0;
        hip_check(hipEventElapsedTime(&milliseconds, begin, end), "raw elapsed");
        return milliseconds / kIters;
    };
    const float raw_dot8_ms = benchmark_raw(true);
    const float raw_dot4_ms = benchmark_raw(false);
    std::printf("PASS: %d exact groups; dot8_two_plane=%.6f ms dot4_i8=%.6f ms ratio=%.3fx\n",
                kGroups, dot8_ms, dot4_ms, dot4_ms / dot8_ms);
    std::printf("PASS: raw Q2_0 exact; raw_dot8=%.6f ms current_dot4=%.6f ms ratio=%.3fx\n",
                raw_dot8_ms, raw_dot4_ms, raw_dot4_ms / raw_dot8_ms);
    return 0;
}
