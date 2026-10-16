#include <hip/hip_runtime.h>

// Compile-only probe for the ROCm compiler used by the MI50 host.  The volatile
// stores keep both paths visible in the generated gfx906 ISA so llvm-objdump can
// prove whether the packed signed-byte dot product lowers to v_dot4_i32_i8.
extern "C" __global__ void gfx906_dot4_probe(const int* a, const int* b,
                                               const int* c, int* native_out,
                                               int* scalar_out) {
    const int x = a[threadIdx.x];
    const int y = b[threadIdx.x];
    const int acc = c[threadIdx.x];

#if __has_builtin(__builtin_amdgcn_sdot4)
    native_out[threadIdx.x] = __builtin_amdgcn_sdot4(x, y, acc, false);
#else
#error "ROCm compiler does not expose __builtin_amdgcn_sdot4"
#endif

    const unsigned ux = static_cast<unsigned>(x);
    const unsigned uy = static_cast<unsigned>(y);
    unsigned sum = static_cast<unsigned>(acc);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const int sx = static_cast<signed char>(ux >> (lane * 8));
        const int sy = static_cast<signed char>(uy >> (lane * 8));
        sum += static_cast<unsigned>(sx * sy);
    }
    scalar_out[threadIdx.x] = static_cast<int>(sum);
}
