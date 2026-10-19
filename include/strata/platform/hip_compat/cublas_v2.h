// cublas_v2.h for a STRATA_ENABLE_HIP build: the cuBLAS names src/prefill/gemm.cu uses, on hipBLAS.
#pragma once
#include "strata_hip.h"
#ifndef HIPBLAS_V2
#define HIPBLAS_V2
#endif
#include <hipblas/hipblas.h>

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasSgemm hipblasSgemm
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#if HIP_VERSION < 60300000
// ROCm 6.2's hipblasGemmStridedBatchedEx still takes hipblasDatatype_t for computeType.
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#else
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#endif
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_16BF HIP_R_16BF
// rocBLAS picks its own math and workspace; the NVIDIA-only knobs are accepted and ignored
#define CUBLAS_DEFAULT_MATH 0
#define cublasSetMathMode(handle, mode) HIPBLAS_STATUS_SUCCESS
#define cublasSetWorkspace(handle, ptr, bytes) HIPBLAS_STATUS_SUCCESS
