#pragma once

// Minimal CUDA Runtime ABI declarations required by TensorRT headers.
// CUDA functions are loaded dynamically in TensorRtYoloBackend.cpp, so the
// full CUDA Toolkit headers and import library are not required at build time.

typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUgraph_st* cudaGraph_t;
typedef struct CUgraphExec_st* cudaGraphExec_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};
