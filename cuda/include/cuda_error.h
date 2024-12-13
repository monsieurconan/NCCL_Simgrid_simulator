#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorNotReady = 6,
    cudaErrorInvalidResourceHandle = 400
};

#endif