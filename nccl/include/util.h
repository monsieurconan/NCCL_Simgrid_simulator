#ifndef __NCCL_TEST_UTIL__
#define __NCCL_TEST_UTIL__

#include "cuda.h"

#define CUDACHECK(cmd)                                                                             \
    do {                                                                                           \
        cudaError_t err = cmd;                                                                     \
        if (err != cudaSuccess) {                                                                  \
            char hostname[1024];                                                                   \
            getHostName(hostname, 1024);                                                           \
            printf("%s: Test CUDA failure %s:%d '%s'\n", hostname, __FILE__, __LINE__,             \
                   cudaGetErrorString(err));                                                       \
            return testCudaError;                                                                  \
        }                                                                                          \
    } while (0)

enum COLL {
    REDUCE,
    ALL_REDUCE,
    BROADCAST,
    REDUCE_SCATTER,
    ALL_GATHER,
    GATHER,
    SCATTER,
    ALL_TO_ALL,
    SENDRECV,
    DEBUG
};

double getAlgoBandwidth(double time, int size) { return (double)(size)*1e-9 / time; }

double getBusBandwidth(COLL coll, double algoBandWidth, int N_RANKS) {
    // according to nccl-tests calculations
    switch (coll) {
    case REDUCE:
        return algoBandWidth;
    case ALL_REDUCE:
        return algoBandWidth * 2 * (N_RANKS - 1) / N_RANKS;
    case BROADCAST:
        return algoBandWidth;
    case REDUCE_SCATTER:
        return algoBandWidth * 2 * (N_RANKS - 1) / N_RANKS;
    case ALL_GATHER:
        return algoBandWidth * 2 * (N_RANKS - 1) / N_RANKS;
    case GATHER:
        return algoBandWidth * 2 * (N_RANKS - 1) / N_RANKS;
    case SCATTER:
        return algoBandWidth * 2 * (N_RANKS - 1) / N_RANKS;
    case ALL_TO_ALL:
        return algoBandWidth * 2 * (N_RANKS - 1) / N_RANKS;
    case SENDRECV:
        return algoBandWidth;
    default:
        return algoBandWidth;
    }
}

#endif