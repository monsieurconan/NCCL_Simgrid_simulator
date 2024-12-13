#ifndef __CUDA_PLATFORM__
#define __CUDA_PLATFORM__

#include "simgrid/s4u.hpp"

const char host_no_deadlock[13] = "no_dead_lock";

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture2D[2];
    int maxTexture3D[3];
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;

  public:
    // static simgrid::xbt::Extension<simgrid::s4u::Host, cudaDeviceProp> EXTENSION_ID;
    cudaDeviceProp();
    double get_speed();
    int parallelisation_degree();
};

simgrid::s4u::NetZone *create_simple_node(simgrid::s4u::NetZone *root, int i_node, int ncpus_cores,
                                          int ngpus, struct cudaDeviceProp gpu_prop,
                                          double cpu_to_gpu_bandwidth, double gpu_to_gpu_bandwidth);

void create_starzone_default(int nspikes, double inter_node_bandwidth, int n_gpus_per_node);

#endif