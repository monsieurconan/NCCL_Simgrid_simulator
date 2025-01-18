#include "cuda_runtime.h"
#include "internal.h"
#include "platform.h"
#include "simgrid/s4u.hpp"

cudaError_t cudaSetDevice(int device) {
    simgrid::cuda::cuda_process()->setDevice(device);
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
    auto device = simgrid::cuda::cuda_process()->getCurrentDevice();
    device->turn_off();
    device->turn_on();
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    auto streams = simgrid::cuda::cuda_process()->getCurrentDevice()->get_all_actors();
    for (int i = 0; i < simgrid::cuda::cuda_process()->getCurrentDevice()->get_actor_count(); ++i) {
        streams[i]->extension<simgrid::cuda::internalStream>()->wait();
    }
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaStreamSynchronise(cudaStream_t stream) {
    stream->stream.wait();
    return cudaSuccess;
}

/*memory management : should we add later an option to fold it like in SMPI ?*/

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    // implicit synchronisation todo
    if (devPtr == nullptr) return cudaError_t::cudaErrorInitializationError;
    *devPtr = malloc(size);
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    // implicit synchronisation todo
    *ptr = malloc(size);
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    free(devPtr);
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaError_t::cudaSuccess;
}

/*
mem cpy*/

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    memcpy(dst, src, count); // shallow copy ?
    simgrid::cuda::COPYTYPE type;
    cudaDeviceSynchronize();
    switch (kind) {
    case cudaMemcpyHostToDevice:
        type = simgrid::cuda::HostToDevice;
        break;
    case cudaMemcpyHostToHost:
        type = simgrid::cuda::HostToHost;
        break;
    case cudaMemcpyDeviceToHost:
        type = simgrid::cuda::DeviceToHost;
        break;
    case cudaMemcpyDeviceToDevice:
        type = simgrid::cuda::DeviceToDevice;
        break;
    default:
        break;
    }
    auto stream_actor = simgrid::cuda::cuda_process()->getCurrentDevice()->get_all_actors()[0];
    auto comm = simgrid::cuda::cuda_process()->send(type, stream_actor, count);
    stream_actor->extension<simgrid::cuda::internalStream>()->wait();
    comm->wait();
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream) {
    memcpy(dst, src, count); // shallow copy ?
    simgrid::cuda::COPYTYPE type;
    cudaDeviceSynchronize();
    switch (kind) {
    case cudaMemcpyHostToDevice:
        type = simgrid::cuda::HostToDevice;
        break;
    case cudaMemcpyHostToHost:
        type = simgrid::cuda::HostToHost;
        break;
    case cudaMemcpyDeviceToHost:
        type = simgrid::cuda::DeviceToHost;
        break;
    case cudaMemcpyDeviceToDevice:
        type = simgrid::cuda::DeviceToDevice;
        break;
    default:
        break;
    }
    simgrid::cuda::cuda_process()->send(type, stream->stream.streamActor, count)->wait();
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    // std::fill_n(devPtr, value, count);
    simgrid::cuda::cuda_process()->write(count);
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    //*prop = *simgrid::cuda::cuda_process()->getCurrentDevice()->extension<cudaDeviceProp>();
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
    (*pStream) = new cudaStream();
    return cudaSuccess;
}

cudaStream::cudaStream() : mode{cudaStreamCaptureModeRelaxed} {}

cudaStream::cudaStream(int flags) : mode{cudaStreamCaptureModeRelaxed} {
    graph = simgrid::cuda::Graph();
    // todo: the flags
}

void cudaStream::launch(simgrid::cuda::GpuActivity new_activity) {
    if (mode == cudaStreamCaptureModeRelaxed) {
        stream.push(new_activity);
    } else {
        graph.add_to_graph(new_activity);
    }
}

void cudaStream::launch(std::vector<simgrid::cuda::GpuActivity> new_activities) {
    if (mode = cudaStreamCaptureModeRelaxed) {
        stream.push(new_activities);
    } else {
        graph.add_to_graph(new_activities);
    }
}
