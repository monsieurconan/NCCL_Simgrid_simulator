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
        if (!streams[i]->is_suspended())
            ;
        simgrid::s4u::this_actor::suspend();
    }
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaStreamSynchronise(cudaStream_t stream) {
    if (!stream->stream.isEmpty()) simgrid::s4u::this_actor::suspend();
    simgrid::s4u::this_actor::sleep_for(1e-9); // deadlock otherwise
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
    simgrid::cuda::HostType sender, recver;
    switch (kind) {
    case cudaMemcpyHostToDevice:
        sender = simgrid::cuda::HostType::HOST;
        recver = simgrid::cuda::HostType::DEVICE;
        break;
    case cudaMemcpyHostToHost:
        sender = simgrid::cuda::HostType::HOST;
        recver = simgrid::cuda::HostType::HOST;
        break;
    case cudaMemcpyDeviceToHost:
        sender = simgrid::cuda::HostType::DEVICE;
        recver = simgrid::cuda::HostType::HOST;
        break;
    case cudaMemcpyDeviceToDevice:
        sender = simgrid::cuda::HostType::DEVICE;
        recver = simgrid::cuda::HostType::DEVICE;
        break;
    default:
        break;
    }
    simgrid::cuda::cuda_process()->send(sender, recver, count);
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream) {
    memcpy(dst, src, count); // shallow copy ?
    simgrid::cuda::HostType sender, recver;
    switch (kind) {
    case cudaMemcpyHostToDevice:
        sender = simgrid::cuda::HostType::HOST;
        recver = simgrid::cuda::HostType::DEVICE;
        break;
    case cudaMemcpyHostToHost:
        sender = simgrid::cuda::HostType::HOST;
        recver = simgrid::cuda::HostType::HOST;
        break;
    case cudaMemcpyDeviceToHost:
        sender = simgrid::cuda::HostType::DEVICE;
        recver = simgrid::cuda::HostType::HOST;
        break;
    case cudaMemcpyDeviceToDevice:
        sender = simgrid::cuda::HostType::DEVICE;
        recver = simgrid::cuda::HostType::DEVICE;
        break;
    default:
        break;
    }
    simgrid::cuda::cuda_process()->send_async(sender, recver, count, stream->stream);
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

/*void cudaStream_t::launch(simgrid::cuda::GpuActivity new_activity) {
    if(mode=cudaStreamCaptureModeRelaxed){
        stream.push(new_activity);
    }
    else{
        graph.add_to_graph(new_activity);
    }
}

void cudaStream_t::launch(std::vector<simgrid::cuda::GpuActivity> new_activities) {
    if(mode=cudaStreamCaptureModeRelaxed){
        stream.push(new_activities);
    }
    else{
        graph.add_to_graph(new_activities);
    }
}*/
