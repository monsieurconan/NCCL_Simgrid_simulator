#ifndef CUDA_INTERNAL_H
#define CUDA_INTERNAL_H

#include "gpu_activity.h"
#include "simgrid/s4u.hpp"
#include "vector"

namespace simgrid {
namespace cuda {
enum HostType { HOST, DEVICE };

struct Stream {

    s4u::ActorPtr streamActor;
    s4u::Host *gpu;

  public:
    Stream();
    void wait();
    void push(GpuActivity new_activity);
    void push(std::vector<GpuActivity> new_activities);
    std::vector<GpuActivity> pop();
};

struct internalStream {
    static simgrid::xbt::Extension<simgrid::s4u::Actor, internalStream> EXTENSION_ID;
    std::queue<std::vector<GpuActivity>> kernel_calls;
    s4u::Mailbox *stream_mb;
    s4u::Mailbox *cpu_mb;
    int kernel_count = 0;

  public:
    internalStream(s4u::ActorPtr stream_actor, s4u::ActorPtr cuda_actor);
    void wait();
    simgrid::s4u::CommPtr push(GpuActivity new_activity);
    simgrid::s4u::CommPtr push(std::vector<GpuActivity> new_activities);
    std::vector<GpuActivity> pop();
    void complete();
};

struct Graph {
    std::vector<GpuActivity> captured_activities;

  public:
    Graph();
    void clear();
    void add_to_graph(GpuActivity activity);
    void add_to_graph(std::vector<GpuActivity> activities);
    std::vector<GpuActivity> get_captured_activities();
    void destroy();
};

struct GraphExec {
    std::vector<GpuActivity> captured_activities;

  public:
    GraphExec();
    GraphExec(std::vector<GpuActivity> captured_activities_);
    void launch(Stream stream);
};

enum MODE { NORMAL, CAPTURE };

class cudaActor {
    std::vector<s4u::Host *> devices;
    int current_device_index;

  public:
    s4u::Actor *actor_ = nullptr;
    static simgrid::xbt::Extension<simgrid::s4u::Actor, cudaActor> EXTENSION_ID;
    cudaActor(s4u::Actor *actor);
    void setDevice(int);
    s4u::Host *getCurrentDevice();
    std::vector<s4u::Host *> getAllDevice();
    void send(HostType src, HostType dst, size_t count);
    void send_async(HostType src, HostType dst, size_t count, Stream stream);
    void write(size_t count);
};

cudaActor *cuda_process();

} // namespace cuda

} // namespace simgrid

#endif