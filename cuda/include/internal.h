#ifndef CUDA_INTERNAL_H
#define CUDA_INTERNAL_H

#include "gpu_activity.h"
#include "simgrid/s4u.hpp"
#include "vector"

namespace simgrid {
namespace cuda {
enum COPYTYPE { HostToDevice, HostToHost, DeviceToHost, DeviceToDevice };

struct Stream {

    s4u::ActorPtr streamActor;
    s4u::Host *gpu;

  public:
    Stream();
    Stream(s4u::ActorPtr already_started_actor);
    void wait();
    void push(GpuActivityPtr new_activity);
    void push(std::vector<GpuActivityPtr> new_activities);
    std::vector<GpuActivityPtr> pop();
};

struct internalStream {
    static simgrid::xbt::Extension<simgrid::s4u::Actor, internalStream> EXTENSION_ID;
    std::queue<std::vector<GpuActivityPtr>> kernel_calls;
    s4u::Mailbox *stream_mb;
    s4u::Mailbox *cpu_mb;
    int kernel_count = 0;

  public:
    internalStream(s4u::ActorPtr stream_actor, s4u::ActorPtr cuda_actor);
    void wait();
    simgrid::s4u::CommPtr push(GpuActivityPtr new_activity);
    simgrid::s4u::CommPtr push(std::vector<GpuActivityPtr> new_activities);
    std::vector<GpuActivityPtr> pop();
    void complete();
};

struct Graph {
    std::vector<std::vector<GpuActivityPtr>> captured_activities;

  public:
    Graph();
    void clear();
    void add_to_graph(GpuActivityPtr activity);
    void add_to_graph(std::vector<GpuActivityPtr> activities);
    std::vector<std::vector<GpuActivityPtr>> get_captured_activities();
    void destroy();
};

struct GraphExec {
    std::vector<std::vector<GpuActivityPtr>> captured_activities;

  public:
    GraphExec();
    GraphExec(std::vector<std::vector<GpuActivityPtr>> captured_activities_);
    void launch(Stream stream);
    ~GraphExec();
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
    s4u::ActivityPtr send(COPYTYPE type, simgrid::s4u::ActorPtr stream_actor, size_t count);
    void write(size_t count);
};

cudaActor *cuda_process();

} // namespace cuda

} // namespace simgrid

#endif