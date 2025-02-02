#include "internal.h"
#include <iostream>
#include <simgrid/s4u.hpp>
#include <simgrid/s4u/Activity.hpp>
#include <stddef.h>

namespace simgrid {
namespace cuda {
static void *dummypayload = malloc(2048);

simgrid::xbt::Extension<simgrid::s4u::Actor, cudaActor> cudaActor::EXTENSION_ID =
    simgrid::s4u::Actor::extension_create<simgrid::cuda::cudaActor>();
simgrid::xbt::Extension<simgrid::s4u::Actor, internalStream> internalStream::EXTENSION_ID =
    simgrid::s4u::Actor::extension_create<simgrid::cuda::internalStream>();

cudaActor *cuda_process() {
    simgrid::s4u::ActorPtr me = simgrid::s4u::Actor::self();

    if (me == nullptr) // This happens sometimes (eg, when linking against NS3 because it pulls
                       // openMPI...)
        return nullptr;

    return me->extension<cudaActor>();
}

namespace {
bool char_equal(const char a[], const char b[], int n) {
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

std::vector<s4u::Host *> filter_gpus(std::vector<s4u::Host *> all_hosts) {
    std::vector<s4u::Host *> res;
    for (int i = 0; i < all_hosts.size(); ++i) {
        if (char_equal(all_hosts[i]->get_property("type"), "gpu", 3)) res.push_back(all_hosts[i]);
    }
    return res;
}

} // namespace

cudaActor::cudaActor(s4u::Actor *cpu) {
    if (not simgrid::cuda::cudaActor::EXTENSION_ID.valid())
        simgrid::cuda::cudaActor::EXTENSION_ID =
            simgrid::s4u::Actor::extension_create<simgrid::cuda::cudaActor>();
    devices = filter_gpus(cpu->get_host()->get_englobing_zone()->get_all_hosts());
}

s4u::ActivityPtr cudaActor::send(COPYTYPE type, simgrid::s4u::ActorPtr stream_actor, size_t count) {
    auto mb = type == HostToDevice ? stream_actor->extension<internalStream>()->stream_mb
                                   : stream_actor->extension<internalStream>()->cpu_mb;
    switch (type) {
    case HostToDevice:
        stream_actor->extension<internalStream>()->push(
            {CreateGpuComm(mb, count, COMM_TYPE::RECV),
             CreateGpuIO(count, IO_TYPE::WRITE)});
        return mb->put_init(dummypayload, count)->start();
    case HostToHost:
        return s4u::Comm::sendto_async(s4u::this_actor::get_host(), s4u::this_actor::get_host(),
                                       count);
    case DeviceToHost:
        stream_actor->extension<internalStream>()->push(
            {CreateGpuIO(count, IO_TYPE::READ),
             CreateGpuComm(mb, count, COMM_TYPE::SEND)});
        return mb->get_init()->start();
    case DeviceToDevice:
        stream_actor->extension<internalStream>()->push(
            {CreateGpuIO(count, IO_TYPE::READ),
             CreateGpuIO(count, IO_TYPE::WRITE)});
    }
    return s4u::this_actor::exec_async(1);
}

void cudaActor::write(size_t count) {
    // todo add disk or use loop back
}

void cudaActor::setDevice(int device_index) { current_device_index = device_index; }

s4u::Host *cudaActor::getCurrentDevice() { return devices[current_device_index]; }

std::vector<s4u::Host *> cudaActor::getAllDevice() { return devices; }

void gpuActor() {
    auto father = s4u::Actor::by_pid(s4u::this_actor::get_ppid());
    simgrid::s4u::ActorPtr me = simgrid::s4u::Actor::self();
    s4u::ActivitySet act_set;
    while (true) {
        auto activities = me->extension<internalStream>()->pop();
        if (activities.size() > 0) {
            for (int i = 0; i < activities.size(); ++i) {
                act_set.push(activities[i]->start());
                delete activities[i];
            }
            act_set.wait_all();
        }
        me->extension<internalStream>()->complete();
    }
}

internalStream::internalStream(s4u::ActorPtr stream_actor, s4u::ActorPtr cuda_actor) {
    if (not simgrid::cuda::internalStream::EXTENSION_ID.valid())
        simgrid::cuda::internalStream::EXTENSION_ID =
            simgrid::s4u::Actor::extension_create<simgrid::cuda::internalStream>();
    stream_mb = s4u::Mailbox::by_name(stream_actor->get_name());
    stream_mb->set_receiver(stream_actor);
    cpu_mb = s4u::Mailbox::by_name(stream_actor->get_name() + "c");
    cpu_mb->set_receiver(cuda_actor);
}

void internalStream::wait() {
    while (kernel_count > 0) {
        cpu_mb->get_init()->wait();
        kernel_count--;
    }
}

simgrid::s4u::CommPtr internalStream::push(GpuActivityPtr new_activity) {
    return push({new_activity});
}

simgrid::s4u::CommPtr internalStream::push(std::vector<GpuActivityPtr> new_activities) {
    kernel_calls.push(std::vector<GpuActivityPtr>(new_activities));
    kernel_count++;
    return stream_mb->put_init(dummypayload, new_activities.size() * 64);
}

std::vector<GpuActivityPtr> internalStream::pop() {
    stream_mb->get_init()->wait();
    auto res = kernel_calls.front();
    kernel_calls.pop();
    return res;
}

void internalStream::complete() { cpu_mb->put_init(dummypayload, 64)->detach(); }

void Graph::add_to_graph(GpuActivityPtr activity) { 
    captured_activities.push_back({activity}); }

Graph::Graph() { captured_activities = std::vector<std::vector<simgrid::cuda::GpuActivityPtr>>(); }

void Graph::clear() { captured_activities.clear(); }

void Graph::add_to_graph(std::vector<GpuActivityPtr> activities) {
    captured_activities.push_back(activities);
}

std::vector<std::vector<GpuActivityPtr>> Graph::get_captured_activities() { return captured_activities; }

void Graph::destroy() { captured_activities.clear(); }

GraphExec::GraphExec() {}

GraphExec::GraphExec(std::vector<std::vector<GpuActivityPtr>> captured_activities_) {
    captured_activities = captured_activities_;
}


void GraphExec::launch(Stream stream) { 
    for (const auto& activity_vec : captured_activities) {
        std::vector<GpuActivityPtr> copied_activities;
        copied_activities.reserve(activity_vec.size());
        for (const auto& activity_ptr : activity_vec) {
            if (activity_ptr) {// no nullptr
                // copy activitites so the original don't get destroyed
                copied_activities.push_back(activity_ptr->copy());
            }
        }
        stream.push(copied_activities);
    }

}

GraphExec::~GraphExec() {
    for(size_t i = 0 ; i < captured_activities.size();++i){
        for(size_t j = 0; j<captured_activities[i].size();++j){
            delete captured_activities[i][j];
        }
    }
}

Stream::Stream() {
    gpu = cuda_process()->getCurrentDevice();
    streamActor =
        s4u::Actor::init(gpu->get_name() + ":" + std::to_string(gpu->get_actor_count()), gpu);
    streamActor->extension_set<internalStream>(new internalStream(streamActor, s4u::Actor::self()));
    // streamActor->daemonize();//to make cleanup easy
    streamActor->start(gpuActor);
}

// for mpi-gpu actors
Stream::Stream(s4u::ActorPtr streamActor) {
    gpu = cuda_process()->getCurrentDevice();
    streamActor->extension_set<internalStream>(new internalStream(streamActor, s4u::Actor::self()));
    streamActor->start(gpuActor);
}

void Stream::wait() { return streamActor->extension<internalStream>()->wait(); }

void Stream::push(GpuActivityPtr new_activity) {
    auto comm = streamActor->extension<internalStream>()->push(new_activity);
    comm->detach();
}

void Stream::push(std::vector<GpuActivityPtr> new_activities) {
    auto comm = streamActor->extension<internalStream>()->push(new_activities);
    comm->wait();
}

std::vector<GpuActivityPtr> Stream::pop() { return streamActor->extension<internalStream>()->pop(); }

} // namespace cuda
} // namespace simgrid