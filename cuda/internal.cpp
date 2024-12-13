#include "internal.h"
#include <stddef.h>
#include <simgrid/s4u.hpp>
#include <simgrid/s4u/Activity.hpp>
#include <iostream>

namespace simgrid{
    namespace cuda{
        static void *dummypayload = malloc(2048);
        namespace{
            s4u::Host *getHost(cudaActor *a, HostType type){
                if(type==HostType::HOST){
                    return a->actor_->get_host();
                }
                else{
                    return a->getCurrentDevice();
                }
            }
        }

        simgrid::xbt::Extension<simgrid::s4u::Actor, cudaActor> cudaActor::EXTENSION_ID = simgrid::s4u::Actor::extension_create<simgrid::cuda::cudaActor>();
        simgrid::xbt::Extension<simgrid::s4u::Actor, internalStream> internalStream::EXTENSION_ID = simgrid::s4u::Actor::extension_create<simgrid::cuda::internalStream>();

        cudaActor* cuda_process(){
            simgrid::s4u::ActorPtr me = simgrid::s4u::Actor::self();

            if (me == nullptr) // This happens sometimes (eg, when linking against NS3 because it pulls openMPI...)
                return nullptr;

            return me->extension<cudaActor>();
        }

        namespace{
            bool char_equal(const char a[], const char b[], int n){
                for(int i=0;i<n;++i){
                    if(a[i]!=b[i])
                        return false;
                }
                return true;
            }

            std::vector<s4u::Host*> filter_gpus(std::vector<s4u::Host*> all_hosts){
                std::vector<s4u::Host*> res;
                for(int i=0;i<all_hosts.size();++i){
                    if(char_equal(all_hosts[i]->get_property("type"),"gpu", 3))
                        res.push_back(all_hosts[i]);
                }   
                return res;
            }

        }

        cudaActor::cudaActor(s4u::Actor* cpu){
            if (not simgrid::cuda::cudaActor::EXTENSION_ID.valid())
                simgrid::cuda::cudaActor::EXTENSION_ID = simgrid::s4u::Actor::extension_create<simgrid::cuda::cudaActor>(); 
            devices = filter_gpus(cpu->get_host()->get_englobing_zone()->get_all_hosts());
        }

        void cudaActor::send(HostType src, HostType dst, size_t count){
            s4u::Comm::sendto(getHost(this, src), getHost(this, dst), count);
        }

        void cudaActor::send_async(HostType src, HostType dst, size_t count, Stream stream) {
            s4u::Comm::sendto_async(getHost(this, src), getHost(this, dst), count);
        }

        void cudaActor::write(size_t count)
        {
            // todo add disk or use loop back
        }

        void cudaActor::setDevice(int device_index){
            current_device_index = device_index;
        }

        s4u::Host* cudaActor::getCurrentDevice()
        {
          return devices[current_device_index];
        }

        std::vector<s4u::Host *> cudaActor::getAllDevice()
        {
          return devices;
        }

        void gpuActor(){
            auto father = s4u::Actor::by_pid(s4u::this_actor::get_ppid());
            simgrid::s4u::ActorPtr me = simgrid::s4u::Actor::self();
            while(true){
                if(me->extension<internalStream>()->stream_mb->empty()){
                    if(father->is_suspended()){ // for stream synchronise
                        father->resume();
                    }
                    s4u::this_actor::suspend();
                }
                auto activities = me->extension<internalStream>()->pop();
                if(activities.size()>0){
                    for(int i=0;i<activities.size();++i){
                        activities[i].wait();
                    }
                }
            }
        }


        internalStream::internalStream(s4u::ActorPtr a)
        {
            if (not simgrid::cuda::internalStream::EXTENSION_ID.valid())
                simgrid::cuda::internalStream::EXTENSION_ID = simgrid::s4u::Actor::extension_create<simgrid::cuda::internalStream>(); 
            stream_mb = s4u::Mailbox::by_name(a->get_name());
            stream_mb->set_receiver(a);
        }

        simgrid::s4u::CommPtr internalStream::push(GpuActivity new_activity){
            return push(std::vector<GpuActivity>{new_activity});
        }

        simgrid::s4u::CommPtr internalStream::push(std::vector<GpuActivity> new_activities){
            // Kernel calls are put into a single communication
            // communication for the kernel launch
            kernel_calls.push(std::vector<GpuActivity>(new_activities));
            // stream is ready to compute
            return stream_mb->put_init(dummypayload, new_activities.size()*64);
        }

        std::vector<GpuActivity> internalStream::pop(){
            stream_mb->get_init()->wait();
            auto res = kernel_calls.front();
            kernel_calls.pop(); 
            return res;
        }
        
        void Graph::add_to_graph(GpuActivity activity)
        {
          captured_activities.push_back(activity);
        }

        Graph::Graph()
        {
            captured_activities = std::vector<simgrid::cuda::GpuActivity>();
        }

        void Graph::clear()
        {
            captured_activities.clear();
        }

        void Graph::add_to_graph(std::vector<GpuActivity> activities) {
            captured_activities.insert(captured_activities.end(),activities.begin(), activities.end());
        }

        std::vector<GpuActivity> Graph::get_captured_activities()
        {
          return captured_activities;
        }

        void Graph::destroy() {
            captured_activities.clear();
        }

        GraphExec::GraphExec()
        {
        }

        GraphExec::GraphExec(std::vector<GpuActivity> captured_activities_)
        {
            captured_activities = captured_activities_;
        }

        void GraphExec::launch(Stream stream)
        {
          stream.push(captured_activities);
        }

        Stream::Stream()
        {
            gpu = cuda_process()->getCurrentDevice();
            streamActor = s4u::Actor::init(gpu->get_name()+":"+std::to_string(gpu->get_actor_count()), gpu);
            streamActor->extension_set<internalStream>(new internalStream(streamActor));
            //streamActor->daemonize();//to make cleanup easy
            streamActor->start(gpuActor);
        }

        bool Stream::isEmpty()
        {
            return streamActor->is_suspended();
        }


        void Stream::push(GpuActivity new_activity)
        {
            auto comm = streamActor->extension<internalStream>()->push(new_activity);
            if(streamActor->is_suspended()) streamActor->resume();
            comm->detach();

        }

        void Stream::push(std::vector<GpuActivity> new_activities)
        {
            auto comm = streamActor->extension<internalStream>()->push(new_activities);
            if(streamActor->is_suspended()) streamActor->resume();
            comm->wait();
        }

        std::vector<GpuActivity> Stream::pop()
        {
            return streamActor->extension<internalStream>()->pop();
        }

    } // namespace cuda
}