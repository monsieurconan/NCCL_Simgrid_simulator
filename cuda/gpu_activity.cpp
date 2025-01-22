#include "gpu_activity.h"
#include "tracer.h"

namespace simgrid {
static int dummy_int = 1;
static void *dummybuf = &dummy_int;

namespace cuda{

GpuExec::GpuExec(double _flops) : flops{_flops}{}

s4u::ActivityPtr GpuExec::start() { 
    return s4u::this_actor::exec_async(flops);}

std::string GpuExec::print_type() { 
    return "execution"; }

GpuActivityPtr GpuExec::copy() { 
    return CreateGpuExec(flops); 
}

GpuComm::GpuComm(s4u::Mailbox *mailbox, double bytes, COMM_TYPE send_or_recv, void *buf)
    : type{send_or_recv}, load{bytes}, mb{mailbox}, payload{buf == nullptr ? dummybuf : buf} {
}

s4u::ActivityPtr GpuComm::start() {
    if(type==SEND) 
            return mb->put_init(payload, load);
        
        else
            return mb->get_async();//todo &payload
        
    
 }

std::string GpuComm::print_type() { 
    if(type==COMM_TYPE::SEND) return "send";
    else return "recv"; }

GpuActivityPtr GpuComm::copy() { 
    return CreateGpuComm(mb, load, type, payload); 
}

GpuIO::GpuIO(double _bytes, IO_TYPE read_or_write)
    : type{read_or_write}, bytes{_bytes}{
}

s4u::ActivityPtr GpuIO::start() { 
    if(type==READ){
        auto disk = s4u::this_actor::get_host()->get_disks()[0];
        return disk->read_async(bytes);
    }
    else{
        auto disk = s4u::this_actor::get_host()->get_disks()[0];
        return disk->write_async(bytes);
    }
 }

std::string GpuIO::print_type() { if(type==IO_TYPE::READ) return "read";
else return "write"; }

GpuActivityPtr GpuIO::copy() { 
    return CreateGpuIO(bytes, type); 
}

template <typename... Args>
GpuFn<Args...>::GpuFn(double flops, std::function<int(Args...)> fn, Args &&...args) : 
function(std::move(fn)), arguments(std::forward<Args>(args)...) {}

template <typename... Args> s4u::ActivityPtr GpuFn<Args...>::start() {
    std::apply(function, arguments);
    return s4u::this_actor::exec_async(flops);
}

template <typename... Args> 
std::string GpuFn<Args...>::print_type() { 
    return "function"; 
}

template <typename... Args> 
GpuActivity *GpuFn<Args...>::copy() { 
    return CreateGpuFn<Args...>(flops, function, arguments); 
}

GpuActivity::~GpuActivity() {}


} // namespace cuda

} // namespace simgrid
