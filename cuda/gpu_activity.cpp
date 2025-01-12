#include "gpu_activity.h"
#include "tracer.h"

namespace simgrid {
static int dummy_int = 1;
static void *dummybuf = &dummy_int;
cuda::GpuActivity::GpuActivity(double flops) : type{EXEC}, load{flops}, mb{nullptr} {}

cuda::GpuActivity::GpuActivity(s4u::Mailbox *mailbox, double bytes, TYPE send_or_recv)
    : type{send_or_recv}, load{bytes}, mb{mailbox} {
    // assert(read_or_write==SEND || read_or_write==RECV);
}

cuda::GpuActivity::GpuActivity(double bytes, TYPE read_or_write)
    : type{read_or_write}, load{bytes}, mb{nullptr} {
    // assert(read_or_write==READ || read_or_write==WRITE);
}

namespace{
    std::string print_type(cuda::GpuActivity::TYPE type){
        switch (type) {
        case cuda::GpuActivity::EXEC:
            return "execution";
        case cuda::GpuActivity::SEND_ASYNC: {
            return "async-send";
        }
        case cuda::GpuActivity::SEND: {
            return "send";
        }
        case cuda::GpuActivity::RECV: {
           return "recv";
        }
        case cuda::GpuActivity::READ: {
            return "read";
        }
        case cuda::GpuActivity::WRITE: {
            return "write";
        }
        default:
            return "default";
        }
    }
}

void simgrid::cuda::GpuActivity::wait() {
    if(trace::tracingOn){
        auto beginning = s4u::Engine::get_clock();
        switch (type) {
        case EXEC:
            s4u::this_actor::execute(load);
            break;
        case SEND_ASYNC: {
            mb->put_init(dummybuf, load)->detach();
            break;
        }
        case SEND: {
            mb->put(dummybuf, load);
            break;
        }
        case RECV: {
            mb->get_async()->wait();
            break;
        }
        case READ: {
            auto disk = s4u::this_actor::get_host()->get_disks()[0];
            disk->read(load);
            break;
        }
        case WRITE: {
            auto disk = s4u::this_actor::get_host()->get_disks()[0];
            disk->write(load);
            break;
        }
        default:
            break;
        }
        trace::tracer.print(print_type(type) + ", " +
            std::to_string(beginning) + ", " +
            std::to_string(s4u::Engine::get_clock()) + ", " + 
            s4u::this_actor::get_host()->get_name()+", " +
            std::to_string(load) + ", " +
            mb->get_name() + "\n" );
    }
    else{
        switch (type) {
        case EXEC:
            s4u::this_actor::execute(load);
            return;
        case SEND_ASYNC: {
            mb->put_init(dummybuf, load)->detach();
            return;
        }
        case SEND: {
            mb->put(dummybuf, load);
            return;
        }
        case RECV: {
            mb->get_async()->wait();
            return;
        }
        case READ: {
            auto disk = s4u::this_actor::get_host()->get_disks()[0];
            disk->read(load);
            return;
        }
        case WRITE: {
            auto disk = s4u::this_actor::get_host()->get_disks()[0];
            disk->write(load);
            return;
        }
        default:
            return;
        }
    }
}


} // namespace simgrid
