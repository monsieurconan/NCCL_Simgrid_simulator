#include "gpu_activity.h"

namespace simgrid{
    static int dummy_int = 1;
    static void * dummybuf = &dummy_int;
    cuda::GpuActivity::GpuActivity(double flops): type{EXEC}, load{flops}, mb{nullptr}{}

    cuda::GpuActivity::GpuActivity(s4u::Mailbox *mailbox, double bytes, TYPE send_or_recv): type{send_or_recv}, load{bytes}, mb{mailbox}
    {
        // assert(read_or_write==SEND || read_or_write==RECV);
    }

    cuda::GpuActivity::GpuActivity(double bytes, TYPE read_or_write): type{read_or_write},load{bytes}, mb{nullptr}
    {
        // assert(read_or_write==READ || read_or_write==WRITE);
    }
    
    
    void simgrid::cuda::GpuActivity::wait()
    {
        switch (type)
        {
        case EXEC:
            s4u::this_actor::execute(load);
            return;
        case SEND_ASYNC:
        {
            mb->put_init(dummybuf, load)->detach();
            return;
        }
        case SEND:
        {
            mb->put(dummybuf, load);
            return;
        }
        case RECV:
        {
            mb->get_async()->wait();
            return;
        }
        case READ:
        {
            auto disk = s4u::this_actor::get_host()->get_disks()[0];
            disk->read(load);
            return;
        }
        case WRITE:
        {
            auto disk = s4u::this_actor::get_host()->get_disks()[0];
            disk->write(load);
            return;
        }
        default:
            return;
        }
    }
}


