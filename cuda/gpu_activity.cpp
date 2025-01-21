#include "gpu_activity.h"
#include "tracer.h"

namespace simgrid {
static int dummy_int = 1;
static void *dummybuf = &dummy_int;
cuda::GpuActivity::GpuActivity(double flops) : type{EXEC}, load{flops}, mb{nullptr} {}

cuda::GpuActivity::GpuActivity(s4u::Mailbox *mailbox, double bytes, TYPE send_or_recv, void *buf)
    : type{send_or_recv}, load{bytes}, mb{mailbox}, payload{buf == nullptr ? dummybuf : buf} {
    // assert(read_or_write==SEND || read_or_write==RECV);
}

cuda::GpuActivity::GpuActivity(double bytes, TYPE read_or_write)
    : type{read_or_write}, load{bytes}, mb{nullptr} {
    // assert(read_or_write==READ || read_or_write==WRITE);
}

cuda::GpuActivity cuda::GpuActivity::exec(double flops) { return GpuActivity(flops); }

cuda::GpuActivity cuda::GpuActivity::comm(s4u::Mailbox *mailbox, double bytes, TYPE send_or_recv,
                                          void *buf) {
    return GpuActivity(mailbox, bytes, send_or_recv, buf);
}

cuda::GpuActivity cuda::GpuActivity::io(double bytes, TYPE read_or_write) {
    return GpuActivity(bytes, read_or_write);
}

namespace {
std::string print_type(cuda::GpuActivity::TYPE type) {
    switch (type) {
    case cuda::GpuActivity::EXEC:
        return "execution";
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
} // namespace

s4u::ActivityPtr simgrid::cuda::GpuActivity::start() {
        auto beginning = s4u::Engine::get_clock();
        switch (type) {
        case EXEC:
            return s4u::this_actor::exec_async(load);
        case SEND: {
            return mb->put_init(payload, load);
        }
        case RECV: {
            return mb->get_async();
        }
        case READ: {
            auto disk = s4u::this_actor::get_host()->get_disks()[0];
            return disk->read_async(load);
        }
        case WRITE: {
            auto disk = s4u::this_actor::get_host()->get_disks()[0];
            return disk->write_async(load);
        }
        default:
            break;
        }
    return s4u::this_actor::exec_async(1);
}

} // namespace simgrid
