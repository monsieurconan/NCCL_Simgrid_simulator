#ifndef GPU_ACTIVITY_H
#define GPU_ACTIVITY_H

#include "simgrid/s4u.hpp"

namespace simgrid {
namespace cuda {

struct GpuActivity {
    enum TYPE { EXEC, SEND, RECV, READ, WRITE, NONE };
    TYPE type;
    double load;
    s4u::Mailbox *mb;
    void *payload;

  private:
    GpuActivity(double flops);
    GpuActivity(s4u::Mailbox *mailbox, double bytes, TYPE send_or_recv, void *buf);
    GpuActivity(double bytes, TYPE read_or_write);

  public:
    static GpuActivity exec(double flops);
    static GpuActivity comm(s4u::Mailbox *mailbox, double bytes, TYPE send_or_recv,
                            void *buf = nullptr);
    static GpuActivity io(double bytes, TYPE read_or_write);
    s4u::ActivityPtr start();
};
} // namespace cuda
} // namespace simgrid

#endif