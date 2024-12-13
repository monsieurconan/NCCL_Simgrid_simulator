#ifndef GPU_ACTIVITY_H
#define GPU_ACTIVITY_H

#include "simgrid/s4u.hpp"
#include "boost/intrusive_ptr.hpp"

namespace simgrid{
    namespace cuda{

        struct GpuActivity{
            enum TYPE{EXEC, SEND, SEND_ASYNC, RECV, READ, WRITE, NONE};
            TYPE type;
            double load;
            s4u::Mailbox* mb;        
            public:
                GpuActivity(double flops);
                GpuActivity(s4u::Mailbox* mailbox, double bytes, TYPE send_or_recv);
                GpuActivity(double bytes, TYPE read_or_write);
                void wait();
        };
    }
}

#endif