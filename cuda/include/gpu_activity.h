#ifndef GPU_ACTIVITY_H
#define GPU_ACTIVITY_H

#include "simgrid/s4u.hpp"

namespace simgrid {
namespace cuda {

struct GpuActivity {
  public:
    virtual ~GpuActivity();
    virtual s4u::ActivityPtr start() = 0;
    virtual std::string print_type() = 0;
    virtual GpuActivity *copy() = 0;
};

struct GpuExec : GpuActivity{
    double flops;
    GpuExec(double flops);
    s4u::ActivityPtr start();
    std::string print_type();
    GpuActivity *copy();
};
enum COMM_TYPE {SEND, RECV};
struct GpuComm : GpuActivity{
    
    COMM_TYPE type;
    double load;
    s4u::Mailbox *mb;
    void *payload;

    GpuComm(s4u::Mailbox *mailbox, double bytes, COMM_TYPE send_or_recv,
                            void *buf = nullptr);
    
    s4u::ActivityPtr start();
    std::string print_type();
    GpuActivity *copy();
};
enum IO_TYPE {WRITE, READ};
struct GpuIO : GpuActivity {
    IO_TYPE type;
    double bytes;
    GpuIO(double bytes, IO_TYPE type);
    s4u::ActivityPtr start();
    std::string print_type();
    GpuActivity *copy();
};

template <typename... Args>
struct GpuFn : GpuActivity {
    double flops;
    std::function<int(Args...)> function;
    std::tuple<Args...> arguments;

    GpuFn(double flops, std::function<int(Args...)> fn, Args&&... args);
    s4u::ActivityPtr start();
    std::string print_type();
    GpuActivity *copy();
};



//using GpuActivityPtr  = std::unique_ptr<GpuActivity>;
typedef struct GpuActivity *GpuActivityPtr;

    static GpuActivityPtr CreateGpuExec(double flops) { 
    return new GpuExec(flops); 
}
    static GpuActivityPtr CreateGpuComm(s4u::Mailbox *mailbox, double bytes, COMM_TYPE send_or_recv,
                            void *buf = nullptr){
    return new GpuComm(mailbox, bytes, send_or_recv, buf); 
}
    static GpuActivityPtr CreateGpuIO(double bytes, IO_TYPE read_or_write){
    return new GpuIO(bytes,read_or_write); 
}
    template <typename... Args>
    static GpuActivityPtr CreateGpuFn(double flops, std::function<s4u::ActivityPtr(Args...)> fn,Args... args){
        return new GpuFn<Args&&...>(flops, fn, args...);
    }

} // namespace cuda
} // namespace simgrid

#endif