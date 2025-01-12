#ifndef TRACER_H
#define TRACER_H
#include <fstream>
#include "simgrid/s4u.hpp"

namespace trace{
class Tracer{
    private:
    std::ofstream trace_file;
    
    public:
        Tracer(const std::string filename);
        void print(std::string to_print);
};

static Tracer tracer = Tracer("nccl.trace");
static bool tracingOn = true;

}


#endif