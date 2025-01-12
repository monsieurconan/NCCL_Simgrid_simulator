#include "tracer.h"

void trace::Tracer::print(std::string to_print) {
    trace_file << to_print;
}

trace::Tracer::Tracer(const std::string filename): trace_file{filename} {

}
