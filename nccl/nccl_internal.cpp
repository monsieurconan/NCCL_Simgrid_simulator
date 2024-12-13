#include "nccl_internal.h"
#include "cuda.h"
#include "nccl.h"
#include "iostream"
#include "cstring"

simgrid::xbt::Extension<simgrid::s4u::Host, ncclRank> ncclRank::EXTENSION_ID = simgrid::s4u::Host::extension_create<ncclRank>();
simgrid::xbt::Extension<simgrid::s4u::Actor, ncclActor> ncclActor::EXTENSION_ID = simgrid::s4u::Actor::extension_create<ncclActor>();

ncclRank::ncclRank(int _rank){
    if(not ncclRank::EXTENSION_ID.valid())
        ncclRank::EXTENSION_ID = simgrid::s4u::Host::extension_create<ncclRank>();
    rank = _rank;
}

ncclComm::ncclComm(int nranks) {
    ranks = nranks;
    pendingComm = std::vector<std::list<ncclMessage>>(ranks);
    ranks_to_gpus = std::vector<simgrid::s4u::Host *>(nranks);
    ranks_to_mailboxes = std::vector<simgrid::s4u::Mailbox *>(nranks, nullptr);
}

int ncclComm::nranks()
{
  return ranks;
}

void ncclComm::add_rank(int rank, simgrid::s4u::Host* gpu) {
    //assert(rank<nranks);
    ranks_to_gpus[rank] = gpu;
    //std::cout << gpu->get_name() << " is rank " << rank << "\n";
    gpu->extension_set<ncclRank>(new ncclRank(rank));
    ranks_to_mailboxes[rank] = simgrid::s4u::Mailbox::by_name("m"+gpu->get_name());
    return;
}

void ncclComm::add_devices(const int* devlist, std::vector<simgrid::s4u::Host*> gpus, int n) {
    for(int i=0;i<n;++i){
        add_rank(i, gpus[devlist[i]]);
    }
}


int ncclComm::rank(cudaStream_t stream)
{
  return stream->stream.gpu->extension<ncclRank>()->rank;
}

int ncclComm::next_i_rank(cudaStream_t stream, int i)
{
    auto gpu_name = stream->stream.gpu->get_name();
 auto rank = stream->stream.gpu->extension<ncclRank>();   
  return (rank->rank+i)%ranks;
}

namespace {
     int ncclTypeSize(ncclDataType_t type) {
        switch (type) {
        case ncclChar:
            return 1;
        case ncclHalf:
            return 2;
        case ncclUint32:
            return 4;
        case ncclInt:
            return 4;
        case ncclFloat:
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclDouble:
            return 8;
        default:
          return -1;
        }
    }
}


void ncclComm::recv(int dst, int src, void* rbuf, size_t count, ncclDataType_t datatype) {
    auto lm = pendingComm[dst];
    if(!lm.empty()){
        ncclMessage current;
        auto e = lm.begin();
        e++;
        while(e!=lm.end()){
            current = *e;
            if(current.src==src && current.count==count && current.datatype==datatype){
                std::memcpy(rbuf, current.sbuf, count*ncclTypeSize(datatype));
                lm.erase(e);
                break;
            }
        }
    }
    lm.push_back({nullptr, rbuf, src, count, datatype});
}

void ncclComm::send(int dst, int src, const void* sbuf, size_t count, ncclDataType_t datatype) {
    auto lm = pendingComm[dst];
    if(!lm.empty()){
        ncclMessage current;
        auto e = lm.begin();
        e++;
        while(e!=lm.end()){
            current = *e;
            if(current.src==src && current.count==count && current.datatype==datatype){
                std::memcpy(current.rbuf, sbuf, count*ncclTypeSize(datatype));
                lm.erase(e);
                break;
            }
        }
    }
    lm.push_back({sbuf, nullptr, src, count, datatype});
}

ncclActor::ncclActor(int user_rank)
{
    if (not ncclActor::EXTENSION_ID.valid())
                ncclActor::EXTENSION_ID = simgrid::s4u::Actor::extension_create<ncclActor>(); 
    group_level=0;
    u_rank = user_rank;
}

ncclActor *nccl_actor()
{
    simgrid::s4u::ActorPtr me = simgrid::s4u::Actor::self();

        if (me == nullptr) // This happens sometimes (eg, when linking against NS3 because it pulls openMPI...)
            return nullptr;

        return me->extension<ncclActor>();
}
