#ifndef NCCL_INTERNAL_H
#define NCCL_INTERNAL_H

#include "list"
#include "nccl.h"
#include "simgrid/s4u.hpp"

typedef unsigned long size_t;

struct ncclMessage{
    const void *sbuf;
    void * rbuf;
    int src;
    size_t count;
    ncclDataType_t datatype;
    // add a handle to the activities to check if the simulated message is finished ?
};

struct ncclRank{
    static simgrid::xbt::Extension<simgrid::s4u::Host, ncclRank> EXTENSION_ID;
    int rank;
    public:
        ncclRank(int rank);
};


struct ncclComm{
    int ranks;
    std::map<void*, void*> zero_copy_buffers;
    // mapping to stream
    std::vector<simgrid::s4u::Host*> ranks_to_gpus;
    std::vector<simgrid::s4u::Mailbox*> ranks_to_mailboxes;
    std::vector<std::list<ncclMessage>> pendingComm; 

    ncclComm(int nranks);
    int nranks();
    void add_rank(int rank, simgrid::s4u::Host* gpu);
    void add_devices(const int* devlist, std::vector<simgrid::s4u::Host *>gpus, int n);
    int rank(cudaStream_t stream);
    int next_i_rank(cudaStream_t stream, int i);
    void recv(int dst, int src, void* rbuf, size_t count, ncclDataType_t datatype);
    void send(int dst, int src, const void* sbuf, size_t count, ncclDataType_t datatype);
};

struct ncclActor{
    static simgrid::xbt::Extension<simgrid::s4u::Actor, ncclActor> EXTENSION_ID;
    public:
        int group_level=0;
        int u_rank;
        ncclActor(int user_rank);
};

ncclActor *nccl_actor();




#endif