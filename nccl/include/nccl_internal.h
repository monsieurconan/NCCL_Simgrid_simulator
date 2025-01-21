#ifndef NCCL_INTERNAL_H
#define NCCL_INTERNAL_H

#include "nccl.h"
#include "simgrid/s4u.hpp"

typedef unsigned long size_t;

struct ncclRank {
    static simgrid::xbt::Extension<simgrid::s4u::Host, ncclRank> EXTENSION_ID;
    std::vector<int> rank;

  public:
    ncclRank();
    void setRank(int rank, int ncclCommId);
};

struct ncclComm {
    int unique_id;
    int ranks;
    bool blocking=true;
    std::map<void *, void *> zero_copy_buffers;
    // mapping to stream
    std::vector<simgrid::s4u::Host *> ranks_to_gpus;
    std::vector<simgrid::s4u::Mailbox *> ranks_to_mailboxes;

    ncclComm(int unique_id, int nranks);
    int nranks();
    void add_rank(int rank, simgrid::s4u::Host *gpu);
    void add_devices(const int *devlist, std::vector<simgrid::s4u::Host *> gpus, int n);
    int rank(cudaStream_t stream);
    int next_i_rank(cudaStream_t stream, int i);
};

struct ncclActor {
    static simgrid::xbt::Extension<simgrid::s4u::Actor, ncclActor> EXTENSION_ID;
  public:
    std::vector<cudaStream_t> streams_to_flush;
    int group_level = 0;
    int u_rank;
    ncclActor(int user_rank);
    void flush();
};

ncclActor *nccl_actor();

#endif