#include "nccl.h"
#include "internal.h"
#include "nccl_internal.h"

static ncclUniqueId ids = -1;
static std::vector<ncclComm_t> communicators = std::vector<ncclComm_t>();

static ncclComm_t get_or_create_communicator(ncclUniqueId id, int nranks) {
    if (id >= communicators.size()) {
        communicators.push_back(new ncclComm(id, nranks));
    }
    return communicators[id];
}

ncclUniqueId newUniqueId() {
    ids++;
    return ids;
}

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
    return cudaMalloc(ptr, size) == cudaError::cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

ncclResult_t ncclMemFree(void *ptr) {
    return cudaFree(ptr) == cudaError::cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

ncclResult_t ncclGroupStart() {
    nccl_actor()->group_level++;
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
    nccl_actor()->group_level--;
    if(nccl_actor()->group_level < 0) return ncclInvalidUsage;
    if(nccl_actor()->group_level==0)nccl_actor()->flush();
    return ncclSuccess;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle) {
    comm->zero_copy_buffers.emplace(*handle, buff);
    return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
    comm->zero_copy_buffers.erase(handle);
    return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
    **comm = ncclComm(newUniqueId(), ndev);
    (*comm)->add_devices(devlist, simgrid::cuda::cuda_process()->getAllDevice(), ndev);
    communicators.push_back(*comm);
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
    // todo (reformulate in fn comm recv(uniqueId))
    *comm = get_or_create_communicator(commId, nranks);
    (*comm)->add_rank(rank, simgrid::cuda::cuda_process()->getCurrentDevice());
    simgrid::s4u::this_actor::sleep_for(1e-8);
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
    *rank = nccl_actor()->u_rank;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
    *count = comm->nranks(); // todo check if local rank is a thing
    return ncclSuccess;
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
    auto message = simgrid::cuda::CreateGpuComm(
        comm->ranks_to_mailboxes[peer], count * sizeof(datatype), simgrid::cuda::COMM_TYPE::SEND, (void *)sendbuff);
    if (nccl_actor()->group_level == 0) stream->launch(message);
    else{
        stream->add_to_buf(message);
        nccl_actor()->streams_to_flush.push_back(stream);
    } 
    if(nccl_actor()->group_level>0) 
        return ncclSuccess;
    else if(!comm->blocking)
        return ncclInProgress;
    else{
        stream->stream.wait();
        return ncclSuccess;
    }
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
    auto message = simgrid::cuda::CreateGpuComm(comm->ranks_to_mailboxes[comm->rank(stream)],
                                                    count * sizeof(datatype),
                                                    simgrid::cuda::COMM_TYPE::RECV, recvbuff);
    if (nccl_actor()->group_level == 0) stream->launch(message);
    else{
        stream->add_to_buf(message);
        nccl_actor()->streams_to_flush.push_back(stream);
    } 
    if(nccl_actor()->group_level>0) 
        return ncclSuccess;
    else if(!comm->blocking)
        return ncclInProgress;
    else{
        stream->stream.wait();
        return ncclSuccess;
    }
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *out) {
    *out = newUniqueId();
    return ncclSuccess;
}

char const *ncclGetLastError(ncclComm_t comm) {
    // todo (annoying)
    return nullptr;
}

char *ncclGetErrorString(ncclResult_t res) {
    switch (res) {
    case ncclSuccess:
        return std::string("success").data();
    case ncclUnhandledCudaError:
        return std::string("cuda error").data();
    case ncclSystemError:
        return std::string("system error").data();
    case ncclInternalError:
        return std::string("internal error").data();
    case ncclInvalidArgument:
        return std::string("invalid argument").data();
    case ncclInvalidUsage:
        return std::string("invalid usage").data();
    case ncclRemoteError:
        return std::string("remote error").data();
    case ncclInProgress:
        return std::string("in progress").data();
    default:
        return std::string("error").data();
    }
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype,
                                      ncclScalarResidence_t residence, ncclComm_t comm) {
    // todo (hard)
    return ncclSuccess;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
    // todo (easy)
    return ncclSuccess;
}
