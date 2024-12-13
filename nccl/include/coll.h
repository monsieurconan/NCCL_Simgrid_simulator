#ifndef __NCCL_COLL__
#define __NCCL_COLL__

#include "cuda.h"
#include "nccl.h"
#include <stddef.h>

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);


ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t sendcount,
    size_t recvcount, ncclDataType_t sendtype, ncclDataType_t recvtype, ncclComm_t comm, 
    cudaStream_t stream);

ncclResult_t debugTest(cudaStream_t stream);
#endif