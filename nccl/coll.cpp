#include "coll.h"
#include "nccl_internal.h"

namespace {
    void plus(void *leftbuf, void *rightbuf, size_t n, ncclDataType_t datatype){
        switch (datatype)
        {
        case ncclInt:
            {int *left = (int*)leftbuf;
            int *right = (int*)rightbuf;
            for(size_t i=0;i<n;++i){
                left[i] = left[i] + right[i];
            }
            return;}
        
        case ncclFloat:
            {float *left = (float*)leftbuf;
            float *right = (float*)rightbuf;
            for(size_t i=0;i<n;++i){
                left[i] = left[i] + right[i];
            }
            return;}
        default:
            return;
        }
    }

    void reduction(ncclRedOp_t op, void *recvbuf, void *tmpbuf, size_t n, ncclDataType_t datatype, cudaStream_t stream){
        auto execution = simgrid::cuda::GpuActivity(n);// todo : better model
        stream->launch(execution);
        /*switch (op)
        {
        case ncclSum:    
            stream->launch(execution);
            plus(recvbuf, tmpbuf, n, datatype);
            break;
        case ncclProd:
            break;
        case ncclMax:
            break;
        case ncclMin:
            break;
        case ncclAvg:
            break;
        default:
            break;
        }*/
    }

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

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream){
        if(comm->rank(stream)==root){
            ncclGroupStart();
            for(int i=0;i<comm->nranks()-1;++i){
                ncclRecv(nullptr/*recvbuff*/, count, datatype,
                i<root ? i : i+i, comm, stream);
                reduction(op, nullptr, nullptr, count, datatype, stream);
            }
            ncclGroupEnd();
        }
        else{
            ncclSend(sendbuff, count, datatype, root, comm, stream);
        }
    return ncclSuccess;
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    if(count < comm->nranks()){
        // small amount of elements, all to all then reduce
        int size = ncclTypeSize(datatype);
        void *tmpbuf = malloc(comm->nranks()*count*size);
        int peer;
        ncclGroupStart();
        for(int i=0;i<comm->nranks()-1;++i){
            peer = comm->next_i_rank(stream, i);
            ncclSend(sendbuff, count, datatype, peer, comm, stream);
            ncclRecv(nullptr, count, datatype, peer, comm, stream);
        }
        ncclGroupEnd();
        // reduce
        //memcpy(recvbuff, sendbuff, count*size);
        for(int i=0;i<comm->nranks()-1;++i){
            reduction(op, recvbuff, nullptr, count, datatype, stream);
        }
        return ncclSuccess;
    }
    // larger amount of elements, more complex algo https://www.linkedin.com/pulse/nccl-allreduce-algorithm-ring-%E5%BD%A6%E6%B0%91-%E8%B4%BE
    int previous = comm->next_i_rank(stream, -1);
    int next = comm->next_i_rank(stream, 1);
    int self = comm->rank(stream);
    void *tmpbuf;
    int size = count*ncclTypeSize(datatype);
    int step_size = count / comm->nranks();
    int step = comm->nranks()-1-self;
    int step_plus_1 = (step+1)%comm->nranks();
    /*if(cudaMalloc(&tmpbuf, size)!= cudaSuccess) 
        return ncclUnhandledCudaError;
    memcpy(tmpbuf, recvbuff, count*size);*/
    ncclGroupStart();
    // recvreducesend part
    for(int i=0; i< comm->nranks();i++){
        /* ncclSend(tmpbuf+step*step_size, step==comm->nranks()?step_size:step_size+count%comm->nranks(), datatype, next, comm, stream);
        ncclRecv(recvbuff+(step_plus_1)*step_size, step_plus_1==comm->nranks()?step_size:step_size+count%comm->nranks(), datatype, previous, comm, stream);
        reduction(op, tmpbuf+step_plus_1, recvbuff+step_plus_1, step_plus_1==comm->nranks()?step_size:step_size+count%comm->nranks(), datatype, stream); */
        ncclSend(nullptr, step_size, datatype, next, comm, stream);
        ncclRecv(nullptr, step_size, datatype, previous, comm, stream);
        reduction(op, nullptr, nullptr, step_size, datatype, stream);
        /*step = step_plus_1;
        step_plus_1 = (step_plus_1+1)%comm->nranks();*/
    }
    // recvcopysend part
    for(int i=0; i< comm->nranks()-1;i++){
        /* ncclSend(tmpbuf+step*step_size, step==comm->nranks()?step_size:step_size+count%comm->nranks(), datatype, next, comm, stream);
        ncclRecv(recvbuff+(step_plus_1)*step_size, step_plus_1==comm->nranks()?step_size:step_size+count%comm->nranks(), datatype, previous, comm, stream); */
        ncclSend(nullptr, step_size, datatype, next, comm, stream);
        ncclRecv(nullptr, step_size, datatype, previous, comm, stream);
        /*step = step_plus_1;
        step_plus_1 = (step_plus_1+1)%comm->nranks();*/
    }
    ncclGroupEnd();
    return ncclSuccess;
}

ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream){
    if(comm->rank(stream)==root){
        ncclSend(buff, count, datatype, comm->next_i_rank(stream, 1), comm, stream);
    }
    else if(comm->next_i_rank(stream, -1)==root){
        ncclRecv(buff, count, datatype, comm->next_i_rank(stream, -1), comm, stream);
    }
    else{
        ncclGroupStart();
        ncclRecv(buff, count, datatype, comm->next_i_rank(stream, -1), comm, stream);
        ncclSend(buff, count, datatype, comm->next_i_rank(stream, 1), comm, stream);
        ncclGroupEnd();
    }
    return ncclSuccess;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream){
    ncclGroupStart();
    int previous = comm->next_i_rank(stream, -1);
    int next = comm->next_i_rank(stream, 1);
    int self = comm->rank(stream);
    int size = recvcount*ncclTypeSize(datatype);
    ncclGroupStart();
    // recvreducesend part
    for(int i=0; i< comm->nranks();i++){
        ncclSend(nullptr, recvcount, datatype, next, comm, stream);
        ncclRecv(nullptr, recvcount, datatype, previous, comm, stream);
        reduction(op, nullptr, nullptr, recvcount, datatype, stream);
    }
    ncclGroupEnd();
    return ncclSuccess;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream){
    if(sendcount < comm->nranks()){
        // small amount of elements, all to all then reduce
        int size = ncclTypeSize(datatype);
        void *tmpbuf = malloc(comm->nranks()*sendcount*size);
        int peer;
        ncclGroupStart();
        for(int i=0;i<comm->nranks()-1;++i){
            peer = comm->next_i_rank(stream, i);
            ncclSend(sendbuff, sendcount, datatype, peer, comm, stream);
            ncclRecv(nullptr/*recvbuff[sendcount*i*sizeof(datatype)]*/, sendcount, datatype, peer, comm, stream);
        }
        ncclGroupEnd();
        return ncclSuccess;
    }
    int previous = comm->next_i_rank(stream, -1);
    int next = comm->next_i_rank(stream, 1);
    int self = comm->rank(stream);
    void *tmpbuf;
    int size = sendcount*ncclTypeSize(datatype);
    ncclGroupStart();
    for(int i=0; i< comm->nranks()-1;i++){
        ncclSend(nullptr, size, datatype, next, comm, stream);
        ncclRecv(nullptr, size, datatype, previous, comm, stream);
    }
    ncclGroupEnd();
    return ncclSuccess;
}

ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    ncclGroupStart();
    if (comm->rank(stream) == root) {
      for (int r=0; r<comm->nranks(); r++)
        ncclRecv(nullptr, sendcount, datatype, r, comm, stream);
    }
    ncclSend(sendbuff, sendcount, datatype, root, comm, stream);
    ncclGroupEnd();
    return ncclSuccess;
}

ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    ncclGroupStart();
    if (comm->rank(stream) == root) {
      for (int r=0; r<comm->nranks(); r++)
        ncclRecv(nullptr, sendcount, datatype, r, comm, stream);
    }
    ncclSend(sendbuff, sendcount, datatype, root, comm, stream);
    ncclGroupEnd();
    return ncclSuccess;
}

ncclResult_t ncclAllToAll(const void *sendbuff, void *recvbuff, size_t sendcount, size_t recvcount,  ncclDataType_t sendtype, ncclDataType_t recvtype, ncclComm_t comm, cudaStream_t stream)
{
    ncclGroupStart();
    for (int r=0; r<comm->nranks(); r++) {
      ncclSend(nullptr, sendcount, sendtype, r, comm, stream);
      ncclRecv(nullptr, recvcount, recvtype, r, comm, stream);
    }
    ncclGroupEnd();
    return ncclSuccess;
}

ncclResult_t debugTest(cudaStream_t stream)
{
    auto execution = simgrid::cuda::GpuActivity(1e8);// todo : better model
    stream->launch(execution);
    return ncclSuccess;
}
