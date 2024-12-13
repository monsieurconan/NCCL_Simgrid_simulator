#ifndef __NCCL__
#define __NCCL__

// subsection of nccl api used for nccl-tests

#define NCCL_VERSION_CODE
#define NCCL_VERSION

#include "cuda_runtime.h"
#include <stddef.h>

typedef struct ncclComm *ncclComm_t;

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7
} ncclResult_t;

typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4 } ncclRedOp_t;

#define NCCL_UNIQUE_ID_BYTES 128
typedef int ncclUniqueId;

enum ncclDataType_t {
    ncclChar = 0,
    ncclUint8 = 1,
    ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclHalf = 6,
    ncclFloat = 7,
    ncclDouble = 8,
    ncclNumTypes = 9
};

/* ncclScalarResidence_t: Location and dereferencing logic for scalar arguments. */
enum ncclScalarResidence_t {
    /* ncclScalarDevice: The scalar is in device-visible memory and will be
     * dereferenced while the collective is running. */
    ncclScalarDevice = 0,

    /* ncclScalarHostImmediate: The scalar is in host-visible memory and will be
     * dereferenced before the ncclRedOpCreate***() function returns. */
    ncclScalarHostImmediate = 1
};

char const *ncclGetLastError(ncclComm_t comm);

char *ncclGetErrorString(ncclResult_t res);

// void NCCLCHECK();

/*template <typename T>
T ncclVerifiablePremulScalar(int rank);*/

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype,
                                      ncclScalarResidence_t residence, ncclComm_t comm);

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

ncclResult_t ncclGroupStart();
ncclResult_t ncclGroupEnd();

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle);
ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist);
ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank);
ncclResult_t ncclCommCount(const ncclComm_t comm, int *count);
ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclMemAlloc(void **ptr, size_t size);
ncclResult_t ncclMemFree(void *ptr);

/*ncclResult_t ncclstringtoop();
ncclResult_t ncclstringtotype();*/ //might not be inside

ncclResult_t ncclGetUniqueId(ncclUniqueId *out);

#endif