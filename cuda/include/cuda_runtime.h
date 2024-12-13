#ifndef CUDART
#define CUDART

#include <stddef.h>
#include "cuda_runtime.h"
#include "internal.h"
#include "gpu_activity.h"
#include "cuda_error.h"

enum cudaStreamCaptureMode{cudaStreamCaptureModeGlobal = 0,
cudaStreamCaptureModeThreadLocal = 1,
cudaStreamCaptureModeRelaxed = 2};

enum cudaStreamCaptureStatus{cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated};

typedef struct cudaStream* cudaStream_t;

typedef struct simgrid::cuda::Graph cudaGraph_t;

typedef struct simgrid::cuda::GraphExec cudaGraphExec_t;

typedef enum cudaError cudaError_t;

enum cudaMemcpyKind{cudaMemcpyHostToDevice, cudaMemcpyHostToHost, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault};

struct cudaStream{
    simgrid::cuda::Stream stream;
	cudaStreamCaptureMode mode;
	cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
	cudaGraph_t graph;
	public:
		cudaStream();
		cudaStream(int flags);
		void launch(simgrid::cuda::GpuActivity new_activity);
        void launch(std::vector<simgrid::cuda::GpuActivity> new_activities);
		
};

struct cudaGraphNode_t;
typedef enum cudaError cudaError_t;


cudaError_t  cudaMalloc(void **devPtr, size_t size);
cudaError_t  cudaMallocHost(void **ptr, size_t size);

cudaError_t  cudaFree(void *devPtr);
cudaError_t  cudaFreeHost(void *ptr);

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, 
			enum cudaMemcpyKind kind);
cudaError_t  cudaMemcpyAsync(void *dst, const void *src, 
			size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t  cudaMemset(void *devPtr, int value, size_t count);


cudaError_t cudaDeviceSynchronize(void);

cudaError_t cudaStreamSynchronise(cudaStream_t stream);

cudaError_t cudaSetDevice(int device);
    
cudaError_t cudaDeviceReset(void);


cudaError_t cudaGetDeviceProperties(
			struct cudaDeviceProp *prop, int device);

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int  flags);


cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);


#endif