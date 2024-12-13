#include "cuda_runtime.h"
#include "internal.h"
#include "simgrid/s4u.hpp"
#include "vector"

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    stream->mode = cudaStreamCaptureModeThreadLocal;
    stream->graph.clear();
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) {
    if (stream->mode == cudaStreamCaptureModeRelaxed) return cudaError_t::cudaErrorInvalidValue;
    stream->mode = cudaStreamCaptureModeRelaxed;
    *pGraph = stream->graph;
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                 cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize) {
    *pGraphExec = simgrid::cuda::GraphExec{graph.get_captured_activities()};
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    graphExec.launch(stream->stream);
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    graph.destroy();
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    graphExec.captured_activities.clear();
    return cudaError_t::cudaSuccess;
}
