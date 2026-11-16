# ----------------------------- CUDA Ops -----------------------------
# CUDA kernel implementations of tensor operations: unary, binary,
# reductions, BLAS, shape/index, and autograd delegations.

set(_FL_CUDA_OPS_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/CudaUnaryOps.cu
  ${CMAKE_CURRENT_LIST_DIR}/CudaBinaryOps.cu
  ${CMAKE_CURRENT_LIST_DIR}/CudaReductions.cu
  ${CMAKE_CURRENT_LIST_DIR}/CudaBLAS.cu
  ${CMAKE_CURRENT_LIST_DIR}/CudaShapeAndIndex.cu
  ${CMAKE_CURRENT_LIST_DIR}/CudaAutogradOps.cu
)

target_sources(flashlight PRIVATE ${_FL_CUDA_OPS_SOURCES})

# __device__ lambdas require --extended-lambda
set_source_files_properties(
  ${_FL_CUDA_OPS_SOURCES}
  PROPERTIES COMPILE_OPTIONS "--extended-lambda"
)
