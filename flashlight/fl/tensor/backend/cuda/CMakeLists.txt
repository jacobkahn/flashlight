# ----------------------------- Native CUDA Backend -----------------------------

find_package(CUDAToolkit REQUIRED)

target_link_libraries(
  flashlight
  PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  CUDA::curand
)

set(_FL_CUDA_BACKEND_CU_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/CudaTensor.cu
  ${CMAKE_CURRENT_LIST_DIR}/CudaBackend.cu
)

target_sources(flashlight PRIVATE
  ${_FL_CUDA_BACKEND_CU_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/CudaBackendTransfer.cpp
)

# __device__ lambdas require --extended-lambda (NVCC .cu files only)
set_source_files_properties(
  ${_FL_CUDA_BACKEND_CU_SOURCES}
  PROPERTIES COMPILE_OPTIONS "--extended-lambda"
)
