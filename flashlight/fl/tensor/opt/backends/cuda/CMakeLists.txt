# ----------------------------- Sources -----------------------------

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/CublasLtBackwardEpilogueFusion.cu
  ${CMAKE_CURRENT_LIST_DIR}/CudaFusedKernelLowering.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CudaJitOptimizerExtension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CudnnSubgraphLowering.cpp
  ${CMAKE_CURRENT_LIST_DIR}/NvrtcKernelCompiler.cpp
)

# Link NVRTC and CUDA driver API
find_package(CUDAToolkit REQUIRED)
target_link_libraries(flashlight PUBLIC CUDA::nvrtc CUDA::cuda_driver)
