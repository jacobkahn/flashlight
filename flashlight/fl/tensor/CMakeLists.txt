# Enable or disable compute backends. Example:
#
#   fl_set_backend_state(ENABLE CUDA DISABLE CPU)
#
# The above indicates a CUDA-compatible backend is being built and should be
# enabled and that CPU backend components should be disabled. Whether to
# build OPENCL backend components will remain unchanged.
function(fl_set_backend_state)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs ENABLE DISABLE)
  cmake_parse_arguments(fl_set_backend_state
    "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN} )

  # Enable
  foreach(backend IN ITEMS ${fl_set_backend_state_ENABLE})
    set(FL_USE_${backend} ON PARENT_SCOPE)
  endforeach()
  # Disable
  foreach(backend IN ITEMS ${fl_set_backend_state_DISABLE})
    set(FL_USE_${backend} OFF PARENT_SCOPE)
  endforeach()
endfunction()

# ]--------------- Backend Options and Selection
# The native CPU and CUDA backends are the defaults; ArrayFire is opt-in.
option(FL_USE_ARRAYFIRE "Build ArrayFire tensor backend" OFF)
option(FL_USE_TENSOR_STUB "Build Stub tensor backend" ON)
option(FL_USE_CPU "Build CPU tensor backend" ON)
cmake_dependent_option(FL_USE_NATIVE_CUDA "Build native CUDA tensor backend" ON
  "FL_USE_CUDA" OFF)


if (FL_USE_ARRAYFIRE)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/af/CMakeLists.txt)
endif()
if (FL_USE_TENSOR_STUB)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/stub/CMakeLists.txt)
endif()
if (FL_USE_CPU)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/cpu/CMakeLists.txt)
  # FL_USE_ONEDNN is set by backend/cpu/CMakeLists.txt when DNNL is found
  if (FL_USE_ONEDNN)
    include(${CMAKE_CURRENT_LIST_DIR}/ops/onednn/CMakeLists.txt)
  endif()
endif()
if (FL_USE_NATIVE_CUDA)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/cuda/CMakeLists.txt)
  include(${CMAKE_CURRENT_LIST_DIR}/ops/cuda/CMakeLists.txt)
endif()

# JIT infrastructure -- always built (the execution model for all tensor ops)
include(${CMAKE_CURRENT_LIST_DIR}/eval/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/ir/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/printer/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/opt/CMakeLists.txt)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/JitBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/JitBackendAutogradOps.cpp
  ${CMAKE_CURRENT_LIST_DIR}/JitBackendUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/JitTensorBase.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ShapeInference.cpp
)

# Auto-enable cuDNN when CUDA is enabled and cuDNN is found
if (FL_USE_CUDA)
  find_package(CUDNN)
  if (CUDNN_FOUND)
    set(FL_USE_CUDNN ON)
    message(STATUS
      "CUDNN found (library: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
    setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindCUDNN.cmake)
  endif()
else()
  set(FL_USE_CUDNN OFF)
endif()

if (FL_USE_CUDNN)
  include(${CMAKE_CURRENT_LIST_DIR}/ops/cudnn/CMakeLists.txt)

  # cudnn_frontend: header-only C++17 wrapper for the cuDNN Backend Graph API.
  # Provides the forward-looking graph-based API that replaces deprecated
  # legacy cuDNN calls (conv forward/backward, pooling, batchnorm, etc.).
  include(FetchContent)
  FetchContent_Declare(
    cudnn_frontend
    GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
    GIT_TAG        v1.9.0
    GIT_SHALLOW    TRUE
  )
  # cudnn_frontend is header-only; we only need the headers, not samples/tests.
  set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(cudnn_frontend)
  # cudnn_frontend is header-only. Use its include dirs directly rather than
  # linking the target, which would require exporting it in install(EXPORT).
  get_target_property(_cfe_includes cudnn_frontend INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(flashlight PUBLIC ${_cfe_includes})
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  FL_USE_ARRAYFIRE=$<BOOL:${FL_USE_ARRAYFIRE}>
  FL_USE_TENSOR_STUB=$<BOOL:${FL_USE_TENSOR_STUB}>
  FL_USE_CPU=$<BOOL:${FL_USE_CPU}>
  FL_USE_ONEDNN=$<BOOL:${FL_USE_ONEDNN}>
  FL_USE_NATIVE_CUDA=$<BOOL:${FL_USE_NATIVE_CUDA}>
  FL_USE_CUDNN=$<BOOL:${FL_USE_CUDNN}>
)

# Make sure at least one tensor backend is enabled
if (NOT (
    ${FL_USE_ARRAYFIRE} OR
    ${FL_USE_TENSOR_STUB} OR
    ${FL_USE_CPU} OR
    ${FL_USE_NATIVE_CUDA}
    )
  )
  message(FATAL_ERROR "Cannot build Flashlight with no tensor backends "
    "enabled. Flashlight must be built with  FL_USE_[backend name] ON "
    "for at least one backend.")
endif()

# Native CPU ops (CBLAS-based matmul, etc.)
include(${CMAKE_CURRENT_LIST_DIR}/ops/native/CMakeLists.txt)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Compute.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Copy.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DefaultTensorType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BackendDispatch.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BackendRegistry.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Index.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Init.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Random.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Shape.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Tensor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorAdapter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorExtension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Types.cpp
  ${CMAKE_CURRENT_LIST_DIR}/AutogradOps.cpp
)

# Profiling -- TODO: move this to runtime things
cmake_dependent_option(FL_BUILD_PROFILING
  "Enable profiling with Flashlight" OFF
  "FL_USE_CUDA" OFF)

if (FL_USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "CUDA is enabled but no CUDA compiler was found")
  endif()

  # Link CUDA components
  find_package(CUDAToolkit REQUIRED cublas)
  target_link_libraries(flashlight PRIVATE CUDA::cublas)

  if (FL_BUILD_PROFILING)
    # Try to find NVTX
    find_library(CUDA_NVTX_LIBRARIES REQUIRED
      NAMES nvToolsExt
      PATHS "${CUDA_TOOLKIT_ROOT_DIR}"
      ENV CUDA_PATH
      ENV CUDA_LIB_PATH
      ENV CUDA_HOME
      PATH_SUFFIXES lib64 lib
      NO_DEFAULT_PATH
      )

    target_sources(flashlight PRIVATE ${CMAKE_CURRENT_LIST_DIR}/CUDAProfile.cpp)
    target_link_libraries(flashlight PUBLIC ${CUDA_NVTX_LIBRARIES})
  endif()
endif()
