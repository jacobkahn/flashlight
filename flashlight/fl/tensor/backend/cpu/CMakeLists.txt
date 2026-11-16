# ----------------------------- OneDNN -----------------------------

# Require oneDNN >= 3.5. Version rationale:
# - 3.5+: matmul improvements, brgemm JIT GEMM refinements (used by TF/PyTorch)
# - 3.6+: brgemm matmul performance fixes on x86
# - 3.7+: improved ACL integration for AArch64, better ARM performance
# - 3.10+: latest stable with accumulated fixes
set(FL_ONEDNN_MIN_VERSION 3.5)

find_package(DNNL ${FL_ONEDNN_MIN_VERSION} CONFIG)
if (NOT DNNL_FOUND AND FL_USE_CPU)
  message(FATAL_ERROR
    "oneDNN >= ${FL_ONEDNN_MIN_VERSION} required to build CPU backend. "
    "oneDNN was not found or the installed version is too old. "
    "Please install oneDNN ${FL_ONEDNN_MIN_VERSION} or later.")
else()
  message(STATUS "oneDNN found")
  set(FL_USE_ONEDNN ON)
endif()

# On AArch64/ARM, set FL_ONEDNN_ARM for conditional code paths.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(FL_ONEDNN_ARM ON)
else()
  set(FL_ONEDNN_ARM OFF)
endif()

# TODO: conditionally set this to use OPENCL if the detected
# DNNL is built with OpenCL support
fl_set_backend_state(ENABLE CPU) # enables FL_USE_CPU

# ----------------------------- Sources -----------------------------
find_package(MKL)
if (NOT MKL_FOUND)
  message(STATUS "Using stdlib random number generation")
  set(${FL_USE_MKL_RNG} FALSE)
else()
  message(STATUS "MKL found - will use for random number generation")
  set(${FL_USE_MKL_RNG} TRUE)
endif()

target_compile_definitions(
  flashlight
  PRIVATE
  FL_USE_MKL_RNG=$<BOOL:${FL_USE_MKL_RNG}>
  FL_ONEDNN_ARM=$<BOOL:${FL_ONEDNN_ARM}>
)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/CPUBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CPUBinaryOps.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CPUCreation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CPUReductions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CPUShapeAndIndex.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CPUUnaryOps.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CPUTensor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CPUAutogradOps.cpp
)

target_include_directories(
  flashlight
  PUBLIC
  ${MKL_INCLUDE_DIR}
)

target_link_libraries(
  flashlight
  PUBLIC
  DNNL::dnnl
  ${MKL_LIBRARIES}
)
