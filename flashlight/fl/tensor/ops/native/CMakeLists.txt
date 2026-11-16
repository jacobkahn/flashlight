# ----------------------------- Native CPU Ops -----------------------------
# Backend-agnostic CPU implementations of tensor operations using CBLAS.
# Used as fallbacks when backend primitives don't support certain types
# (e.g., f64 matmul for OneDNN) or when no accelerated kernel exists.

# CBLAS detection: prefer MKL (already found by OneDNN backend if available),
# fall back to system CBLAS (OpenBLAS, ATLAS, Apple Accelerate, etc.)
if (NOT MKL_FOUND)
  find_package(CBLAS REQUIRED)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindCBLAS.cmake)
  set(NATIVE_OPS_CBLAS_INCLUDE_DIR ${CBLAS_INCLUDE_DIR})
  set(NATIVE_OPS_CBLAS_LIBRARIES ${CBLAS_LIBRARIES})
else()
  set(NATIVE_OPS_CBLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  set(NATIVE_OPS_CBLAS_LIBRARIES ${MKL_LIBRARIES})
endif()

target_compile_definitions(
  flashlight
  PRIVATE
  FL_USE_MKL=$<BOOL:${MKL_FOUND}>
)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Binary.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CpuTensorHelpers.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Creation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Matmul.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Reduction.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RNN.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ShapeAndIndex.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Sort.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Transpose.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Unary.cpp
)

target_include_directories(
  flashlight
  PRIVATE
  ${NATIVE_OPS_CBLAS_INCLUDE_DIR}
)

target_link_libraries(
  flashlight
  PRIVATE
  ${NATIVE_OPS_CBLAS_LIBRARIES}
)
