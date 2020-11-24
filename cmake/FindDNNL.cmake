# - Try to find DNNL
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#  USE_DNNL
#
# The following are set after configuration is done:
#  DNNL_FOUND          : set to true if mkl-dnn is found.
#  DNNL_INCLUDE_DIR    : path to mkl-dnn include dir.
#  DNNL_LIBRARIES      : list of libraries for mkl-dnn

set(DNNL_LIBRARIES)
set(DNNL_INCLUDE_DIR)

# Check if there's any BLAS implementation available
find_package(BLAS)

if (NOT MKL_FOUND)
  message(STATUS "Attempting to build DNNL without MKL in current configuration.")
endif()

# Find headers
find_path(
  DNNL_INCLUDE_DIR
  mkldnn.hpp mkldnn.h
  PATHS
  ${DNNL_ROOT}
  PATH_SUFFIXES
  include
  HINTS
  ${DNNL_INC_DIR}
  )

if (DNNL_INCLUDE_DIR)
  message(STATUS "DNNL headers found in ${DNNL_INCLUDE_DIR}")
else()
  message(STATUS "DNNL headers not found; please set CMAKE_INCLUDE_PATH or DNNL_ROOT, or DNNL_INC_DIR")
endif()

# Find library
find_library(
  DNNL_LIBRARY
  mkldnn
  PATHS
  ${DNNL_ROOT}
  PATH_SUFFIXES
  lib
  HINTS
  ${DNNL_LIB_DIR}
)

if (DNNL_LIBRARY)
  message(STATUS "Using DNNL library found in ${DNNL_LIBRARY}")
else()
  message(STATUS "DNNL library not found; please set CMAKE_LIBRARY_PATH or DNNL_ROOT, or DNNL_LIB_DIR")
endif()

set(DNNL_LIBRARIES ${DNNL_LIBRARY})

# In order of preference, try to find and use MKL, mklml, then any system BLAS lib
if (MKL_FOUND)
  # Add MKL to DNNL deps if found
  message(STATUS "Using MKL with MKL-DNN")
  list(APPEND DNNL_LIBRARIES ${MKL_LIBRARIES})
  list(APPEND DNNL_INCLUDE_DIR ${MKL_INCLUDE_DIR})
else()
  message(STATUS "MKL not found; trying to fall back to mklml library")
  # MKL isn't found, so use the mini-MKL library (mklml) with MKL-DNN
  find_library(
    MKLML_LIBRARY
    NAMES
      mklml
      mklml_intel
    PATHS
    ${DNNL_ROOT}
    PATH_SUFFIXES
    lib
    HINTS
    ${DNNL_LIB_DIR}
    )
  # Find mklml headers. Perform this check anyways even though it's not clear
  # if they're being used, and they're not moved to the install dir on the
  # MKL-DNN install step.
  find_path(
    MKLML_INCLUDE_DIR
    mkl.h mkl_dnn_types.h
    PATHS
    ${DNNL_ROOT}
    PATH_SUFFIXES
    include
    external
    HINTS
    ${DNNL_INC_DIR}
    ${MKLML_INC_DIR}
    )
  if (MKLML_INCLUDE_DIR)
    message(STATUS "Found mklml headers: ${MKLML_INCLUDE_DIR}")
    list(APPEND DNNL_INCLUDE_DIR ${MKLML_INCLUDE_DIR})
  else()
    message(STATUS "Using MKL-DNN without mklml headers")
  endif()
  
  if (MKLML_LIBRARY)
    message(STATUS "Found libmklml: ${MKLML_LIBRARY}")
    message(STATUS "Using mklml with MKL-DNN")
    list(APPEND DNNL_LIBRARIES ${MKLML_LIBRARY})
  else()
    # If we still can't find mklml, look for any viable BLAS library as a last resort
    message(STATUS "mklml not found; trying to fall back to system BLAS library")
    if (BLAS_FOUND)
      message(STATUS "BLAS libraries found: ${BLAS_LIBRARIES}")
    else()
      # Build without a GEMM implementation
      message(STATUS "No GEMM implementation found - MKL-DNN will use internal GEMM implementation")
    endif()
  endif()
endif ()

# TODO: link?
# Override OpenMP configuration for DNNL if MKL is found, so MKL OpenMP is used.
if (EXISTS "${MKL_LIBRARIES_gomp_LIBRARY}")
  set(MKLIOMP5LIB ${MKL_LIBRARIES_gomp_LIBRARY} CACHE STRING "Override MKL-DNN omp dependency" FORCE)
elseif(EXISTS "${MKL_LIBRARIES_iomp5_LIBRARY}")
  set(MKLIOMP5LIB ${MKL_LIBRARIES_iomp5_LIBRARY} CACHE STRING "Override MKL-DNN omp dependency" FORCE)
elseif(EXISTS "${MKL_LIBRARIES_libiomp5md_LIBRARY}")
  set(MKLIOMP5DLL ${MKL_LIBRARIES_libiomp5md_LIBRARY} CACHE STRING "Override MKL-DNN omp dependency" FORCE)
else(EXISTS "${MKL_LIBRARIES_gomp_LIBRARY}")
  set(MKLIOMP5LIB "" CACHE STRING "Override MKL-DNN omp dependency" FORCE)
  set(MKLIOMP5DLL "" CACHE STRING "Override MKL-DNN omp dependency" FORCE)
endif(EXISTS "${MKL_LIBRARIES_gomp_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DNNL DEFAULT_MSG DNNL_LIBRARIES DNNL_INCLUDE_DIR)
