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

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DNNL DEFAULT_MSG DNNL_LIBRARIES DNNL_INCLUDE_DIR)