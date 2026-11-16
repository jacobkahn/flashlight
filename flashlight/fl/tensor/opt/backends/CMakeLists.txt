# ----------------------------- Sources -----------------------------
if (FL_USE_ONEDNN)
  include(${CMAKE_CURRENT_LIST_DIR}/onednn/CMakeLists.txt) # onednn
endif()

if (FL_USE_NATIVE_CUDA)
  include(${CMAKE_CURRENT_LIST_DIR}/cuda/CMakeLists.txt) # cuda
endif()
