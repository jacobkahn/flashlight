# ----------------------------- OneDNN Ops -----------------------------
# OneDNN-accelerated tensor operations: autograd primitives (conv2d, pool2d,
# batchnorm, rnn) and shared utilities. These are free functions in
# fl::detail::onednn namespace, usable by any backend that has a dnnl::engine.

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DnnlUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnOps.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnAsyncCPUStream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Conv2D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Pool2D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BatchNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RNN.cpp
)
