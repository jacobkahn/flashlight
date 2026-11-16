# ----------------------------- Sources -----------------------------

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/AlgebraicSimplification.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CommonSubexpressionElimination.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ConvBatchNormFolding.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ElementwiseFusion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ReductionOfElementwiseFusion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ScalarFolding.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TransposeSimplification.cpp
)
