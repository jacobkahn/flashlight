# ----------------------------- Sources -----------------------------

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnJitOptimizerExtension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnOpFusion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnPrimitivePostOpFusion.cpp
)
