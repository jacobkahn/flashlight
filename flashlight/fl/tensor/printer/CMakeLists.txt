# ----------------------------- Sources -----------------------------
target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/GraphvizPrinter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ScopedPostEvalGraphvizPrinter.cpp
)
