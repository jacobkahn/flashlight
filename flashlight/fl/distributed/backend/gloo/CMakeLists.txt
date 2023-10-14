cmake_minimum_required(VERSION 3.16)

find_package(Gloo CONFIG)
if (NOT Gloo_FOUND)
  if (NOT FL_BUILD_STANDALONE)
    message(FATAL_ERROR "Required dependency Gloo not found")
  endif()
  message(STATUS "Gloo not found - downloading and building from source")
  include(${PROJECT_SOURCE_DIR}/cmake/BuildGloo.cmake)
endif()

target_link_libraries(flashlight PRIVATE gloo)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/DistributedBackend.cpp
)

