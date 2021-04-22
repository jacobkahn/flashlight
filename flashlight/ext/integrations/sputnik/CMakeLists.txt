cmake_minimum_required(VERSION 3.10)

if (NOT FL_USE_CUDA)
  message(FATAL_ERROR "Flashlight sputnik integration "
    "only available with the CUDA backendo")
endif()

find_package(sputnik REQUIRED)

target_link_libraries(
  flashlight
  PUBLIC
  sputnik::sputnik)

# target_sources(
#   flashlight
#   PRIVATE
#   ${CMAKE_CURRENT_LIST_DIR}/SputnikInterface.cpp
#   )
