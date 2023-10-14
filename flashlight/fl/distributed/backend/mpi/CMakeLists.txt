cmake_minimum_required(VERSION 3.16)

find_package(MPI REQUIRED)

target_link_libraries(flashlight PUBLIC MPI::MPI_CXX)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/MPIBackend.cpp
)
