cmake_minimum_required(VERSION 3.16)

find_package(NCCL REQUIRED)
if (NCCL_FOUND)
  message(STATUS "NCCL found: (include: ${NCCL_INCLUDE_DIRS} lib: ${NCCL_LIBRARIES}")
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindNCCL.cmake)
endif()

target_link_libraries(flashlight PRIVATE ${NCCL_LIBRARIES})
target_include_directories(flashlight PRIVATE ${NCCL_INCLUDE_DIRS})

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/DistributedBackend.cpp
)

target_compile_definitions(
  flashlight
  PUBLIC
  "-DNO_NCCL_COMM_DESTROY_HANDLE"
)
    
