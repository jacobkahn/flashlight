target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Device.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeviceManager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeviceType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Stream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SynchronousStream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/AsyncCPUStream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Event.cpp
  ${CMAKE_CURRENT_LIST_DIR}/EventImpl.cpp
  )

if (FL_USE_CUDA)
  target_link_libraries(flashlight PUBLIC $<BUILD_INTERFACE:CUDA::cudart>)

  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/CUDADevice.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAStream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAEventImpl.cpp
    )
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  "-DNO_CUDA_STREAM_DESTROY_EVENT"
  )
