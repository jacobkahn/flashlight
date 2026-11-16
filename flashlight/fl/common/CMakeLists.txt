target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DevicePtr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Histogram.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Timer.cpp
)

if (${FL_BUILD_PLUGIN})
  target_sources(flashlight PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Plugin.cpp)
else()
  message(WARNING "Compiler doesn't support -rdynamic - not building Plugin")
endif()

# A native threading library
find_package(Threads REQUIRED)
target_link_libraries(flashlight PUBLIC Threads::Threads)

# Dynamic lib loading needed for Plugins
target_link_libraries(flashlight PUBLIC ${CMAKE_DL_LIBS})

# Standard-conforming preprocessor for __VA_ARGS__ in serialization macros
if (MSVC)
  target_compile_options(flashlight PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>)
endif()

# std::filesystem is part of the standard library in C++17+/GCC >= 9
# No extra linking needed since we require C++20

# backward
include(${CMAKE_CURRENT_LIST_DIR}/stacktrace/CMakeLists.txt)
