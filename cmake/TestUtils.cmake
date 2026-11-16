set(GTEST_IMPORTED_TARGETS "")

# Get or find Google Test and Google Mock
find_package(GTest 1.10.0)
if (NOT GTEST_FOUND)
  if (NOT TARGET gtest)
    message(STATUS "googletest not found - will download and build from source")
    # Download and build googletest
    include(${PROJECT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)
    list(APPEND GTEST_IMPORTED_TARGETS GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
  endif()
else()
  message(STATUS "gtest found: (include: ${GTEST_INCLUDE_DIRS}, lib: ${GTEST_BOTH_LIBRARIES}")
  if (TARGET GTest::GTest)
    # We found the differently-named CMake targets from FindGTest
    if (NOT TARGET GTest::Main)
      message(FATAL_ERROR "Google Test must be built with main")
    endif()
    list(APPEND GTEST_IMPORTED_TARGETS GTest::GTest GTest::Main)
  endif()
  if (NOT TARGET GTest::gmock)
    find_package(GMock REQUIRED)
    message(STATUS "gmock found: (include: ${GMOCK_INCLUDE_DIRS}, lib: ${GMOCK_BOTH_LIBRARIES})")
  endif()
  list(APPEND GTEST_IMPORTED_TARGETS GTest::gmock GTest::gmock_main)
  message(STATUS "Found gtest and gmock on system.")
endif()

include(GoogleTest)
find_package(Threads REQUIRED)

# Optional OpenMP thread count for tests run through ctest. Empty (default)
# leaves the environment alone; the library itself never sets a thread count.
# Useful on large multi-socket/SMT machines, where OpenMP's default (all logical
# CPUs) makes tiny CPU ops very slow.
set(FL_TEST_NUM_THREADS "" CACHE STRING
  "If set, run ctest tests with OMP_NUM_THREADS=<value>")

function(build_test)
  set(options)
  set(oneValueArgs SRC SUFFIX)
  set(multiValueArgs LIBS PREPROC)
  cmake_parse_arguments(build_test "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  get_filename_component(src_name ${build_test_SRC} NAME_WE)
  set(target "${src_name}${build_test_SUFFIX}")
  add_executable(${target} ${build_test_SRC})
  target_compile_features(${target} PUBLIC cxx_std_20)
  if (TARGET gtest)
    add_dependencies(${target} gtest) # make sure gtest is built first
  endif()
  target_link_libraries(
    ${target}
    PUBLIC
    ${GTEST_IMPORTED_TARGETS}
    ${build_test_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    )
  target_include_directories(
    ${target}
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    )
  target_compile_definitions(
    ${target}
    PUBLIC
    ${build_test_PREPROC}
    )
  if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(
      ${target}
      PUBLIC
      GTEST_LINKED_AS_SHARED_LIBRARY=$<BOOL:${BUILD_SHARED_LIBS}>
      GMOCK_LINKED_AS_SHARED_LIBRARY=$<BOOL:${BUILD_SHARED_LIBS}>
    )
  endif()
  if (build_test_SUFFIX)
    gtest_add_tests(TARGET ${target} TEST_PREFIX "${build_test_SUFFIX}."
      TEST_LIST registered_tests)
  else()
    gtest_add_tests(TARGET ${target} TEST_LIST registered_tests)
  endif()
  if (FL_TEST_NUM_THREADS AND registered_tests)
    set_tests_properties(${registered_tests} PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=${FL_TEST_NUM_THREADS}")
  endif()
endfunction(build_test)
