set(DIR ${FL_CORE_DIR}/test)
set(LIBS flashlight ${CMAKE_DL_LIBS})

build_test(SRC ${DIR}/autograd/AutogradTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/autograd/AutogradReductionTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/autograd/AutogradBinaryOpsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/autograd/AutogradUnaryOpsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/autograd/AutogradNormalizationTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/autograd/AutogradRnnTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/autograd/AutogradConv2DTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/autograd/AutogradComposedTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/common/DevicePtrTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/common/HistogramTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/common/LoggingTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/common/SerializationTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/common/UtilsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/optim/OptimTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/runtime/DeviceManagerTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/runtime/DeviceTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/runtime/DeviceTypeTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/runtime/AsyncCPUStreamTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/runtime/EventTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/runtime/FutureTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/nn/ModuleTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/nn/NNSerializationTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/nn/NNUtilsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/dataset/DatasetTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/dataset/DatasetUtilsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/meter/MeterTest.cpp LIBS ${LIBS})
if (FL_BUILD_DISTRIBUTED)
  build_test(SRC ${DIR}/distributed/AllReduceTest.cpp LIBS ${LIBS})
endif()
if (FL_BUILD_CONTRIB)
  build_test(SRC ${DIR}/contrib/modules/ContribModuleTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/contrib/modules/ContribSerializationTest.cpp LIBS ${LIBS})
endif()
build_test(SRC ${DIR}/tensor/TensorBaseTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/TensorBinaryOpsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/TensorBLASTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/TensorReductionTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/TensorUnaryOpsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/ComputeTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/IndexTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/ShapeTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/ShapeInferenceTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/tensor/BackendRegistryTest.cpp LIBS ${LIBS})
if (FL_USE_CUDA)
  build_test(SRC ${DIR}/runtime/CUDADeviceTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/runtime/CUDAStreamTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/runtime/EventCUDATest.cpp LIBS ${LIBS})
endif()

# Tensor backend tests
set(_build_tensor_backend_tests (NOT MSVC OR NOT BUILD_SHARED_LIBS))
if (${_build_tensor_backend_tests})
  build_test(SRC ${DIR}/tensor/TensorAdapterTest.cpp LIBS ${LIBS})
  if (FL_USE_ARRAYFIRE)
    build_test(SRC ${DIR}/tensor/TensorExtensionTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/af/ArrayFireTensorBaseTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/af/CachingMemoryManagerTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/af/MemoryFrameworkTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/af/MemoryInitTest.cpp LIBS ${LIBS})
    if (FL_ARRAYFIRE_USE_CPU)
      build_test(SRC ${DIR}/tensor/af/ArrayFireCPUStreamTest.cpp LIBS ${LIBS})
    endif()
  endif()
  if (FL_USE_CPU)
    build_test(SRC ${DIR}/tensor/cpu/CPUTensorTest.cpp LIBS ${LIBS})
  endif()
  build_test(SRC ${DIR}/tensor/jit/JitEvaluatorTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitNodeTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitCSETest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitElementwiseFusionTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitReductionOfElementwiseFusionTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitScalarFoldingTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitAlgebraicSimplificationTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitTransposeSimplificationTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitConvBatchNormFoldingTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitMatmulFusionTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitTensorTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitStreamTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitDeadOutputEliminationTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitFusionSideOutputTest.cpp LIBS ${LIBS})
  build_test(SRC ${DIR}/tensor/jit/JitMultiValueCompactionTest.cpp LIBS ${LIBS})
  if (FL_USE_ONEDNN)
    build_test(SRC ${DIR}/tensor/jit/JitOneDnnOpFusionTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/jit/JitOneDnnPrimitivePostOpFusionTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/jit/OneDnnJitEndToEndTest.cpp LIBS ${LIBS})
  endif()
  if (FL_USE_NATIVE_CUDA)
    build_test(SRC ${DIR}/tensor/jit/CudaJitEndToEndTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/jit/JitCudnnSubgraphLoweringTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/jit/JitBackwardEpilogueFusionTest.cpp LIBS ${LIBS})
  endif()
  if (FL_USE_NATIVE_CUDA)
    build_test(SRC ${DIR}/tensor/cuda/CudaTensorBaseTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/cuda/CudaTensorUnaryOpsTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/cuda/CudaTensorBinaryOpsTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/cuda/CudaTensorReductionTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/cuda/CudaTensorBLASTest.cpp LIBS ${LIBS})
    build_test(SRC ${DIR}/tensor/cuda/CudaTensorIndexTest.cpp LIBS ${LIBS})
  endif()
  # Cross-device transfer tests (require both CPU and CUDA backends)
  if (FL_USE_CPU AND FL_USE_NATIVE_CUDA)
    build_test(SRC ${DIR}/tensor/CrossDeviceTransferTest.cpp LIBS ${LIBS}
               PREPROC FL_TEST_BACKEND_CUDA)
  endif()
endif()

# ---------------------------------------------------------------------------
# Backend-specific variants of generic test suites (JIT full pipeline tests).
#
# These compile the same test sources with a backend-specific preprocessor
# define, causing flTestInit() (in TestInit.h) to set the appropriate JIT
# tensor type instead of the compile-time default (native CUDA if built,
# else CPU; see DefaultTensorType.h).
#
# Naming: SUFFIX controls both the binary name and CTest prefix.
#   Binary:  TensorBaseTest_CPU
#   CTest:   _CPU.TensorBaseTest.where
# Run all CPU backend tests:  ctest -R '^_CPU\.'
# Run all CUDA backend tests: ctest -R '^_CUDA\.'
# Run all ArrayFire baseline tests: ctest -R '^_AF\.'
# ---------------------------------------------------------------------------

# All test sources shared across CPU and CUDA backends.
set(_BACKEND_TEST_SRCS
  # Tensor tests
  ${DIR}/tensor/CopyTest.cpp
  ${DIR}/tensor/TensorBaseTest.cpp
  ${DIR}/tensor/TensorBinaryOpsTest.cpp
  ${DIR}/tensor/TensorBLASTest.cpp
  ${DIR}/tensor/TensorReductionTest.cpp
  ${DIR}/tensor/TensorUnaryOpsTest.cpp
  ${DIR}/tensor/ComputeTest.cpp
  ${DIR}/tensor/IndexTest.cpp
  ${DIR}/tensor/ShapeTest.cpp
  # Autograd tests
  ${DIR}/autograd/AutogradTest.cpp
  ${DIR}/autograd/AutogradReductionTest.cpp
  ${DIR}/autograd/AutogradBinaryOpsTest.cpp
  ${DIR}/autograd/AutogradUnaryOpsTest.cpp
  ${DIR}/autograd/AutogradNormalizationTest.cpp
  ${DIR}/autograd/AutogradRnnTest.cpp
  ${DIR}/autograd/AutogradConv2DTest.cpp
  ${DIR}/autograd/AutogradComposedTest.cpp
  # NN tests
  ${DIR}/nn/ModuleTest.cpp
  ${DIR}/nn/NNSerializationTest.cpp
  ${DIR}/nn/NNUtilsTest.cpp
  # Optim tests
  ${DIR}/optim/OptimTest.cpp
)

# Helper: register a full set of backend-suffixed tests.
#   SUFFIX:  e.g. "_CPU" or "_CUDA"
#   PREPROC: e.g. "FL_TEST_BACKEND_CPU" or "FL_TEST_BACKEND_CUDA"
function(build_backend_tests)
  set(options)
  set(oneValueArgs SUFFIX)
  set(multiValueArgs PREPROC)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  foreach(_src ${_BACKEND_TEST_SRCS})
    build_test(SRC ${_src} LIBS ${LIBS} SUFFIX ${ARG_SUFFIX} PREPROC ${ARG_PREPROC})
  endforeach()
  if (${_build_tensor_backend_tests})
    build_test(SRC ${DIR}/tensor/TensorAdapterTest.cpp LIBS ${LIBS} SUFFIX ${ARG_SUFFIX} PREPROC ${ARG_PREPROC})
  endif()
endfunction()

# CPU backend (JIT + CPUTensor)
if (FL_USE_CPU)
  build_backend_tests(SUFFIX _CPU PREPROC FL_TEST_BACKEND_CPU)
  if (FL_BUILD_CONTRIB)
    build_test(SRC ${DIR}/contrib/modules/ContribModuleTest.cpp LIBS ${LIBS} SUFFIX _CPU PREPROC FL_TEST_BACKEND_CPU)
    build_test(SRC ${DIR}/contrib/modules/ContribSerializationTest.cpp LIBS ${LIBS} SUFFIX _CPU PREPROC FL_TEST_BACKEND_CPU)
  endif()
endif()

# CUDA backend (JIT + CudaTensor)
if (FL_USE_NATIVE_CUDA)
  build_backend_tests(SUFFIX _CUDA PREPROC FL_TEST_BACKEND_CUDA)
endif()

# ArrayFire baseline (bare ArrayFireTensor, no JIT): the support/performance
# bar for the native backends. Compare with ctest -R '^_AF\.'.
if (FL_USE_ARRAYFIRE)
  build_backend_tests(SUFFIX _AF PREPROC FL_TEST_BACKEND_AF)
  if (FL_BUILD_CONTRIB)
    build_test(SRC ${DIR}/contrib/modules/ContribModuleTest.cpp LIBS ${LIBS} SUFFIX _AF PREPROC FL_TEST_BACKEND_AF)
    build_test(SRC ${DIR}/contrib/modules/ContribSerializationTest.cpp LIBS ${LIBS} SUFFIX _AF PREPROC FL_TEST_BACKEND_AF)
  endif()
endif()
