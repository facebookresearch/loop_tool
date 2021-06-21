project(loop_tool)
cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 14)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/)

file(GLOB CORE_SRCS
  ${SRC_DIR}/core/*.cpp
)

file(GLOB CPU_SRCS
  ${SRC_DIR}/backends/cpu/*.cpp
)

set(LIB_INCLUDE ${SRC_DIR}/core)

add_library(loop_tool SHARED ${CORE_SRCS})
target_include_directories(loop_tool PUBLIC ${LIB_INCLUDE})

list(APPEND CMAKE_MODULE_PATH "${SRC_DIR}/backends/cuda")
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  message("Found CUDA toolkit version ${CUDAToolkit_VERSION}")
  file(GLOB CUDA_SRCS ${SRC_DIR}/backends/cuda/*.cpp)

  add_library(loop_tool_cuda SHARED ${CUDA_SRCS})
  target_include_directories(loop_tool_cuda PUBLIC ${LIB_INCLUDE} ${SRC_DIR}/backends/cuda ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(loop_tool_cuda CUDA::cudart_static CUDA::nvrtc)
endif()

option(BUILD_TESTS "Build all available tests" ON)
if (BUILD_TESTS)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/)

if (CUDAToolkit_FOUND)
  add_executable(cuda_test ${TEST_DIR}/cuda_test.cpp)
  target_link_libraries(cuda_test loop_tool loop_tool_cuda)
endif() # CUDAToolkit_FOUND

add_executable(test ${TEST_DIR}/test.cpp)
target_link_libraries(test loop_tool)# CUDA::cudart_static CUDA::nvrtc)

endif() # BUILD_TESTS

find_package(pybind11 CONFIG)
if (pybind11_FOUND)
  message("Building python bindings...")
  file(GLOB PY_SRCS ${SRC_DIR}/frontends/python*.cpp)
  pybind11_add_module(loop_tool_py MODULE ${PY_SRCS})
  target_include_directories(loop_tool_py PUBLIC ${LIB_INCLUDE})
  target_link_libraries(loop_tool_py PUBLIC loop_tool)
  if (CUDAToolkit_FOUND)
    target_compile_definitions(loop_tool_py PUBLIC ENABLE_CUDA)
    target_link_libraries(loop_tool_py PUBLIC loop_tool_cuda)
  endif()
else()
  message("To build python bindings, pip install pybind11 and run `cmake .. -Dpybind11_DIR=$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')`")
endif()

if(DEFINED ENV{EMCC_DIR})
  SET(EMCC $ENV{EMCC_DIR}/emcc)
  SET(EMCC_INCLUDE $ENV{EMCC_DIR}/system/include)
  file(GLOB JS_SRCS ${SRC_DIR}/frontends/javascript*.cpp)
  add_custom_command(OUTPUT loop_tool_js_emcc
    COMMAND ${EMCC} -I${EMCC_INCLUDE} -I${LIB_INCLUDE} ${CORE_SRCS} ${JS_SRCS} -o loop_tool.js -Oz --bind
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${CORE_SRCS}"
    COMMENT "Compiling with emcc"
    VERBATIM
  )

  add_custom_target(loop_tool_js ALL DEPENDS loop_tool_js_emcc)
endif()

