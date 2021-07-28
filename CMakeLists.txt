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

set(LIB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_library(loop_tool SHARED ${CORE_SRCS})
target_include_directories(loop_tool PUBLIC ${LIB_INCLUDE})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(loop_tool PRIVATE Threads::Threads)

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
  add_executable(loop_tool_cuda_test ${TEST_DIR}/cuda_test.cpp)
  target_link_libraries(loop_tool_cuda_test
    -Wl,--no-as-needed
    -Wl,--whole-archive
    loop_tool
    loop_tool_cuda
    -Wl,--no-whole-archive
    -Wl,--as-needed
  )
endif() # CUDAToolkit_FOUND

add_executable(loop_tool_test ${TEST_DIR}/test.cpp)
target_link_libraries(loop_tool_test loop_tool)

add_executable(loop_tool_test_custom_backend ${TEST_DIR}/test_backend.cpp)
target_link_libraries(loop_tool_test_custom_backend loop_tool)

endif() # BUILD_TESTS

find_package(pybind11 CONFIG)
if (pybind11_FOUND)
  message("Building python bindings...")
  file(GLOB PY_SRCS ${SRC_DIR}/frontends/python*.cpp)
  pybind11_add_module(loop_tool_py MODULE ${PY_SRCS})
  target_include_directories(loop_tool_py PUBLIC ${LIB_INCLUDE})
  if (CUDAToolkit_FOUND)
    target_compile_definitions(loop_tool_py PUBLIC ENABLE_CUDA)
    target_link_libraries(loop_tool_py PUBLIC
      -Wl,--no-as-needed
      -Wl,--whole-archive
      loop_tool_cuda
      loop_tool
      -Wl,--no-whole-archive
      -Wl,--as-needed
    )
  else()
	  target_link_libraries(loop_tool_py PUBLIC
      -Wl,--no-as-needed
      -Wl,--whole-archive
      loop_tool
      -Wl,--no-whole-archive
      -Wl,--as-needed
    )
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

