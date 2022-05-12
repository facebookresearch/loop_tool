project(loop_tool)
cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/)

set(LIB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
set(LIB_PRIVATE_INCLUDE ${SRC_DIR})

file(GLOB CORE_SRCS
  ${SRC_DIR}/core/*.cpp
)
list(APPEND CORE_SRCS "${SRC_DIR}/frontends/lazy.cpp")
list(APPEND CORE_SRCS "${SRC_DIR}/frontends/mutate.cpp")

option(BUILD_WASM "Build a WebAssembly compilation target" OFF)
option(BUILD_WASM_RUNTIME "Build a WebAssembly runtime with WAMR" OFF)
option(BUILD_ES6 "Build an ES6 target in emcc" OFF)
if(DEFINED EMCC_DIR)
  set(BUILD_WASM ON)
endif()
if (BUILD_WASM)
  list(APPEND CORE_SRCS "${SRC_DIR}/backends/wasm/wasm.cpp")
  set(LIB_INCLUDE ${LIB_INCLUDE} "${CMAKE_CURRENT_SOURCE_DIR}/extern/wasmblr")
endif()

if (BUILD_WASM_RUNTIME)
  list(APPEND CORE_SRCS "${SRC_DIR}/backends/wasm/wasm_runtime.cpp")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/wasm-micro-runtime")
endif()

option(BUILD_LOOP_NEST "Build loop_nest backend for fast contractions on CPU" OFF)
if (BUILD_LOOP_NEST)
  list(APPEND CORE_SRCS "${SRC_DIR}/backends/cpu/loop_nest.cpp")
  set(DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OFF)
  set(DABUN_BUILD_TESTS_FOR_ALL_ARCH_VEX OFF)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/loop_nest")
  set_target_properties(dabun PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(CPU_SRCS ${SRC_DIR}/backends/cpu/cpp.cpp)

set(WHOLE_ARCHIVE_START "-Wl,--whole-archive -Wl,--no-as-needed")
set(WHOLE_ARCHIVE_END "-Wl,--no-whole-archive -Wl,--as-needed")
if (APPLE)
  set(WHOLE_ARCHIVE_START "-Wl,-all_load")
  set(WHOLE_ARCHIVE_END "-Wl,-noall_load")
endif()


add_library(loop_tool SHARED ${CORE_SRCS} ${CPU_SRCS})
target_include_directories(loop_tool PRIVATE ${LIB_PRIVATE_INCLUDE})
target_include_directories(loop_tool PUBLIC
    "$<BUILD_INTERFACE:${LIB_INCLUDE}>"
    $<INSTALL_INTERFACE:include/>
)
if (BUILD_LOOP_NEST)
  target_link_libraries(loop_tool PUBLIC dabun)
endif()
if (BUILD_WASM_RUNTIME)
  target_link_libraries(loop_tool PUBLIC iwasm_shared)
  target_include_directories(loop_tool PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/extern/wasm-micro-runtime/core/iwasm/include")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(loop_tool PUBLIC ${CMAKE_DL_LIBS} PRIVATE Threads::Threads)

list(APPEND CMAKE_MODULE_PATH "${SRC_DIR}/backends/cuda")
if(DEFINED ENV{CUDA_PATH})
  set(CUDAToolkit_ROOT $ENV{CUDA_PATH})
endif()
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  message("Found CUDA toolkit version ${CUDAToolkit_VERSION}")
  file(GLOB CUDA_SRCS ${SRC_DIR}/backends/cuda/*.cpp)

  add_library(loop_tool_cuda SHARED ${CUDA_SRCS})
  target_include_directories(loop_tool_cuda PUBLIC ${LIB_INCLUDE} ${SRC_DIR}/backends/cuda ${CUDAToolkit_INCLUDE_DIRS})
  #target_link_libraries(loop_tool_cuda CUDA::cudart_static CUDA::nvrtc)
endif()

option(BUILD_TESTS "Build all available tests" ON)
if (BUILD_TESTS)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/)

file(GLOB TEST_SRCS ${TEST_DIR}/test_*.cpp)
list(APPEND TEST_SRCS "${TEST_DIR}/test.cpp") # main file
add_library(loop_tool_test_utils "${TEST_DIR}/utils.cpp")
target_include_directories(loop_tool_test_utils PUBLIC ${SRC_DIR})

if (BUILD_LOOP_NEST)
  list(APPEND TEST_SRCS "${TEST_DIR}/loop_nest_test.cpp")
endif()

if (BUILD_WASM)
  list(APPEND TEST_SRCS "${TEST_DIR}/wasm_test.cpp")
endif()
if (BUILD_WASM_RUNTIME)
  list(APPEND TEST_SRCS "${TEST_DIR}/wasm_runtime_test.cpp")
endif()

if (CUDAToolkit_FOUND)
  list(APPEND TEST_SRCS "${TEST_DIR}/cuda_test.cpp")
endif() # CUDAToolkit_FOUND

add_executable(loop_tool_test ${TEST_SRCS})
target_include_directories(loop_tool_test PUBLIC ${SRC_DIR})
target_link_libraries(loop_tool_test loop_tool_test_utils loop_tool)

if (CUDAToolkit_FOUND)
  target_link_libraries(loop_tool_test
    ${WHOLE_ARCHIVE_START}
    loop_tool_cuda
    ${WHOLE_ARCHIVE_END}
  )
endif() # CUDAToolkit_FOUND

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
     -rdynamic
      ${WHOLE_ARCHIVE_START}
      loop_tool_cuda
      loop_tool
      ${WHOLE_ARCHIVE_END}
    )
  else()
	  target_link_libraries(loop_tool_py PUBLIC
     -rdynamic
      ${WHOLE_ARCHIVE_START}
      loop_tool
      ${WHOLE_ARCHIVE_END}
    )
  endif()
else()
  message("To build python bindings, pip install pybind11 and run `cmake .. -Dpybind11_DIR=$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')`")
endif()

FUNCTION(PREPEND var prefix)
   SET(listVar "")
   FOREACH(f ${ARGN})
      LIST(APPEND listVar "${prefix}/${f}")
   ENDFOREACH(f)
   SET(${var} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)

if(DEFINED EMCC_DIR)
  SET(EMCC ${EMCC_DIR}/emcc)
  message("Using ${EMCC} to compile javascript bindings...")
  SET(EMCC_INCLUDE ${EMCC_DIR}/system/include)
  file(GLOB JS_SRCS ${SRC_DIR}/frontends/javascript*.cpp)
  SET(EMCC_FLAGS -s NO_DISABLE_EXCEPTION_CATCHING -s MODULARIZE -s SINGLE_FILE=1 -s "EXPORT_NAME=\"createMyModule\"" -s "TOTAL_MEMORY=268435456")
  set(EMCC_TARGET libloop_tool.js)
  if (BUILD_ES6)
      SET(EMCC_FLAGS ${EMCC_FLAGS} -s EXPORT_ES6=1)
      set(EMCC_TARGET libloop_tool.mjs)
  endif()
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    message("Running a debug build for emcc...")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET(EMCC_FLAGS ${EMCC_FLAGS} -gsource-map)
    endif()
    SET(EMCC_FLAGS ${EMCC_FLAGS} -g -s ASSERTIONS=1 -s DEMANGLE_SUPPORT)
  else()
    SET(EMCC_FLAGS ${EMCC_FLAGS} -Oz)
  endif()
  PREPEND(INC "-I" ${LIB_INCLUDE} ${LIB_PRIVATE_INCLUDE})
  add_custom_command(OUTPUT loop_tool_js_emcc
    COMMAND ${EMCC} -I${EMCC_INCLUDE} ${INC} ${CORE_SRCS} ${JS_SRCS} ${EMCC_FLAGS} -o ${EMCC_TARGET} --bind
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${CORE_SRCS}"
    COMMENT "Compiling with emcc"
    VERBATIM
  )

  add_custom_target(loop_tool_js ALL DEPENDS loop_tool_js_emcc)
endif()

