cmake_minimum_required(VERSION 3.20)
project(mx_float_cuda LANGUAGES CXX CUDA)

# Allow user to specify GPU architectures or default to a recent set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable all warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(mx_fp6 STATIC
  src/fp6.cu
)

target_include_directories(mx_fp6 PUBLIC include)

set_target_properties(mx_fp6 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(example src/main.cu)

target_link_libraries(example PRIVATE mx_fp6)

add_executable(test_fp6 tests/test_fp6.cu)

target_link_libraries(test_fp6 PRIVATE mx_fp6)

# Add a simple test via CTest
include(CTest)
if (BUILD_TESTING)
  add_test(NAME fp6_roundtrip COMMAND test_fp6)
endif()
