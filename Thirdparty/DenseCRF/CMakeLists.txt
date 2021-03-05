cmake_minimum_required(VERSION 3.8)
project(densecrf)
 
# CPU Example Target
add_executable(example_cpu
  examples/example_cpu.cpp
  examples/util.cpp
  examples/util.h
  include/densecrf_base.h
  include/densecrf_cpu.h
  include/pairwise_cpu.h
  include/permutohedral_cpu.h
)

add_library(densecrf SHARED
  include/densecrf_base.h
  include/densecrf3d.h
  include/pairwise3d.h
  include/permutohedral_cpu.h
)
target_link_libraries (${PROJECT_NAME}
	)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(example_cpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(example_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >)
target_compile_features(example_cpu PUBLIC cxx_std_11)