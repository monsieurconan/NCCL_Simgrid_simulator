
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
find_package(SimGrid REQUIRED)
include_directories(${SimGrid_INCLUDE_DIR})

add_library(cudart SHARED cuda_runtime.cpp cuda_graph.cpp internal.cpp
 gpu_activity.cpp platform.cpp tracer.cpp)

target_include_directories(cudart PUBLIC include)

target_link_libraries(cudart ${SimGrid_LIBRARY})

include_directories(include)