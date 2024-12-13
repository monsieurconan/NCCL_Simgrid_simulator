set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
find_package(SimGrid REQUIRED)
include_directories(${SimGrid_INCLUDE_DIR})

add_library(nccl SHARED coll.cpp nccl.cpp  nccl_internal.cpp)

target_include_directories(nccl PUBLIC include)

target_link_libraries(nccl cudart ${SimGrid_LIBRARY})