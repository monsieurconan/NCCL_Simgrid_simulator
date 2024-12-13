add_executable(snvccmain main_test.cpp)

target_link_libraries(snvccmain cudart nccl ${SimGrid_LIBRARY})