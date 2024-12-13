cmake_minimum_required(VERSION 3.10)
project(libcuda)

add_subdirectory(cuda ${PROJECT_BINARY_DIR}/cuda)

add_subdirectory(nccl ${PROJECT_BINARY_DIR}/nccl)

add_subdirectory(tests ${PROJECT_BINARY_DIR}/tests)
