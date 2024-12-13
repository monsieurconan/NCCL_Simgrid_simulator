# Simulation library of NCCL using SimGrid
## Description

This project aims to simulate NCCL code, by implementing a subset of NCCL and Cuda Runtime Library in SimGrid.

## How to build
### Requirements

cmake version 3.10
SimGrid version 3.35 (since release version is only 3.30, compiling from the source is mandatory, see https://simgrid.org/doc/latest/Installing_SimGrid.html for installation details)

### Build the library
```
mkdir build
cd build
cmake .. -DSimGrid_PATH=/path/to/simgrid/
make
```

## How to use 
Copy tests/main_test.cpp into your project, replace nccl_test function by your former main. 

If your project uses MPI, there can be an fake deadlock issue : a simgrid mecanism throws an exception if all process enter a supending state
which can happen if all process call cudaStreamSynchronise for example. The trick to resolve this is to add a another process that doesn't really 
do anything but prevent the issue.

You can either write your own platform file (see section Platform description rules) or use the default.
Compile your project with a standart c++ compiler, using this library as a substitute of cudart and nccl.

## Platform description rules
In SimGrid, the description of the simulated machine is refered as a platform.
Documentation of simgrid platforms can be found at https://simgrid.org/doc/latest/Platform.html.

To work properly, this simulator requires additionnal data : Host must have a "type" property, either "cpu" or "gpu".
A cpu Host has to be defined in the same Netzone as the gpu Hosts it can access.

If you use cudaGetDeviceProperties in your code, you have to set it beforehand :
```
simgrid::s4u::Host::extension_set<cudaDeviceProp>(your_cuda_device_property)
```

