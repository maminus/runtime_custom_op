# Custom operator implementation C++(different domain name corresponding with data types and CUDAExecutionProvider)
This is a sample implementation of FAM operator with CUDA.

```
FMA = A * B + C

# C is optional
```

# pre-requirements
* numpy
* onnx
* onnxruntime-gpu (NEED CUDAExecutionProvider)
* git > 2.25
* cmake >= 3.18

# files
* CMakeLists.txt
  - cmake file to make so
* cuda_kernel.cu
  - CUDA kernel implementation of custom operator
* cuda_kernel.cuh
  - header file of CUDA host functions
* fma_multi_domain.cpp
  - C++ implementation of custom operator
* demo_fma.py
  - script which demonstrate how to inference ONNX model(has custom operator) with CUDAExecutionProvider
* docker/Dockerfile
  - Dockerfile of ONNXRuntime with CUDAExecutionProvider for Jetson

# Jetson Environment

docker command sample
```
cd runtime_custom_op/fma_multi_with_cuda/docker
docker build --tag cuda_ep_base .

# run container with fma_multi_with_cuda directory is mounted to /work
docker run -it --rm --name custom_op_demo --net=host --runtime nvidia --shm-size=1g -e DISPLAY=$DISPLAY -v ~/runtime_custom_op/fma_multi_with_cuda:/work -v /tmp/.X11-unix/:/tmp/.X11-unix cuda_ep_base
```

# build
1. check onnxruntime package version
```
python3 -c "import onnxruntime as ort; print(ort.__version__)"
```

2. checkout onnxruntime headers
```
cd
git clone --filter=blob:none --no-checkout https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git sparse-checkout init --cone
git sparse-checkout set include/onnxruntime/core/session
git checkout -b 1.13.1 v1.13.1
```
match tag version with onnxruntime package version

!CAUTION!

docker/Dockerfile's onnxruntime is not specify version. so if you use docker/Dockerfile, checkout latest version at main branch.

3. make
```
# change current to fma_multi_with_cuda dir.(in docker case, /work)
cd fma_multi_with_cuda

mkdir build
cd build
cmake -DONNXRUNTIME_TOP_DIR=~/onnxruntime ..
make
```

# run
```
cd fma_multi_with_cuda
cp build/*my_custom_multi_with_cuda.* .
python3 demo_fma.py
```

# known limitation
* There can't has operator overload
  - only one op_type name can register at one domain

