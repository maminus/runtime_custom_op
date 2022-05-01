# Custom operator implementation C++(different domain name corresponding with data types)
This is a sample implementation of FAM operator.

```
FMA = A * B + C

# C is optional
```

# pre-requirements
* numpy
* onnx
* onnxruntime
* git > 2.25
* cmake

# files
* CMakeLists.txt
  - cmake file to make so
* fma_multi_domain.cpp
  - C++ implementation of custom operator
* test_fma.py
  - test script which demonstrate how to inference ONNX model(has custom operator)

# build
1. check onnxruntime package version
```
python3 -c "import onnxruntime as ort; print(ort.__version__)"
```

2. checkout onnxruntime headers
```
git clone --filter=blob:none --no-checkout https://github.com/microsoft/onnxruntime.git
git sparse-checkout init --cone
git sparse-checkout set include/onnxruntime/core/session
git checkout -b 1.11.0 v1.11.0
```
match tag version with onnxruntime package version

3. make
```
mkdir build
cd build
cmake -DONNXRUNTIME_TOP_DIR=../onnxruntime ..
make
```

# run
```
python3 test_fma.py
```

# known limitation
* There can't has operator overload
  - only one op_type name can register at one domain
