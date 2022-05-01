# Custom operator implementation C++(using type "T")
This is a sample implementation of FAM operator using type "T" (any type).

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
* fma_use_t.cpp
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
* The operator has 3 inputs, but input[0] only has type "T"
