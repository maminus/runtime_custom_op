# ONNXRuntime custom operator implementation sample

## motivation
[ONNX](https://github.com/onnx/onnx) model can have custom operator.
But tools somewhat like [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier) needs inference by [ONNXRuntime](https://github.com/microsoft/onnxruntime/).
So we need implementation of custom operator for onnxruntime.

## samples
There are simple FMA(fused multiply-add) custom operator samples.

* [pure Python sample](fma_pure_python)
* [C++ implementation sample](fma_use_t)
* [C++ implementation multi-domain sample](fma_multi_domain)
