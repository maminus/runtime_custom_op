import onnx
import numpy as np
import onnxruntime as ort


print('FMA custom op sample with CUDA')

# load custom op library
option = ort.SessionOptions()
option.register_custom_ops_library('./libmy_custom_multi_with_cuda.so')
# specify execution providers which you use
providers = ['CUDAExecutionProvider']
# opset v17 at default domain
opsets = [onnx.helper.make_opsetid('', 17)]

# ONNX model creation
inputs = [
    onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [1]),
    onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [1]),
    onnx.helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT, [1]),
]
outputs = [onnx.helper.make_tensor_value_info('out', onnx.TensorProto.FLOAT, [1])]
nodes = [onnx.helper.make_node('Fma', ['A', 'B', 'C'], ['out'], domain='my_ops.float')]
graph = onnx.helper.make_graph(nodes, 'fma', inputs, outputs)
model = onnx.helper.make_model(graph, opset_imports=opsets)

# create session object with custom op, CUDAExecutionProvider
sess = ort.InferenceSession(model.SerializeToString(), option, providers)

# input data
A = np.ones([1], dtype=np.float32)
B = np.ones([1], dtype=np.float32)
C = np.ones([1], dtype=np.float32)

# inference
results = sess.run(None, {'A': A, 'B': B, 'C': C})

# print result
print(f'fp32: {A} * {B} + {C} -> {results[0]}')


# fp64 version
inputs = [
    onnx.helper.make_tensor_value_info('A', onnx.TensorProto.DOUBLE, [1]),
    onnx.helper.make_tensor_value_info('B', onnx.TensorProto.DOUBLE, [1]),
    onnx.helper.make_tensor_value_info('C', onnx.TensorProto.DOUBLE, [1]),
]
outputs = [onnx.helper.make_tensor_value_info('out', onnx.TensorProto.DOUBLE, [1])]
nodes = [onnx.helper.make_node('Fma', ['A', 'B', 'C'], ['out'], domain='my_ops.double')]
graph = onnx.helper.make_graph(nodes, 'fma', inputs, outputs)
model = onnx.helper.make_model(graph, opset_imports=opsets)

sess = ort.InferenceSession(model.SerializeToString(), option, providers)
A = np.ones([1], dtype=np.float64)
B = np.ones([1], dtype=np.float64)
C = np.ones([1], dtype=np.float64)
results = sess.run(None, {'A': A, 'B': B, 'C': C})
print(f'fp64: {A} * {B} + {C} -> {results[0]}')

