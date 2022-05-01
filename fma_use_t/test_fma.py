import onnx
import numpy as np
import onnxruntime as ort

option = ort.SessionOptions()
option.register_custom_ops_library('./libmy_custom_t.so')

# FMA(float)
inputs = [
    onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [1]),
    onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [1]),
    onnx.helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT, [1]),
]
outputs = [onnx.helper.make_tensor_value_info('out', onnx.TensorProto.FLOAT, [1])]
nodes = [onnx.helper.make_node('Fma', ['A', 'B', 'C'], ['out'], domain='my_ops')]
graph = onnx.helper.make_graph(nodes, 'fma', inputs, outputs)
model = onnx.helper.make_model(graph)

sess = ort.InferenceSession(model.SerializeToString(), option)
A = np.ones([1], dtype=np.float32)
B = np.ones([1], dtype=np.float32)
C = np.ones([1], dtype=np.float32)
results = sess.run(None, {'A': A, 'B': B, 'C': C})
assert results[0] == 2

# FMA(double)
inputs = [
    onnx.helper.make_tensor_value_info('A', onnx.TensorProto.DOUBLE, [1]),
    onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [1]),
]
outputs = [onnx.helper.make_tensor_value_info('out', onnx.TensorProto.DOUBLE, [1])]
nodes = [onnx.helper.make_node('Fma', ['A', 'B'], ['out'], domain='my_ops')]
graph = onnx.helper.make_graph(nodes, 'fma', inputs, outputs)
model = onnx.helper.make_model(graph)

sess = ort.InferenceSession(model.SerializeToString(), option)
A = np.ones([1], dtype=np.float64)
B = np.ones([1], dtype=np.float32)
results = sess.run(None, {'A': A, 'B': B})
assert results[0] == 1
