import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import PyOp, onnx_op, PyOrtFunction


_ONNX_FILE_NAME = 'fma.onnx'

@onnx_op(op_type='Fma', inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float], outputs=[PyOp.dt_float])
def fma(a, b, c):
	"""Custom operator FMA
	"""
    return a * b + c


# make FMA model
inputs = [
    onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [1]),
    onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [1]),
    onnx.helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT, [1]),
]
outputs = [onnx.helper.make_tensor_value_info('out', onnx.TensorProto.FLOAT, [1])]
nodes = [onnx.helper.make_node('Fma', ['A', 'B', 'C'], ['out'], domain='ai.onnx.contrib')]
graph = onnx.helper.make_graph(nodes, 'fma', inputs, outputs)
model = onnx.helper.make_model(graph)
onnx.save(model, _ONNX_FILE_NAME)

# input data
A = np.ones([1], dtype=np.float32)
B = np.ones([1], dtype=np.float32)
C = np.ones([1], dtype=np.float32)

# do inference
model_func = PyOrtFunction.from_model(_ONNX_FILE_NAME)
result = model_func(A, B, C)

# test result value
assert result == 2
