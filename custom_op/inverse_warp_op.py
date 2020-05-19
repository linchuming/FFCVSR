import tensorflow as tf
from tensorflow.python.framework import ops
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'inverse_warp.so')
_inverse_warp_module = tf.load_op_library(filename)
inverse_warp = _inverse_warp_module.inverse_warp
inverse_warp_grad = _inverse_warp_module.inverse_warp_grad

@ops.RegisterGradient("InverseWarp")
def _inverse_warp_grad(op, grad):
	inp = op.inputs[0]
	flow = op.inputs[1]
	# compute gradient
	inp_grad, flow_grad = inverse_warp_grad(inp, flow, grad)
	
	return [inp_grad, flow_grad]