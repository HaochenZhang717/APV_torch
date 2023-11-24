import pickle

import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras import layers as tfkl
import numpy as np
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def to_torch_params(x: list):
    # y = []
    # for item in x:
    #     y.append(np.array(item))
    y = {}
    y['weight'] = torch.permute(torch.from_numpy(np.array(x[0])), (3,2,0,1))
    y['bias'] = torch.from_numpy(np.array(x[1]))

    y['weight'].type(torch.float32)
    y['bias'].type(torch.float32)
    return y


torch_conv = nn.Conv2d(3, 48, 4, stride=2)
nn.init.xavier_uniform_()

tf_conv = tfkl.Conv2D(48, 4, strides=2)

input = np.random.randn(800, 64, 64, 3)
input_torch = torch.permute(torch.from_numpy(input), (0,3,1,2))
input_tf = tf.convert_to_tensor(input)

# tf outputs
output_tf = tf_conv(input_tf)
tf_params = tf_conv.variables
# torch_params = to_torch_params(tf_params)

# torch_conv.load_state_dict(torch_params)

output_torch = torch_conv(input_torch.type(torch.float32))
output_torch = torch.permute(output_torch,(0,2,3,1))
print(np.sum(np.abs(np.array(output_torch.detach()) - np.array(output_tf))))

