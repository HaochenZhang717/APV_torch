import collections
import contextlib
import re
import time

import numpy as np
import torch
# import tensorflow as tf
# from tensorflow_probability import distributions as tfd
import tensorflow.nest as tf_nest

# from . import dists
# from . import tfutils


def static_scan(fn, inputs, start, reverse=False):
    # inputs = {embed, is_first}
    # start = {state, state}
    last = start
    outputs = [[] for _ in tf_nest.flatten(start)]
    indices = range(tf_nest.flatten(inputs)[0].shape[0])
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf_nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, tf_nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [torch.stack(x, 0) for x in outputs]
    return tf_nest.pack_sequence_as(start, outputs)


class CarryOverState:
    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out
