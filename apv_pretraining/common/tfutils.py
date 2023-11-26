import pathlib
import pickle
import re

import numpy as np
# import tensorflow as tf
# from tensorflow.keras import mixed_precision as prec
import torch

# try:
#     from tensorflow.python.distribute import values
# except Exception:
#     from google3.third_party.tensorflow.python.distribute import values

# tf.tensor = tf.convert_to_tensor
# for base in (tf.Tensor, tf.Variable, values.PerReplica):
#     base.mean = tf.math.reduce_mean
#     base.std = tf.math.reduce_std
#     base.var = tf.math.reduce_variance
#     base.sum = tf.math.reduce_sum
#     base.any = tf.math.reduce_any
#     base.all = tf.math.reduce_all
#     base.min = tf.math.reduce_min
#     base.max = tf.math.reduce_max
#     base.abs = tf.math.abs
#     base.logsumexp = tf.math.reduce_logsumexp
#     base.transpose = tf.transpose
#     base.reshape = tf.reshape
#     base.astype = tf.cast


# values.PerReplica.dtype = property(lambda self: self.values[0].dtype)

# tf.TensorHandle.__repr__ = lambda x: '<tensor>'
# tf.TensorHandle.__str__ = lambda x: '<tensor>'
# np.set_printoptions(threshold=5, edgeitems=0)


# class Module(tf.Module):
#     def save(self, filename, verbose=True):
#         values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
#         amount = len(tf.nest.flatten(values))
#         count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
#         if verbose:
#             print(f"Save checkpoint with {amount} tensors and {count} parameters.")
#         with pathlib.Path(filename).open("wb") as f:
#             pickle.dump(values, f)
#
#     def load(self, filename, verbose=True):
#         with pathlib.Path(filename).open("rb") as f:
#             values = pickle.load(f)
#         amount = len(tf.nest.flatten(values))
#         count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
#         if verbose:
#             print(f"Load checkpoint with {amount} tensors and {count} parameters.")
#         tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)
#
#     def get(self, name, ctor, *args, **kwargs):
#         # Create or get layer by name to avoid mentioning it in the constructor.
#         if not hasattr(self, "_modules"):
#             self._modules = {}
#         if name not in self._modules:
#             self._modules[name] = ctor(*args, **kwargs)
#         return self._modules[name]

class Optimizer(object):
    def __init__(
            self, model_params, name, lr, eps=1e-4, clip=None, wd=None, opt="adam", wd_pattern=r".*"
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._model_params = model_params
        self._name = name
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._once = True

        if opt == "adam":
            self._opt = torch.optim.Adam(self._model_params, lr, eps=eps, weight_decay=wd)
            # self._opt = torch.optim.Adam(model_params, lr, eps=eps, weight_decay=wd)
            # if not isinstance(self._model, list):
            #     self._opt = torch.optim.Adam(self._model.parameters(), lr, eps=eps, weight_decay=wd)
            # else:
            #     train_para = []
            #     for model in self._model:
            #         train_para += list(model.parameters())
            #     self._opt = torch.optim.Adam(train_para, lr, eps=eps, weight_decay=wd)
        else:
            raise NotImplementedError(opt)

    def step(self, loss, retain_graph=False):
        metrics = {}
        tensor_to_value = lambda x: x.detach().cpu().numpy().item()

        # if self._once:
        #     if not isinstance(self._model, list):
        #         count = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        #     else:
        #         count = 0
        #         for model in self._model:
        #             count += sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     print(f"Found {count} {self._name} parameters.")
        #     self._once = False

        metrics[f"{self._name}_loss"] = tensor_to_value(loss)

        # gradient backward and gradient clipping
        self._opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if self._clip is not None:
            torch.nn.utils.clip_grad_norm_(self._model_params, self._clip)
        self._opt.step()
        return metrics





