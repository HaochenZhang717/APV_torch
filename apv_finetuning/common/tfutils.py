import pathlib
import pickle
import re
import torch


import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

try:
    from tensorflow.python.distribute import values
except Exception:
    from google3.third_party.tensorflow.python.distribute import values

tf.tensor = tf.convert_to_tensor
for base in (tf.Tensor, tf.Variable, values.PerReplica):
    base.mean = tf.math.reduce_mean
    base.std = tf.math.reduce_std
    base.var = tf.math.reduce_variance
    base.sum = tf.math.reduce_sum
    base.any = tf.math.reduce_any
    base.all = tf.math.reduce_all
    base.min = tf.math.reduce_min
    base.max = tf.math.reduce_max
    base.abs = tf.math.abs
    base.logsumexp = tf.math.reduce_logsumexp
    base.transpose = tf.transpose
    base.reshape = tf.reshape
    base.astype = tf.cast


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

class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    ##save and load should be revised
    def save(self, filename, verbose=True):
        values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
        amount = len(tf.nest.flatten(values))
        count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
        if verbose:
            print(f"Save checkpoint with {amount} tensors and {count} parameters.")
        with pathlib.Path(filename).open("wb") as f:
            pickle.dump(values, f)

    def load(self, filename, verbose=True):
        with pathlib.Path(filename).open("rb") as f:
            values = pickle.load(f)
        amount = len(tf.nest.flatten(values))
        count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
        if verbose:
            print(f"Load checkpoint with {amount} tensors and {count} parameters.")
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

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
        # loss.backward(retain_graph=retain_graph)
        loss.backward()
        if self._clip is not None:
            torch.nn.utils.clip_grad_norm_(self._model_params, self._clip)
        self._opt.step()
        return metrics

# class Optimizer(object):
#     def __init__(
#         self, name, lr, eps=1e-4, clip=None, wd=None, opt="adam", wd_pattern=r".*"
#     ):
#         assert 0 <= wd < 1
#         assert not clip or 1 <= clip
#         self._name = name
#         self._clip = clip
#         self._wd = wd
#         self._wd_pattern = wd_pattern
#         self._opt = {
#             "adam": lambda: torch.optim.Adam(lr=lr, eps=eps, weight_decay=wd),
#             "nadam": lambda: torch.optim.Nadam(lr, epsilon=eps),
#             "adamax": lambda: tf.optimizers.Adamax(lr, epsilon=eps),
#             "sgd": lambda: tf.optimizers.SGD(lr),
#             "momentum": lambda: tf.optimizers.SGD(lr, 0.9),
#         }[opt]()
#         self._mixed = prec.global_policy().compute_dtype == tf.float16
#         if self._mixed:
#             self._opt = prec.LossScaleOptimizer(self._opt, dynamic=True)
#         self._once = True
#
#     @property
#     def variables(self):
#         return self._opt.variables()
#
#     def __call__(self, tape, loss, modules):
#         assert loss.dtype is tf.float32, (self._name, loss.dtype)
#         assert len(loss.shape) == 0, (self._name, loss.shape)
#         metrics = {}
#
#         # Find variables.
#         modules = modules if hasattr(modules, "__len__") else (modules,)
#         varibs = tf.nest.flatten([module.variables for module in modules])
#         count = sum(np.prod(x.shape) for x in varibs)
#         if self._once:
#             print(f"Found {count} {self._name} parameters.")
#             self._once = False
#
#         # Check loss.
#         tf.debugging.check_numerics(loss, self._name + "_loss")
#         metrics[f"{self._name}_loss"] = loss
#
#         # Compute scaled gradient.
#         if self._mixed:
#             with tape:
#                 loss = self._opt.get_scaled_loss(loss)
#         grads = tape.gradient(loss, varibs)
#         if self._mixed:
#             grads = self._opt.get_unscaled_gradients(grads)
#         if self._mixed:
#             metrics[f"{self._name}_loss_scale"] = self._opt.loss_scale
#
#         # Distributed sync.
#         context = tf.distribute.get_replica_context()
#         if context:
#             grads = context.all_reduce("mean", grads)
#
#         # Gradient clipping.
#         norm = tf.linalg.global_norm(grads)
#         if not self._mixed:
#             tf.debugging.check_numerics(norm, self._name + "_norm")
#         if self._clip:
#             grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
#         metrics[f"{self._name}_grad_norm"] = norm
#
#         # Weight decay.
#         if self._wd:
#             self._apply_weight_decay(varibs)
#
#         # Apply gradients.
#         self._opt.apply_gradients(
#             zip(grads, varibs), experimental_aggregate_gradients=False
#         )
#
#         return metrics
#
#     def _apply_weight_decay(self, varibs):
#         nontrivial = self._wd_pattern != r".*"
#         if nontrivial:
#             print("Applied weight decay to variables:")
#         for var in varibs:
#             if re.search(self._wd_pattern, self._name + "/" + var.name):
#                 if nontrivial:
#                     print("- " + self._name + "/" + var.name)
#                 var.assign((1 - self._wd) * var)

