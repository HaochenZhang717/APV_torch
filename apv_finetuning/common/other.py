import collections
import contextlib
import re
import time

import numpy as np
# import tensorflow as tf
# from tensorflow_probability import distributions as tfd

# from . import dists
from . import tfutils

from tensorflow import nest as tf_nest
import torch
from torch.distributions.independent import Independent
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class RandomAgent:
    def __init__(self, act_space, logprob=False):
        self.act_space = act_space["action"]
        self.logprob = logprob
        if hasattr(self.act_space, "n"):
            self._dist = Categorical(probs=torch.ones(self.act_space.n)/self.act_space.n)
        else:
            dist = Uniform(low=torch.from_numpy(self.act_space.low),
                           high=torch.from_numpy(self.act_space.high))
            self._dist = Independent(dist, 1)

    def __call__(self, obs, state=None, mode=None):
        action = self._dist.sample(torch.Size([len(obs["is_first"])]))
        output = {"action": action}
        if self.logprob:
            output["logprob"] = self._dist.log_prob(action)
        return output, None

def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in tf_nest.flatten(start)]
    indices = range(tf_nest.flatten(inputs)[0].size(0))
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


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step.type(torch.FloatTensor)
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, min=0, max=1)
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.climp(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r"horizon\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.size()) == len(value.size()), (reward.size(), value.size())#maybe error
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.size())))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = torch.permute(reward, tuple(dims))
        value = torch.permute(value, tuple(dims))
        pcont = torch.permute(pcont, tuple(dims))
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.concat((value[1:], bootstrap[None]), 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
        (inputs, pcont),
        bootstrap,
        reverse=True,
    )
    if axis != 0:
        returns = torch.permute(returns, tuple(dims))
    return returns


def action_noise(action, amount, act_space):
    if amount == 0:
        return action
    amount.type(action.dtype)
    if hasattr(act_space, "n"):
        probs = amount / action.shape[-1] + (1 - amount) * action
        return Categorical(probs=probs).sample()
    else:
        return torch.clamp(Normal(action, amount).sample(), -1, 1)


class StreamNorm(tfutils.Module):
    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        super().__init__()
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.mag = torch.ones(shape, dtype=torch.float32, requires_grad=False).cuda()

    def forward(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics["mean"] = inputs.mean()
        metrics["std"] = inputs.std()
        outputs = self.transform(inputs)
        metrics["normed_mean"] = outputs.mean()
        metrics["normed_std"] = outputs.std()
        return outputs, metrics

    def reset(self):
        self.mag = torch.ones_like(self.mag, dtype=torch.float32, requires_grad=False)

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self._shape)
        mag = torch.abs(batch).mean(0).type(torch.FloatTensor)
        self.mag = self._momentum * self.mag + (1 - self._momentum) * mag

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self._shape)
        values = values / self.mag.type(inputs.dtype)[None] + self._eps
        values = values * self._scale
        return values.reshape(inputs.shape)


class Timer:
    def __init__(self):
        self._indurs = collections.defaultdict(list)
        self._outdurs = collections.defaultdict(list)
        self._start_times = {}
        self._end_times = {}

    @contextlib.contextmanager
    def section(self, name):
        self.start(name)
        yield
        self.end(name)

    def wrap(self, function, name):
        def wrapped(*args, **kwargs):
            with self.section(name):
                return function(*args, **kwargs)

        return wrapped

    def start(self, name):
        now = time.time()
        self._start_times[name] = now
        if name in self._end_times:
            last = self._end_times[name]
            self._outdurs[name].append(now - last)

    def end(self, name):
        now = time.time()
        self._end_times[name] = now
        self._indurs[name].append(now - self._start_times[name])

    def result(self):
        metrics = {}
        for key in self._indurs:
            indurs = self._indurs[key]
            outdurs = self._outdurs[key]
            metrics[f"timer_count_{key}"] = len(indurs)
            metrics[f"timer_inside_{key}"] = np.sum(indurs)
            metrics[f"timer_outside_{key}"] = np.sum(outdurs)
            indurs.clear()
            outdurs.clear()
        return metrics


class CarryOverState:
    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out


class DecayLearningRate:
    def __init__(self, boundaries, values):
        boundaries.insert(0,0)
        self._boundaries = boundaries
        self._values = values
        self._step = 0
        assert len(self._boundaries) == len(self._values)

    def step(self):
        self._step += 1

    def __call__(self, *args, **kwargs):
        for bound, lr in zip(self._boundaries, self._values):
            if self._step >= bound:
                return lr


