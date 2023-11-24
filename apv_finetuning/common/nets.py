import re
from functools import partial

import numpy as np
import common

import torch
import torch.nn as nn
from torch.distributions.independent import Independent
from torch.distributions import Normal
from torchrl.modules import OneHotCategorical, TruncatedNormal
from tensorflow import nest as tf_nest
from collections import OrderedDict


def calculate_conv_out_dim(in_dim, kernel, padding=0, dilation=1, stride=1):
    C_in, H_in, W_in = in_dim
    H_out = np.floor((H_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    W_out = np.floor((W_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    return int(H_out), int(W_out)


def calculate_deconv_out_dim(in_dim, kernel, padding=0, dilation=1, stride=1):
    C_in, H_in, W_in = in_dim
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + padding + 1
    return int(H_out), int(W_out)


class EnsembleRSSM(nn.Module):
    def __init__(
            self,
            embed_dim,
            action_free=False,
            ensemble=5,
            stoch=30,
            deter=200,
            hidden=200,
            discrete=False,
            act="elu",
            norm="none",
            std_act="softplus",
            min_std=0.1,
            rnn_layers=1
    ):
        super().__init__()
        self._action_free = action_free
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._embed_dim = embed_dim
        self._std_act = std_act
        self._min_std = min_std
        self._rnn_layers = rnn_layers
        self._cells = [GRUCell(self._hidden, self._deter, norm=True).cuda() for _ in range(self._rnn_layers)]
        # self._cell = GRUCell(self._deter, norm=True)
        self._cast = lambda x: x.type(torch.FloatTensor).cuda() if torch.cuda.is_available() else lambda x: x.type(
            torch.FloatTensor)

        # build blocks for img in
        self._imag_in_modules = OrderedDict()
        self._imag_in_modules["img_in"] = nn.Linear(self._stoch * self._discrete + 50, self._hidden)
        if self._norm: self._imag_in_modules["img_in_norm"] = nn.LayerNorm(self._hidden)
        self._imag_in_modules["img_in_act"] = self._act()
        self._imag_in_modules = nn.Sequential(self._imag_in_modules)

        # prior out
        self._prior_out_modules = []
        for k in range(self._ensemble):
            module = OrderedDict()
            module["img_out" + str(k)] = nn.Linear(self._cells[0].state_size, self._hidden)
            if self._norm: module["img_out_norm" + str(k)] = nn.LayerNorm(self._hidden)
            module["img_out_act" + str(k)] = self._act()
            if self._discrete:
                module["img_out_dist" + str(k)] = nn.Linear(self._hidden, self._stoch * self._discrete)
            else:
                module["img_out_dist" + str(k)] = nn.Linear(self._hidden, 2 * self._stoch)
            self._prior_out_modules.append(nn.Sequential(module))
        self._prior_out_modules = torch.nn.ModuleList(self._prior_out_modules)

        # post out
        module = OrderedDict()
        module["obs_out"] = nn.Linear(self._cells[0].state_size + embed_dim, self._hidden)
        if self._norm: module["obs_out_norm"] = nn.LayerNorm(self._hidden)
        module["obs_out_act"] = self._act()
        if self._discrete:
            module["obs_dist"] = nn.Linear(self._hidden, self._stoch * self._discrete)
        else:
            module["obs_dist"] = nn.Linear(self._hidden, 2 * self._stoch)
        self._post_out_modules = torch.nn.Sequential(module)

    def initial(self, batch_size):
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]),
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]),
                std=torch.zeros([batch_size, self._stoch]),
                stoch=torch.zeros([batch_size, self._stoch]),
            )

        for i in range(self._rnn_layers):
            state[f"deter{i}"] = self._cells[i].get_initial_state(
                None, batch_size, torch.float32
            )

        if torch.cuda.is_available():
            for k, v in state.items():
                state[k] = v.cuda()

        return state

    # def zero_action(self, batch_size):
    #     act_zero = self._cast(torch.zeros((batch_size, 50)))
    #     if torch.cuda.is_available():
    #         act_zero = act_zero.cuda()
    #     return act_zero

    def fill_action_with_zero(self, action):
        # action: [B, action]
        B, D = action.size(0), action.size(1)
        if self._action_free:
            act_zero = self._cast(torch.zeros((B, 50)))
            # if torch.cuda.is_available(): act_zero = act_zero.cuda()
            return act_zero
        else:
            zeros = self._cast(torch.zeros((B, 50 - D)))
            act_zero = torch.concat((action, zeros), axis=1)
            # if torch.cuda.is_available(): act_zero = act_zero.cuda()
            return act_zero

    def forward(self, embed, action, is_first, state=None, sample=True):
        swap = lambda x: torch.permute(x, (1, 0) + tuple(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.size(0))
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs, sample=sample),
            (swap(action), swap(embed), swap(is_first)),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None, sample=True):
        swap = lambda x: torch.permute(x, (1, 0) + tuple(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.size(0))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(
            partial(self.img_step, sample=sample), action, state
        )
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = self._cast(state["stoch"])
        if self._discrete:
            shape = tuple(stoch.size()[:-2]) + tuple([self._stoch * self._discrete])
            stoch = torch.reshape(stoch, shape)
        return torch.concat((stoch, state[f"deter{self._rnn_layers - 1}"]), -1)

    def get_dist(self, state, ensemble=False):
        if ensemble:
            state = self._suff_stats_ensemble(state[f"deter{self._rnn_layers - 1}"])
        if self._discrete:
            logit = state["logit"]
            logit = self._cast(logit)
            dist = OneHotCategorical(logit)
            dist = torch.distributions.independent.Independent(dist, 1)
        else:
            mean, std = state["mean"], state["std"]
            mean = self._cast(mean)
            std = torch.diag(self._cast(std))
            dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, std)
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):

        (prev_state, prev_action) = tf_nest.map_structure(
            lambda x: torch.einsum("b,b...->b...", 1.0 - is_first.type(x.dtype), x),
            (prev_state, prev_action),
        )
        prior = self.img_step(prev_state, prev_action, sample)
        x = torch.concat((prior[f"deter{self._rnn_layers - 1}"], embed), -1)
        x = self._post_out_modules(x)

        stats = self._suff_stats_layer(x)
        dist = self.get_dist(stats)
        stoch = dist.rsample() if sample else dist.mode

        post = {"stoch": stoch, **stats}
        for i in range(self._rnn_layers):
            post[f"deter{i}"] = prior[f"deter{i}"]

        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = self._cast(prev_state["stoch"])
        prev_action = self._cast(prev_action)
        if self._discrete:
            shape = tuple(prev_stoch.size()[:-2]) + tuple([self._stoch * self._discrete])
            prev_stoch = torch.reshape(prev_stoch, shape)

        x = torch.concat((prev_stoch, self.fill_action_with_zero(prev_action)), -1)
        x = self._imag_in_modules(x)

        deters = []
        for i in range(self._rnn_layers):
            deter = prev_state[f"deter{i}"]
            x, deter = self._cells[i](x, [deter])
            deters.append(deter[0])

        stats = self._suff_stats_ensemble(x)
        index = torch.randint(low=0, high=self._ensemble, size=(), dtype=torch.int32)
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.rsample() if sample else dist.mode

        prior = {"stoch": stoch, **stats}
        for i in range(self._rnn_layers):
            prior[f"deter{i}"] = deters[i]

        return prior

    def _suff_stats_ensemble(self, inp):
        bs = list(inp.shape[:-1])
        inp = inp.reshape((-1, inp.size(-1)))
        stats = []
        for k in range(self._ensemble):
            x = self._prior_out_modules[k](inp)
            stats.append(self._suff_stats_layer(x))
        stats = {k: torch.stack([x[k] for x in stats], 0) for k, v in stats[0].items()}
        stats = {
            k: v.reshape(tuple([v.size(0)]) + tuple(bs) + tuple(v.size()[2:]))
            for k, v in stats.items()
        }
        return stats

    def _suff_stats_layer(self, x):
        if self._discrete:
            logit = torch.reshape(x, tuple(x.size()[:-1]) + tuple([self._stoch, self._discrete]))
            return {"logit": logit}
        else:
            mean, std = torch.split(x, 2, -1)
            std = {
                "softplus": lambda: torch.nn.functional.softplus(std),
                "sigmoid": lambda: torch.nn.functional.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.nn.functional.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        kld = torch.distributions.kl_divergence
        free = torch.tensor(free).cuda() if torch.cuda.is_available() else torch.tensor(free)

        def sg(x: dict):
            y = {}
            for k, v in x.items():
                y[k] = v.detach().clone()
            return y

        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = torch.maximum(value, free).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free)
                loss_rhs = torch.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = torch.maximum(value_lhs, free).mean()
                loss_rhs = torch.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class Encoder(nn.Module):
    def __init__(
            self,
            shapes,
            cnn_keys=r".*",
            mlp_keys=r".*",
            act="elu",
            norm="none",
            cnn_depth=48,
            cnn_kernels=(4, 4, 4, 4),
            mlp_layers=[400, 400, 400, 400],
    ):
        super().__init__()
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Encoder CNN inputs:", list(self.cnn_keys))
        print("Encoder MLP inputs:", list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

        if len(self.cnn_keys) > 0:
            # build cnn blocks
            self._cnn_modules = OrderedDict()
            channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
            input_depth = self._shapes['image'][2]  # a image has three channels
            input_dim = [0, 0, 0]
            input_dim[0] = self._shapes['image'][2]
            input_dim[1] = self._shapes['image'][0]
            input_dim[2] = self._shapes['image'][1]

            for i, kernel in enumerate(self._cnn_kernels):
                depth = 2 ** i * self._cnn_depth
                self._cnn_modules['conv' + str(i)] = nn.Conv2d(in_channels=input_depth,
                                                               out_channels=depth,
                                                               kernel_size=kernel,
                                                               stride=2)
                input_depth = depth
                H_out, W_out = calculate_conv_out_dim(input_dim, kernel=kernel, stride=2)
                input_dim = [depth, H_out, W_out]
                if norm != "none":
                    self._cnn_modules['convnorm' + str(i)] = nn.LayerNorm(input_dim)
                self._cnn_modules['act' + str(i)] = self._act()
            self._cnn_modules['flatten'] = nn.Flatten(start_dim=-3, end_dim=-1)
            self._cnn_modules = nn.Sequential(self._cnn_modules)

        if len(self.mlp_keys) > 0:
            # build MLP blocks
            self._mlp_modules = OrderedDict()
            shapes = {k: self._shapes[k] for k in self.mlp_keys}
            input_mlp = 0
            for _, v in shapes.items():
                input_mlp += sum(list(v))

            for i, width in enumerate(self._mlp_layers):
                self._mlp_modules["dense" + str(i)] = nn.Linear(input_mlp, width)
                if self._norm: self._mlp_modules["norm" + str(i)] = nn.LayerNorm(width)
                self._mlp_modules['act' + str(i)] = self._act()
                input_mlp = width
            self._mlp_modules = nn.Sequential(self._mlp_modules)

    @property
    def output_dim(self):
        return self._mlp_layers[-1] * (len(self.mlp_keys)!=0) + 2 * 2 * 384

    def forward(self, data):
        key, shape = list(self._shapes.items())[0]
        batch_dims = data[key].size()[: -len(shape)]  # torch.Size
        data = {
            k: torch.reshape(v, (-1,) + tuple(v.size()[len(batch_dims):]))
            for k, v in data.items()
        }
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = torch.concat(outputs, -1)
        return output.reshape(tuple(batch_dims + output.size()[1:]))

    def _cnn(self, data):
        x = torch.concat(tuple(data.values()), -1)
        x = self._cnn_modules(x)
        return x

    def _mlp(self, data):
        x = torch.concat(tuple(data.values()), -1)
        x = self._mlp_modules(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            shapes,
            cnn_keys=r".*",
            mlp_keys=r".*",
            act="elu",
            norm="none",
            input_size=2048,
            cnn_depth=48,
            cnn_kernels=(4, 4, 4, 4),
            mlp_layers=[400, 400, 400, 400],
    ):
        super().__init__()
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Decoder CNN outputs:", list(self.cnn_keys))
        print("Decoder MLP outputs:", list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

        if self.cnn_keys:
            # build cnn blocks
            self._cnn_modules = OrderedDict()
            channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
            ConvT = nn.ConvTranspose2d
            self._cnn_modules['convin'] = nn.Linear(input_size, 32 * self._cnn_depth)
            self._cnn_modules['unflatten_inpnut'] = nn.Unflatten(2, (32 * self._cnn_depth, 1, 1))
            self._cnn_modules['flatten_input'] = nn.Flatten(start_dim=0, end_dim=1)
            input_depth = 32 * self._cnn_depth
            in_dim = [1, 1, 32*self._cnn_depth]
            for i, kernel in enumerate(self._cnn_kernels):
                depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
                act, norm = self._act, self._norm
                if i == len(self._cnn_kernels) - 1:
                    depth, act, norm = sum(channels.values()), nn.Identity, "none"
                self._cnn_modules['conv' + str(i)] = ConvT(in_channels=input_depth,
                                                           out_channels=depth,
                                                           kernel_size=kernel,
                                                           stride=2)

                # def calculate_deconv_out_dim(in_dim, kernel, padding=0, dilation=1, stride=1):

                input_depth = depth
                H_out, W_out = calculate_deconv_out_dim(in_dim, kernel, stride=2)
                in_dim = [input_depth, H_out, W_out]
                if norm != "none": self._cnn_modules['convnorm' + str(i)] = nn.LayerNorm(in_dim)
                self._cnn_modules['act' + str(i)] = act()
                input_depth = depth
            self._cnn_modules = nn.Sequential(self._cnn_modules)

        if self.mlp_keys:
            # build MLP blocks
            self._mlp_modules = OrderedDict()
            self._mlp_out = OrderedDict()
            shapes = {k: self._shapes[k] for k in self.mlp_keys}
            input_mlp = input_size
            for i, width in enumerate(self._mlp_layers):
                self._mlp_modules["dense" + str(i)] = nn.Linear(input_mlp, width)
                if self._norm: self._mlp_modules["norm" + str(i)] = nn.LayerNorm(width)
                self._mlp_modules['act' + str(i)] = self._act()
                input_mlp = width
            self._mlp_modules = nn.Sequential(self._mlp_modules)

            for key, shape in shapes.items():
                assert len(shape)==1, "the tuple shape should only contain one element"
                self._mlp_out[key] = nn.Linear(width, shape[0])
            self._mlp_out = nn.ModuleDict(self._mlp_out)

    def forward(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        outputs = {}
        if self.cnn_keys:
            # outputs.update(self._cnn(features))
            cnn_out = self._cnn_modules(features)
            cnn_out = cnn_out.reshape(tuple(features.size()[:-1]) + tuple(cnn_out.size()[1:]))
            cnn_means = torch.split(cnn_out, list(channels.values()), -3)
            cnn_means = cnn_means[0]
            cnn_dists = {
                key: Independent(Normal(cnn_means, 1.0), 3)
                for (key, shape), mean in zip(channels.items(), cnn_means)
            }
            outputs.update(cnn_dists)

        if self.mlp_keys:
            shapes = {k: self._shapes[k] for k in self.mlp_keys}
            mlp_out = self._mlp_modules(features)
            mlp_dists = {}
            for k, v in self._mlp_out.items():
                mlp_mean = torch.reshape(v(mlp_out), tuple(features.size()[:-1]) + tuple(shapes[k]))
                # mlp_mean = self._mlp_out[k](mlp_out)
                mlp_dists[k] = Independent(Normal(mlp_mean, 1.0), len(shapes[k]))
            outputs.update(mlp_dists)

        return outputs



class MLP(nn.Module):
    def __init__(self, input_size, shape, layers, units, act="elu", norm="none", **out):
        '''
        :param input_size: dim of input feature
        :param shape: output dim
        :param layers: num of layers
        :param units: hidden units
        :param act: activation type
        :param norm: layernorm or not?
        :param out: output distribution type
        '''

        super().__init__()
        self._input_size = input_size
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

        self._hidden_modules = OrderedDict()
        in_size = input_size
        for index in range(self._layers):
            self._hidden_modules["dense" + str(index)] = nn.Linear(in_size, self._units)
            if self._norm:
                self._hidden_modules["norm" + str(index)] = nn.LayerNorm(self._units)
            self._hidden_modules['act' + str(index)] = self._act()
            in_size = self._units
        self._hidden_modules = nn.Sequential(self._hidden_modules)

        if self._out['dist'] == "mse":
            self._output_mean = nn.Linear(self._units, int(np.prod(self._shape)))
            self._output_std = None
        elif self._out['dist'] in ("normal", "tanh_normal", "trunc_normal"):
            self._output_mean = nn.Linear(self._units, int(np.prod(self._shape)))
            self._output_std = nn.Linear(self._units, int(np.prod(self._shape)))

    def forward(self, features):
        x = features.reshape((-1, features.size(-1)))
        x = self._hidden_modules(x)
        mean = self._output_mean(x)
        mean = mean.reshape(tuple(features.size()[:-1]) + tuple(self._shape))
        if self._output_std is not None:
            std = self._output_std(x)
            std = std.reshape(features.size()[:-1] + tuple(self._shape))

        if self._out['dist'] == "normal":
            dist = Normal(loc=mean, scale=std)
            dist = Independent(dist, len(self._shape))
        elif self._out['dist'] == "mse":
            dist = Normal(loc=mean, scale=1.0)
            dist = Independent(dist, len(self._shape))
        elif self._out['dist'] == 'trunc_normal':
            mean = torch.tanh(mean)
            std = 2*torch.sigmoid(std/2)+0.1
            dist = TruncatedNormal(loc=mean, scale=std, tanh_loc=False, min=-1, max=1)
        else:
            raise NotImplementedError(self._dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, input_size, size, norm=False, act="tanh", update_bias=-1, *args, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)()
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(self.state_size + input_size, 3 * size)
        if norm:
            self._norm = nn.LayerNorm(3 * size)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        if isinstance(state, list):
            state = state[0]
        parts = self._layer(torch.concat((inputs, state), -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size, self._size, self._size], dim=-1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = inputs.size()[0]
            dtype = inputs.dtype

        if batch_size is None or dtype is None:
            raise ValueError(
                "batch_size and dtype cannot be None while constructing initial "
                f"state. Received: batch_size={batch_size}, dtype={dtype}"
            )

        def create_zeros(unnested_state_size):
            flat_dims = tuple([unnested_state_size])
            init_state_size = tuple([batch_size]) + flat_dims
            init_state = torch.zeros(init_state_size, dtype=dtype)
            if torch.cuda.is_available(): init_state = init_state.cuda()
            return init_state

        return create_zeros(self.state_size)


def get_act(name):
    if name == "none":
        return nn.Identity
    # elif name == "mish":
    #     return lambda x: x * nn.functional.tanh(nn.functional.softplus(x))
    elif name == 'tanh':
        return nn.Tanh
    elif name == "elu":
        return nn.ELU
    else:
        raise NotImplementedError(name)
