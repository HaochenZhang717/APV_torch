import common
import torch
import torch.nn as nn
from common import Optimizer
from torch.utils.tensorboard import SummaryWriter

class Agent(object):
    def __init__(self, config, obs_space, act_space, step):
        # super().__init__()
        self.config = config
        # self._device = config['defaults']['device']
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.tfstep = torch.tensor(int(self.step), dtype=torch.int64)
        self .wm = WorldModel(config, obs_space, self.tfstep)
        # self.optimizer = Optimizer(self.wm, "wm_optimizer", **config.model_opt)

    def train(self, data, state=None):
        metrics = {}
        # tensor_to_value = lambda x: x.detach().cpu().numpy().item()
        def sg(x:dict):
            y = {}
            for k, v in x.items():
                y[k] = v.detach().clone()
            return y

        state, outputs, mets = self.wm.train(data, state)
        metrics.update(mets)
        start = outputs["post"]
        return sg(state), metrics

    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads["decoder"].cnn_keys:
            name = key.replace("/", "_")
            report[f"openl_{name}"] = (self.wm.video_pred(data, key)).detach().cpu().numpy()
        return report



class WorldModel(object):
    def __init__(self, config, obs_space, tfstep):
        super().__init__()
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        # self._device = config['defaults']['device']
        self.tfstep = tfstep
        self.encoder = common.Encoder(shapes, **config.encoder).cuda(1)
        self.rssm = common.EnsembleRSSM(self.encoder.output_dim, **config.rssm).cuda(1)
        ####continue debugging
        self.heads = {}
        self.heads["decoder"] = common.Decoder(shapes, **config.decoder).cuda(1)

        modules_params = list(self.encoder.parameters()) \
                         + list(self.rssm.parameters()) \
                         + list(self.heads["decoder"].parameters())

        self.model_opt = common.Optimizer(modules_params, "model", **config.model_opt)

    def train(self, data, state):
        model_loss, state, outputs, metrics = self.loss(data, state)
        metrics.update(self.model_opt.step(model_loss))
        return state, outputs, metrics

    def loss(self, data, state=None):
        tensor_to_value = lambda x: x.detach().cpu().numpy().item()
        # import numpy as np
        def to_tensor(x: dict):
            y = {}
            for k, v in x.items():
                y[k] = torch.tensor(v, dtype=torch.float32)
            return y
        # data = np.load('/home/haochen/apv_remote/apv_pretraining/one_batch_from_tf.npy')


        # import pickle
        # with open('/home/hchen/apv_remote/apv_pretraining/encoder_params.pickle', 'rb') as handle:
        #     encoder_params_tf = pickle.load(handle)

        # with open('/home/hchen/apv_remote/apv_pretraining/decoder_params.pickle', 'rb') as handle:
        #     decoder_params_tf = pickle.load(handle)

        # with open('/home/hchen/apv_remote/apv_pretraining/rssm_params.pickle', 'rb') as handle:
        #     rssm_params_tf = pickle.load(handle)

        # self.encoder.load_state_dict(encoder_params_tf)
        # self.rssm.load_state_dict(rssm_params_tf)
        # self.heads['decoder'].load_state_dict(decoder_params_tf)

        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm(embed, data["is_first"], state)

        # import pickle
        # with open('/home/hchen/apv_remote/apv_pretraining/post.pickle', 'rb') as handle:
        #     post = pickle.load(handle)
        # with open('/home/hchen/apv_remote/apv_pretraining/prior.pickle', 'rb') as handle:
        #     prior = pickle.load(handle)
        # import numpy as np
        # post_np = np.load('/home/hchen/apv_remote/apv_pretraining/post.npy', allow_pickle=True).item()
        # prior_np = np.load('/home/hchen/apv_remote/apv_pretraining/prior.npy', allow_pickle=True).item()

        # post = to_tensor(post_np)
        # prior = to_tensor(prior_np)

        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = name in self.config.grad_heads
            inp = feat if grad_head else feat.detach().clone()
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = dist.log_prob(data[key]).type(torch.float32)
                likes[key] = like
                losses[key] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f"{name}_loss": tensor_to_value(value) for name, value in losses.items()}
        metrics["model_kl"] = tensor_to_value(kl_value.mean())
        metrics["prior_ent"] = tensor_to_value(self.rssm.get_dist(prior).entropy().mean())
        metrics["post_ent"] = tensor_to_value(self.rssm.get_dist(post).entropy().mean())
        last_state = {k: v[:, -1] for k, v in post.items()}
        # for k, v in metrics.items():
        #     print(k)
        #     print(v)
        #     print('~~~~~~~~~~~~~~~~~~~~~')

        return model_loss, last_state, outs, metrics

    def preprocess(self, obs):
        dtype = torch.float32
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == torch.int32:
                value = value.type(dtype)
            if value.dtype == torch.uint8:
                value = value.type(dtype) / 255.0 - 0.5
            obs[key] = value
        return obs

    def video_pred(self, data, key):
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm(embed[:6, :5], data["is_first"][:6, :5])
        recon = decoder(self.rssm.get_feat(states))[key].mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["is_first"][:6, 5:], init)
        openl = decoder(self.rssm.get_feat(prior))[key].mode
        model = torch.concat((recon[:, :5] + 0.5, openl + 0.5), 1)
        error = (model - truth + 1) / 2
        video = torch.concat((truth, model, error), 3)
        video = torch.permute(video, (0, 1, 3, 4, 2))
        B, T, H, W, C = video.size()
        return torch.permute(video, (1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

    def save(self, logdir):
        torch.save({"rssm_variables": self.rssm.state_dict()}, logdir / "rssm_variables.pt")
        torch.save({"encoder_variables": self.encoder.state_dict()}, logdir / "encoder_variables.pt")
        torch.save({"decoder_variables": self.heads["decoder"].state_dict()}, logdir / "decoder_variables.pt")

