import numpy as np
import torch


class Driver:
    def __init__(self, envs, **kwargs):
        self._envs = envs
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            # we have one worker, so obs is a dict like {0: obs}
            obs = {
                i: self._envs[i].reset()
                for i, ob in enumerate(self._obs)
                if ob is None or ob["is_last"]
            }
            # if len(obs) != 1:
                # raise NotImplementedError
            # for i in range(len(obs)):
                # obs[i]['image'] = np.transpose(obs[i]['image'], (2, 0, 1))
                # for k, v in obs[i].items():
                #         obs[i][k] = torch.tensor(v)

            for i, ob in obs.items(): #obs=={0: obs}
            # ob is a dictionary with 8 keys
            # 'reward', 'is_first', 'is_last', 'is_terminal', 'image', 'orientations', 'height', 'velocity'
                self._obs[i] = ob() if callable(ob) else ob
                act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}# act is a dict with key 'action'
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
                self._eps[i] = [tran]
            obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
            actions, self._state = policy(obs, self._state, **self._kwargs)

            actions = [
                {k: np.array(actions[k][i].detach().cpu()) for k in actions}
                for i in range(len(self._envs))
            ]
            assert len(actions) == len(self._envs)
            obs = [e.step(a) for e, a in zip(self._envs, actions)]
            obs = [ob() if callable(ob) else ob for ob in obs]
            for i, (act, ob) in enumerate(zip(actions, obs)):
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}

                # self._on_steps has three elements in total, the first one is couter, the second one is to append new
                # transition to replay buffer, the third is to train agent.

                # I wrote it for debuging
                # [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]# append one step to self._ongoing_eps
                for fn in self._on_steps:
                    fn(tran, worker=i, **self._kwargs)

                self._eps[i].append(tran)
                step += 1
                if ob["is_last"]:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
            self._obs = obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
