import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import torch

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common
import torch


def main():
    print(f"allocated memory: {torch.cuda.memory_allocated(0)}")

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    parsed, remaining = common.Flags(configs=["defaults"]).parse(known_only=True)
    config = common.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    config.save(logdir / "config.yaml")
    print(config, "\n")

    print("Logdir:", logdir)

    if config.load_logdir != "none":
        load_logdir = pathlib.Path(config.load_logdir).expanduser()
        print("Loading Logdir", load_logdir)

    message = "No GPU found. To actually train on CPU remove this assert."
    assert torch.cuda.is_available(), message


    train_replay = common.Replay(logdir / "train_episodes", **config.replay)
    eval_replay = common.Replay(
        logdir / "eval_episodes",
        **dict(
            capacity=config.replay.capacity // 10,
            minlen=config.dataset.length,
            maxlen=config.dataset.length,
        ),
    )
    step = common.Counter(train_replay.stats["total_steps"]) # common.Counter is a total_ordering class

    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)
    should_expl = common.Until(config.expl_until)

    def make_env(mode):
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            env = common.DMC(
                task, config.action_repeat, config.render_size, config.dmc_camera
            )
            env = common.NormalizeAction(env)
        elif suite == "atari":
            env = common.Atari(
                task, config.action_repeat, config.render_size, config.atari_grayscale
            )
            env = common.OneHotAction(env)
        elif suite == "crafter":
            assert config.action_repeat == 1
            outdir = logdir / "crafter" if mode == "train" else None
            reward = bool(["noreward", "reward"].index(task)) or mode == "eval"
            env = common.Crafter(outdir, reward)
            env = common.OneHotAction(env)
        elif suite == "metaworld":
            task = "-".join(task.split("_"))
            env = common.MetaWorld(
                task,
                config.seed,
                config.action_repeat,
                config.render_size,
                config.camera,
            )
            env = common.NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = common.TimeLimit(env, config.time_limit)
        return env

    def per_episode(ep, mode):
    # mode can be 'train' or 'eval'

        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        if "metaworld" in config.task:
            success = float(np.sum(ep["success"]) >= 1.0)
            print(
                f"{mode.title()} episode has {float(success)} success, {length} steps and return {score:.1f}."
            )
            logger.scalar(f"{mode}_success", float(success))
        else:
            print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        # do not understand why do we have this
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        should = {"train": should_video_train, "eval": should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f"{mode}_policy_{key}", ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()
    #here is the end of function per_episode

    print("Create envs.")
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == "none": # will run this line
        train_envs = [make_env("train") for _ in range(config.envs)]
        eval_envs = [make_env("eval") for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode: common.Async(
            functools.partial(make_env, mode), config.envs_parallel
        )
        train_envs = [make_async_env("train") for _ in range(config.envs)]
        eval_envs = [make_async_env("eval") for _ in range(eval_envs)]
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    train_driver = common.Driver(train_envs)
    train_driver.on_episode(lambda ep: per_episode(ep, mode="train"))
    train_driver.on_step(lambda tran, worker: step.increment()) # to let step +1
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)

    eval_driver = common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode="eval"))

    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    if prefill:
        print(f"Prefill dataset ({prefill} steps).")
        random_agent = common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    print("Create agent.")

    train_dataset = common.get_one_batch(train_replay, **config.dataset)
    report_dataset = common.get_one_batch(train_replay, **config.dataset)

    agnt = agent.Agent(config, obs_space, act_space, step)
    train_agent = common.CarryOverState(agnt.train)

    train_agent(train_dataset()) #I annotate it for faster debug


    if (logdir / "variables.pkl").exists():
        agnt.load(logdir / "variables.pkl")
    else:
        if config.load_logdir != "none":
            if "af_rssm" in config.load_modules:
                agnt.wm.af_rssm.load(load_logdir / "rssm_variables.pkl")
            if "encoder" in config.load_modules:
                agnt.wm.encoder.load(load_logdir / "encoder_variables.pkl")
            if "decoder" in config.load_modules:
                agnt.wm.heads["decoder"].load(load_logdir / "decoder_variables.pkl")
        print("Pretrain agent.")

        for _ in range(config.pretrain):
            train_agent(train_dataset())  # I annotate it for faster debug


    train_policy = lambda *args: agnt.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )
    eval_policy = lambda *args: agnt.policy(*args, mode="eval")

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                # mets = train_agent(next(train_dataset))
                mets = train_agent(train_dataset())
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(report_dataset()), prefix="train")
            logger.write(fps=True)

    train_driver.on_step(train_step)

    while step < config.steps:
        logger.write()
        print("Start evaluation.")
        # logger.add(agnt.report(next(eval_dataset)), prefix="eval")
        eval_driver(eval_policy, episodes=config.eval_eps)
        print("Start training.")
        train_driver(train_policy, steps=config.eval_every)
        print(f"allocated memory: {torch.cuda.memory_allocated(0)}")
        agnt.save_all(logdir)
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    agnt.save_all(logdir)


if __name__ == "__main__":
    main()
