import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import numpy as np


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
from torch.utils.tensorboard import SummaryWriter



# np.random.seed(0)

def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    parsed, remaining = common.Flags(configs=["defaults"]).parse(known_only=True)
    config = common.Config(configs["defaults"])
    # for name in parsed.configs:
    #     config = config.update(configs[name])
    config = config.update(configs['metaworld_pretrain'])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    load_logdir = pathlib.Path(config.load_logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)
    print("Loading Logdir", load_logdir)

    message = "No GPU found. To actually train on CPU remove this assert."
    assert torch.cuda.is_available(), message


    train_replay = common.ReplayWithoutAction(
        logdir / "train_episodes",
        load_directory=load_logdir / "train_episodes",
        **config.replay
    )

    step = common.Counter(train_replay.stats["total_steps"])
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_log = common.Every(config.log_every)
    should_save = common.Every(config.eval_every)

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

    print("Create envs.")
    env = make_env("train")
    act_space, obs_space = env.act_space, env.obs_space

    print("Create agent.")

    train_dataset = common.get_one_batch(train_replay, **config.dataset)
    report_dataset = common.get_one_batch(train_replay, **config.dataset)
    # tb_writer = SummaryWriter('runs/experiment1')
    # datum = train_dataset()
    # datum = list(datum.values())
    agnt = agent.Agent(config, obs_space, act_space, step)

    # tb_writer.add_graph(agnt.wm, [datum])


    train_agent = common.CarryOverState(agnt.train)
    train_agent(train_dataset())

    if (logdir / "variables.pkl").exists():
        agnt.load(logdir / "variables.pkl")

    print("Train a video prediction model.")
    # for step in tqdm(range(step, config.steps), total=config.steps, initial=step):
    for _ in range(int(step.value), int(config.steps)):
        mets = train_agent(train_dataset())
        [metrics[key].append(value) for key, value in mets.items()]
        step.increment()

        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(report_dataset()), prefix="train")
            logger.write(fps=True)

        if should_save(step):
            agnt.wm.save(logdir)
            # agnt.save(logdir / "variables.pkl")
            # agnt.wm.rssm.save(logdir / "rssm_variables.pkl", verbose=False)
            # agnt.wm.encoder.save(logdir / "encoder_variables.pkl", verbose=False)
            # agnt.wm.heads["decoder"].save(
            #     logdir / "decoder_variables.pkl", verbose=False
            # )

    env.close()
    agnt.wm.save(logdir)
    # tb_writer.close()
    # agnt.save(logdir / "variables.pkl")
    # agnt.wm.rssm.save(logdir / "rssm_variables.pkl", verbose=False)
    # agnt.wm.encoder.save(logdir / "encoder_variables.pkl", verbose=False)
    # agnt.wm.heads["decoder"].save(logdir / "decoder_variables.pkl", verbose=False)


if __name__ == "__main__":
    main()


