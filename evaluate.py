"""Test a policy from hanging starts: python evaluate.py runs/links_3/seed_0"""

import argparse
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import torch

from environment import EPISODE_STEPS, STATE, SUBSTEPS, Pendulums, observation_size
from model import PHYSICS_TIMESTEP
from ppo import Agent

STARTS = 50
SEED = 10_000
HOLD_STEPS = 500  # the last 5 s
FRAME_EVERY = 2


def successes(agent, links, starts=STARTS):
    """Which hanging starts end with every link held in the top for the last 5 s."""
    env = Pendulums(links, starts, SEED, top_starts=0)
    observations = env.observe()
    alive = np.ones(starts, dtype=bool)
    held = np.zeros(starts, dtype=int)
    for _ in range(EPISODE_STEPS):
        observations, _, terminated, _, info = env.step(agent.act(observations))
        alive &= ~terminated
        held = np.where(info["in_top"], held + 1, 0)
    env.close()
    return alive & (held >= HOLD_STEPS)


def record(agent, links, path):
    env = Pendulums(links, 1, SEED, top_starts=0)
    data = mujoco.MjData(env.model)
    fps = round(1 / (PHYSICS_TIMESTEP * SUBSTEPS * FRAME_EVERY))
    observations = env.observe()
    with (
        mujoco.Renderer(env.model, 480, 640) as renderer,
        mediapy.VideoWriter(path, (480, 640), fps=fps) as video,
    ):
        for step in range(EPISODE_STEPS):
            if step % FRAME_EVERY == 0:
                mujoco.mj_setState(env.model, data, env.state[0], STATE)
                mujoco.mj_forward(env.model, data)
                renderer.update_scene(data, camera="overview")
                video.add_image(renderer.render())
            observations, _, terminated, _, _ = env.step(agent.act(observations))
            if terminated[0]:
                break
    env.close()


def load(path):
    checkpoint = torch.load(path)
    agent = Agent(observation_size(checkpoint["links"]))
    agent.load_state_dict(checkpoint["agent"])
    return agent, checkpoint["links"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    folder = parser.parse_args().run
    agent, links = load(folder / "policy.pt")
    held = successes(agent, links)
    print(f"{held.sum()}/{len(held)} hanging starts held the top for the last 5 s")
    record(agent, links, folder / "video.mp4")


if __name__ == "__main__":
    main()
