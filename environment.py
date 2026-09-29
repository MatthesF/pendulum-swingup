"""Many cart pendulums stepped together by MuJoCo's threaded rollout."""

import mujoco
import numpy as np
from mujoco import rollout

from model import CART_LIMIT, GRAVITY, LINK_LENGTH, MAX_CART_FORCE, build_model
from reward import RAIL_COST, TOP_ANGLE, reward

SUBSTEPS = 5  # 100 Hz control
EPISODE_STEPS = 1500  # 15 s
RAIL_END = 2.3
TRAVEL_BUDGET = 4 * np.pi  # per link, counted along every swing back and forth
TOP_STARTS = 0.25
HANGING_TILT = np.deg2rad(5)
HANGING_BEND = np.deg2rad(1)
# Top starts must stay catchable with 10 N; at three links that is only a few degrees.
TOP_TILT = np.deg2rad(2)
TOP_SPIN = 0.2
LINK_RATE = np.sqrt(GRAVITY / LINK_LENGTH)
CART_RATE = np.sqrt(GRAVITY * LINK_LENGTH)
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS


def observation_size(links):
    return 2 + 4 * links


class Pendulums:
    def __init__(self, links, count, seed, top_starts=TOP_STARTS, threads=None):
        self.model = build_model(links)
        self.links, self.count, self.top_starts = links, count, top_starts
        self.rng = np.random.default_rng(seed)
        self.rollout = rollout.Rollout(nthread=threads)
        self.datas = [mujoco.MjData(self.model) for _ in range(threads or 1)]
        # The state is time, cart position, hinge angles, cart speed, hinge speeds.
        self.state = np.zeros((count, mujoco.mj_stateSize(self.model, STATE)))
        self.hinges = slice(2, 2 + links)
        self.spins = slice(3 + links, 3 + 2 * links)
        self.travel = np.zeros((count, links))
        self.steps = np.zeros(count, dtype=int)
        self.reset(np.ones(count, dtype=bool))

    def angles(self, state):
        """World angles from upright, the sum of the hinges below each link."""
        return np.cumsum(state[..., self.hinges], axis=-1)

    def speeds(self, state):
        return np.cumsum(state[..., self.spins], axis=-1)

    def cart(self):
        return self.state[:, 1], self.state[:, 2 + self.links]

    def in_top(self):
        return np.all(np.cos(self.angles(self.state)) >= np.cos(TOP_ANGLE), axis=1)

    def observe(self):
        position, speed = self.cart()
        angles = self.angles(self.state)
        return np.column_stack(
            (
                position / CART_LIMIT,
                speed / CART_RATE,
                np.cos(angles),
                np.sin(angles),
                self.speeds(self.state) / LINK_RATE,
                np.maximum(0, 1 - self.travel / TRAVEL_BUDGET),
            )
        ).astype(np.float32)

    def reset(self, which):
        count, links = int(which.sum()), self.links
        up = self.rng.random((count, 1)) < self.top_starts
        hanging = np.pi + self.rng.uniform(-HANGING_TILT, HANGING_TILT, (count, 1))
        hanging = hanging + self.rng.uniform(
            -HANGING_BEND, HANGING_BEND, (count, links)
        )
        top = self.rng.uniform(-TOP_TILT, TOP_TILT, (count, links))
        spin = np.where(
            up,
            self.rng.uniform(-TOP_SPIN, TOP_SPIN, (count, links)),
            self.rng.normal(0, 0.01, (count, links)),
        )
        state = np.zeros((count, self.state.shape[1]))
        state[:, 1] = self.rng.normal(0, 0.01, count)
        state[:, self.hinges] = np.diff(np.where(up, top, hanging), prepend=0)
        state[:, 2 + links] = self.rng.normal(0, 0.01, count)
        state[:, self.spins] = np.diff(spin, prepend=0)
        self.state[which] = state
        self.travel[which] = 0
        self.steps[which] = 0

    def step(self, actions):
        force = MAX_CART_FORCE * np.clip(actions, -1, 1).astype(np.float64)
        path, _ = self.rollout.rollout(
            self.model, self.datas, self.state, force[:, None, :], nstep=SUBSTEPS
        )
        angles = self.angles(np.concatenate((self.state[:, None], path), axis=1))
        self.travel += np.abs(np.diff(angles, axis=1)).sum(axis=1)
        self.state = path[:, -1].copy()
        self.steps += 1

        position, speed = self.cart()
        rail = np.abs(position) >= RAIL_END
        terminated = rail | np.any(self.travel >= TRAVEL_BUDGET, axis=1)
        truncated = ~terminated & (self.steps >= EPISODE_STEPS)
        rewards = reward(
            self.angles(self.state), self.speeds(self.state), position, speed
        )
        rewards -= RAIL_COST * rail
        info = {"final_observation": self.observe(), "in_top": self.in_top()}
        done = terminated | truncated
        if done.any():
            self.reset(done)
        return self.observe(), rewards, terminated, truncated, info

    def close(self):
        self.rollout.close()
