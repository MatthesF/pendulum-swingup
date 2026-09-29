# pendulum-swingup

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-PPO-ee4c2c.svg)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.11-1f6feb.svg)](https://mujoco.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

PPO learns to swing up and balance one, two or three pendulum links on a cart, from
scratch on a laptop CPU.

| 1 link | 2 links | 3 links |
| :-: | :-: | :-: |
| ![One link swinging up and balancing](media/links_1.gif) | ![Two links swinging up and balancing](media/links_2.gif) | ![Three links swinging up and balancing](media/links_3.gif) |

## Results

Each policy is tested from 50 hanging starts. A start counts if every link stays within
10° of upright for the last 5 of 15 seconds. A seed solves the task if it holds at
least 45 of the 50.

| Links | Training steps | Time per seed | Seeds that solve it |
| --- | --- | --- | --- |
| 1 | 2M | 3 min | 8 of 8 |
| 2 | 8M | 11 min | 7 of 8 |
| 3 | 20M | 35 min | 8 of 8 |

Times are on an M2 Max with eight seeds training at once, one core each.

The failing two-link seed whips the outer link over in a full loop on the way up. It
reaches the top with its rotation budget almost spent, so it never gets to practise
the catch.

Three links became reliable when the policy and value gradients were clipped
separately. CleanRL clips them together, but it also normalises rewards. Here the
values approach 1000, and the value gradient is often a hundred times the policy's,
so a joint clip shrinks each policy step by however wrong the value happens to be.
Some seeds learned to balance and then lost it; others were slow to learn it at all.
Over the same eight seeds:

| Gradient clipping | Seeds that solve 3 links | Tests after 10M steps that hold at least 45 |
| --- | --- | --- |
| Together | 5 of 8 | 37% |
| Separately | 8 of 8 | 91% |

## Run

```bash
pip install -e .
python train.py --links 3 --seed 0
python evaluate.py runs/links_3/seed_0
```

Training writes the policy, a progress plot and a video to `runs/links_3/seed_0/`.
Seeds use one core each, so train several at once:

```bash
for seed in 0 1 2 3 4 5 6 7; do python train.py --links 3 --seed $seed & done; wait
```

## How it works

The simulation is MuJoCo: a 0.5 kg cart on a ±2.4 m rail, links of 0.4 m and 0.1 kg,
and a motor limited to ±10 N at 100 Hz. `mujoco.rollout` steps 32 cart-pendulums at
once.

The policy sees the cart's position and velocity, and for each link the cosine and
sine of its angle, its angular velocity and how much of its rotation budget is left.

The reward is always between 0 and 1:

    reward = 0.05 × height + 0.95 / (1 + error)

The link furthest from upright decides both terms. Its height is 0 hanging and 1
upright. The error grows with its angle (scale 10°), the fastest link's speed
(0.5 rad/s), the cart's speed (0.5 m/s) and the cart's distance from the centre
(1.5 m). Height alone is worth at most 0.05; the rest requires being upright, still
and centred.

An episode ends in one of three ways:

- The cart reaches 2.3 m. This costs 1, because hanging earns almost nothing and
  hitting the rail would otherwise be free.
- A link has rotated 4π in total, counting every swing back and forth. Without this
  limit, spinning the links is an easy way to collect height reward.
- 15 seconds pass. This only cuts the episode short: PPO bootstraps from the value of
  the last state, as if the episode went on.

Three in four episodes start hanging. One in four starts within 2° of upright, so the
policy practises balancing before it can swing up.

PPO follows CleanRL: separate 128 × 128 tanh networks for policy and value,
32 environments × 256 steps per update, minibatches of 512 for 10 epochs, γ = 0.999,
GAE λ = 0.98, entropy 0.01, and a learning rate of 3e-4 that decays linearly to zero.
The gradient norm is clipped to 0.5, for policy and value separately.

## Files

- `model.py` builds the MuJoCo model.
- `reward.py` computes the reward.
- `environment.py` steps many pendulums at once.
- `ppo.py` is PPO.
- `train.py` trains one seed.
- `evaluate.py` tests a policy and records a video.

MIT license.
