"""Train PPO to swing up and balance N links: python train.py --links 3 --seed 0"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from environment import Pendulums, observation_size
from evaluate import STARTS, record, successes
from ppo import BATCH, ENVIRONMENTS, Agent, train

STEPS = {1: 2_000_000, 2: 8_000_000, 3: 20_000_000, 4: 40_000_000}
EVALUATE_EVERY = 100  # updates
THREADS = 2  # four seeds at once fill the eight performance cores


def run(links, seed, updates, folder):
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    env = Pendulums(links, ENVIRONMENTS, seed, threads=THREADS)
    agent = Agent(observation_size(links))
    start, returns, rows = time.time(), [], []
    for update, finished in enumerate(train(env, agent, updates), start=1):
        returns += finished
        if update % EVALUATE_EVERY and update < updates:
            continue
        row = {
            "steps": update * BATCH,
            "minutes": round((time.time() - start) / 60, 1),
            "return": round(float(np.mean(returns)), 1),
            "held": int(successes(agent, links).sum()),
        }
        rows.append(row)
        returns = []
        print(
            f"{row['steps'] / 1e6:5.1f}M steps  {row['minutes']:5.1f} min  "
            f"return {row['return']:7.1f}  held {row['held']}/{STARTS}",
            flush=True,
        )
        torch.save({"links": links, "agent": agent.state_dict()}, folder / "policy.pt")
        write(rows, folder)
    env.close()
    record(agent, links, folder / "video.mp4")


def write(rows, folder):
    with open(folder / "progress.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    figure = Figure(figsize=(7, 5))
    top, bottom = figure.subplots(2, 1, sharex=True)
    steps = [row["steps"] / 1e6 for row in rows]
    top.plot(steps, [row["held"] for row in rows])
    top.set(ylabel=f"held / {STARTS}", ylim=(0, STARTS))
    bottom.plot(steps, [row["return"] for row in rows])
    bottom.set(ylabel="episode return", xlabel="million steps")
    figure.savefig(folder / "progress.png", dpi=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=int, choices=STEPS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    folder = Path("runs") / f"links_{args.links}" / f"seed_{args.seed}"
    run(args.links, args.seed, STEPS[args.links] // BATCH, folder)


if __name__ == "__main__":
    main()
