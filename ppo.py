"""PPO for a batch of environments, following CleanRL's continuous-action PPO."""

import numpy as np
import torch
from torch import nn

ENVIRONMENTS = 32
ROLLOUT_STEPS = 256
BATCH = ENVIRONMENTS * ROLLOUT_STEPS
MINIBATCH = 512
EPOCHS = 10
LEARNING_RATE = 3e-4  # falls linearly to zero over the run
GAMMA = 0.999
GAE_LAMBDA = 0.98
CLIP = 0.2
ENTROPY_COST = 0.01
VALUE_COST = 0.5
MAX_GRAD_NORM = 0.5
MAX_KL = 0.03  # ends an update's epochs early
HIDDEN = 128


def network(inputs, outputs, output_gain):
    layers = [
        nn.Linear(inputs, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, outputs),
    ]
    for layer in layers[::2]:
        nn.init.orthogonal_(layer.weight, np.sqrt(2))
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, output_gain)
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        self.actor = network(inputs, 1, 0.01)
        self.critic = network(inputs, 1, 1.0)
        self.log_std = nn.Parameter(torch.zeros(1))

    def policy(self, observations):
        return torch.distributions.Normal(self.actor(observations), self.log_std.exp())

    def value(self, observations):
        return self.critic(observations).squeeze(-1)

    @torch.no_grad()
    def act(self, observations):
        """The mean action, for evaluation."""
        return self.actor(torch.as_tensor(observations)).numpy()


def train(env, agent, updates):
    """Run PPO, yielding the returns of the episodes that ended in each update."""
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    shape = (ROLLOUT_STEPS, env.count)
    observations = torch.zeros(shape + (agent.inputs,))
    actions = torch.zeros(shape + (1,))
    log_probs, values, rewards, dones = (torch.zeros(shape) for _ in range(4))
    observation = torch.as_tensor(env.observe())
    episode_returns = np.zeros(env.count)

    for update in range(updates):
        optimizer.param_groups[0]["lr"] = LEARNING_RATE * (1 - update / updates)
        finished = []
        for step in range(ROLLOUT_STEPS):
            with torch.no_grad():
                policy = agent.policy(observation)
                action = policy.sample()
                log_probs[step] = policy.log_prob(action).sum(-1)
                values[step] = agent.value(observation)
            observations[step], actions[step] = observation, action
            next_observation, reward, terminated, truncated, info = env.step(
                action.numpy()
            )
            done = terminated | truncated
            episode_returns += reward
            finished += episode_returns[done].tolist()
            episode_returns[done] = 0

            reward = torch.as_tensor(reward, dtype=torch.float32)
            if truncated.any():
                # The time limit is not part of the task: the episode would go on.
                cut = torch.as_tensor(truncated)
                final = torch.as_tensor(info["final_observation"][truncated])
                with torch.no_grad():
                    reward[cut] += GAMMA * agent.value(final)
            rewards[step], dones[step] = reward, torch.as_tensor(done)
            observation = torch.as_tensor(next_observation)

        with torch.no_grad():
            last_value = agent.value(observation)
        advantages = advantage(rewards, values, dones, last_value)
        improve(
            agent,
            optimizer,
            observations,
            actions,
            log_probs,
            advantages,
            advantages + values,
        )
        yield finished


def advantage(rewards, values, dones, last_value):
    """Generalised advantage estimation that never looks past a finished episode."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(len(rewards))):
        alive = 1 - dones[step]
        delta = rewards[step] + GAMMA * next_value * alive - values[step]
        running = delta + GAMMA * GAE_LAMBDA * alive * running
        advantages[step] = running
        next_value = values[step]
    return advantages


def improve(agent, optimizer, observations, actions, log_probs, advantages, returns):
    observations, actions, log_probs, advantages, returns = (
        x.flatten(0, 1) for x in (observations, actions, log_probs, advantages, returns)
    )
    for _ in range(EPOCHS):
        for index in torch.randperm(len(observations)).split(MINIBATCH):
            policy = agent.policy(observations[index])
            log_ratio = policy.log_prob(actions[index]).sum(-1) - log_probs[index]
            ratio = log_ratio.exp()
            with torch.no_grad():
                if ((ratio - 1) - log_ratio).mean() > MAX_KL:
                    return
            gain = advantages[index]
            gain = (gain - gain.mean()) / (gain.std() + 1e-8)
            clipped = ratio.clamp(1 - CLIP, 1 + CLIP)
            policy_loss = -torch.min(gain * ratio, gain * clipped).mean()
            value_loss = (agent.value(observations[index]) - returns[index]).pow(2)
            entropy = policy.entropy().sum(-1).mean()
            loss = policy_loss + VALUE_COST * value_loss.mean() - ENTROPY_COST * entropy
            optimizer.zero_grad()
            loss.backward()
            # Clipped apart, or the value error would set the size of the policy's step.
            nn.utils.clip_grad_norm_(
                [*agent.actor.parameters(), agent.log_std], MAX_GRAD_NORM
            )
            nn.utils.clip_grad_norm_(agent.critic.parameters(), MAX_GRAD_NORM)
            optimizer.step()
