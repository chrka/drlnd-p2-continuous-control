import random
from collections import deque

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent


def train(env, agent, weight_path, n_episodes=1000, threshold=30.0,
          noise_scale=0.5):
    """Train agent and store weights if successful.

    Args:
        env (UnityEnvironment): Environment to train agent in
        agent (Agent): Agent to train
        weight_path (str): Path for weights file
        n_episodes (int): Max number of episodes to train agent for
        threshold (float): Min mean score over 100 episodes consider success
    """
    # TODO: Consider adding discount rate
    # Assume we're operating brain 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    num_agents = 20

    mean_scores = []
    score_window = deque(maxlen=100)

    best_agent = agent
    previous_best_score = -np.Inf

    for i in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        scores = np.zeros(num_agents)

        agents = [best_agent] + [best_agent.randomly_displaced(noise_scale)
                                 for _ in range(num_agents - 1)]
        states = env_info.vector_observations

        while True:
            # TODO: Do more runs before updating to get better estimate of reward?
            actions = np.array([np.squeeze(agent.act(states[i]))
                                for i, agent in enumerate(agents)])
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        score_window.append(np.mean(scores))
        mean_scores.append(np.mean(scores))

        best_score = np.max(scores)

        print(
            f"\rEpisode {i:4d}\tAverage score {np.mean(score_window):.2f} (best {best_score:.2f}, scale {noise_scale:.3f})",
            end="\n" if i % 100 == 0 else "")
        if len(score_window) >= 100 and np.mean(score_window) > threshold:
            print(f"\nEnvironment solved in {i} episodes.")
            best_agent.save_weights(weight_path)
            break

        if best_score > previous_best_score:
            noise_scale = max(0.001, noise_scale / 2.0)
        else:
            noise_scale = min(2.0, noise_scale * 2.0)
        previous_best_score = best_score
        best_agent = agents[np.argmax(scores)]

    return mean_scores


@click.command()
@click.option('--environment', required=True,
              help="Path to Unity environment", type=click.Path())
@click.option('--layer1', default=16, help="Number of units in input layer")
@click.option('--plot-output', default="score.png",
              help="Output file for score plot", type=click.Path())
@click.option('--weights-output', default='weights.pth',
              help="File to save weights to after success", type=click.Path())
@click.option('--seed', type=int, help="Random seed")
def main(environment, layer1, plot_output, weights_output, seed):
    # Set seed if given to help with reproducibility
    if seed:
        print(f"Using seed {seed}")
        random.seed(seed)
        torch.random.manual_seed(seed)

    # Initialize Unity environment from external file
    env = UnityEnvironment(file_name=environment, no_graphics=True)

    # Use CUDA if available, cpu otherwise
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create agent given model parameters
    agent = Agent(state_size=33, action_size=4, device=device, layer1=layer1)

    # Train agent (will save weights if successful)
    scores = train(env, agent, weights_output)

    env.close()

    # Generate score plot
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(plot_output)


if __name__ == '__main__':
    main()
