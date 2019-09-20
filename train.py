import random
from collections import deque

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent


def train(env, agent, weight_path, n_episodes=1000, threshold=30.0,
          n_agents=20):
    """Train agent and store weights if successful.

    Args:
        env (UnityEnvironment): Environment to train agent in
        agent (Agent): Agent to train
        weight_path (str): Path for weights file
        n_episodes (int): Max number of episodes to train agent for
        threshold (float): Min mean score over 100 episodes consider success
    """
    # Assume we're operating brain 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    mean_scores = []
    mean_score_window = deque(maxlen=100)
    best_score = -np.Inf

    for i in range(1, n_episodes + 1):
        agent.reset()

        env_info = env.reset(train_mode=True)[brain_name]

        states = env_info.vector_observations
        scores = np.zeros(n_agents)

        while True:
            # TODO: Do more runs before updating to get better estimate of reward?
            actions = np.array([np.squeeze(agent.act(states[j]))
                                for j in range(n_agents)])
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for j in range(n_agents):
                agent.step(states[j], actions[j], rewards[j],
                           next_states[j], dones[j])
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        mean_score = np.mean(scores)
        max_score = np.max(scores)

        mean_score_window.append(mean_score)
        mean_scores.append(mean_score)

        if max_score > best_score:
            best_score = max_score

        print(
            f"\rEpisode {i:4d}\tAverage score {np.mean(mean_score_window):.2f} (last {mean_score:.2f} [best now {max_score:.2f} vs. ever {best_score:.2f}])",
            end="\n" if i % 100 == 0 else "")
        if len(mean_score_window) >= 100 and np.mean(mean_score_window) > threshold:
            print(f"\nEnvironment solved in {i} episodes.")
            agent.save_weights(weight_path)
            break

        if i % 100 == 0:
            # Save checkpoint
            agent.save_weights(f"{weight_path}-{i}-CHECKPOINT")

    # Save weights even if not solved
    if len(mean_score_window) < 100 or np.mean(mean_score_window) < threshold:
        print("\nFailed to solve environment.")
        agent.save_weights(weight_path + "-FAILED")

    return scores

def export_scores(path, score):
    """Quick and dirty export of array to text file"""
    with open(path, "w") as f:
        lines = map(lambda x: str(x) + '\n', score)
        f.writelines(lines)

@click.command()
@click.option('--environment', required=True,
              help="Path to Unity environment", type=click.Path())
@click.option('--layer1', default=16, help="Number of units in input layer")
@click.option('--plot-output', default="score.png",
              help="Output file for score plot", type=click.Path())
@click.option('--scores-output', default="scores.txt",
              help="Output file for scores", type=click.Path())
@click.option('--weights-output', default='weights.pth',
              help="File to save weights to after success", type=click.Path())
@click.option('--seed', type=int, help="Random seed")
def main(environment, layer1, plot_output, scores_output, weights_output, seed):
    # Set seed if given to help with reproducibility
    if not seed:
        seed = random.randint(0, 30000000)

    if seed:
        print(f"Using seed {seed}")
        random.seed(seed)
        torch.random.manual_seed(seed)

    # Initialize Unity environment from external file
    env = UnityEnvironment(file_name=environment, no_graphics=True,
                           seed=seed if seed else 0)

    # Use CUDA if available, cpu otherwise
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create agent given model parameters
    agent = Agent(state_size=33, action_size=4, device=device)

    # Train agent (will save weights if successful)
    scores = train(env, agent, weights_output)

    env.close()

    # Save scores and generate crude plot
    export_scores(scores_output, scores)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(plot_output)


if __name__ == '__main__':
    main()
