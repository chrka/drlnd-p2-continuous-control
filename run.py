import click
import torch
from unityagents import UnityEnvironment

from agent import Agent


def run(env, agent, n_episodes):
    """Run a pre-trained agent.

    Args:
        env (UnityEnvironment): Environment to run agent in
        agent (Agent): Agent to run
        n_episodes (int): Number of episodes to run
    """
    # Assume we're operating brain 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores = []
    for i in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        step = 0
        while True:
            step += 1
            action = agent.act(state, add_noise=False)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            state = next_state
            score += reward
            print(f"\rStep: {step:3d}\tScore {score}", end="")
            if done:
                break
        scores.append(score)
        print(f"\nEpisode {i}\t Score {score}")
    return scores


@click.command()
@click.option('--environment', required=True,
              help="Path to Unity environment", type=click.Path())
@click.option('--layer1', default=16, help="Number of units in hidden layer")
@click.option('--n-episodes', default=3, help="Number of episodes to run")
@click.option('--weights-input', default='weights.pth', help="Network weights",
              type=click.Path())
def main(environment, layer1, n_episodes, weights_input):
    env = UnityEnvironment(file_name=environment)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(state_size=33, action_size=4, device=device)
    agent.load_weights(weights_input)

    run(env, agent, n_episodes)

    env.close()


if __name__ == '__main__':
    main()
