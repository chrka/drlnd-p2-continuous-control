# DRLND Project 2: Continuous Control

_insert youtube capture here

The goal of the agent is to control a double-jointed arm and have it follow
a goal target.

The agent receives information about the position, rotation, velocity, and
angular velocities of the arm, for a total of 33 variables, ie., the agent
is dealing with a 33-dimensional state space. It controls the torques of the two 
joints described by a 4-dimensional action where each value is in the interval
$[-1,+1]$.

The agent receives a reward of +0.1 each time step the agent manages to keep
the arm's hand in the goal target. The task is considered solved when the agent
manages to get an average score of at least +30 over 100 consecutive episodes.

## Installation

Clone this repository and install the requirements needed as per the instructions below.

### Python Requirements

Follow the instructions in the Udacity [Deep Reinforcement Learning repository](https://github.com/udacity/deep-reinforcement-learning)
on how to set up the `drlnd` environment, and then also install the [Click](https://click.palletsprojects.com/en/7.x/)
package (used for handling command line arguments):
```shell
pip install click
```

Alternatively, on some systems it might be enough to install the required packages
from the provided `requirements.txt` file:
```shell
pip install -r requirements.txt
```

### Unity environment

Download the Unity environment appropriate for your operating system using the links below and unzip
it into the project folder.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Training and Running the Agent

To train the agent, use the `train.py` program which takes the Unity environment
and optional arguments to experiment with various parameters.

```shell
(drlnd) $ python train.py --help
Usage: train.py [OPTIONS]

Options:
  --environment PATH     Path to Unity environment  [required]
    ...
  --seed INTEGER         Random seed
  --help                 Show this message and exit.
```

The default values are:

| Option | Value |
|--------|-------|
|seed | None — do not set | 

For example:

```shell
(drlnd) $ python train.py --environment=Reacher.app --seed=20190415 
```

After successfully training the agent, use the `run.py` program to load
the weights and run the simulation, which takes similar parameters as
the training program:

```shell
(drlnd) $ python run.py --help
Usage: run.py [OPTIONS]

Options:
  --environment PATH    Path to Unity environment  [required]
    ...
  --help                Show this message and exit.
```

Default values for running the agent are:

| Option | Value |
|--------|-------|
|n-episodes | 3|

For example:
```
(drlnd) $ python run.py --environment=Reacher.app ...
```