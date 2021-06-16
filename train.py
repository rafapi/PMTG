import itertools

import numpy as np
import matplotlib.pyplot as plt
import torch

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tg_env import BasicTG
from agent import DDPG
from utils import ReplayMemory


lr = 0.001
tau = 0.001
gamma = 0.99
random_seed = 0
replay_size = 1000000
batch_size = 256
num_steps = 1000000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.98
eval_every_N_episodes = 10
w_decay = 0

state_size = 5
action_size = 4


def compute_path(env, agent, num_steps, reward_acc):
    all_x = []
    all_y = []

    time = 0
    current_state = env.reset()

    while True:
        action = []
        action = agent.chooseAction(current_state)[0]

        # observation, reward, done, info
        next_state, reward, done, info = agent.step(action)
        print(next_state)

        all_x.append(next_state[1])
        all_y.append(next_state[2])

        # Update counters
        num_steps = num_steps + 1
        reward_acc = reward_acc + reward

        if done == True:
            env.handleGameEnd()
            break

        current_state = next_state
        time = time + 1

    time = 0
    current_state = env.reset()
    return all_x, all_y


def plotter(reward_window, env, learned_x, learned_y):
    # Plot the data
    plt.plot(reward_window)
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episode')
    plt.show()

    plt.scatter(env.x, env.y)

    # Compute the agent's path
    plt.scatter(learned_x, learned_y)
    plt.show()


def main():
    env = BasicTG()

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    agent = DDPG(
            state_size=state_size,
            action_size=action_size,
            lr=lr,
            tau=tau,
            gamma=gamma,
            random_seed=random_seed,
            use_cuda=True,
    )

    memory = ReplayMemory(replay_size)
    tb_writer = SummaryWriter('runs/{}_DDPG_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "OptimalPath"))

    eps = epsilon_start

    total_num_steps = 0
    reward_window = []
    for i_episode in itertools.count(0):
        episode_reward = 0
        episode_steps = 0
        done = False

        all_x = []
        all_y = []

        state = env.reset()
        # agent.reset()

        while not done:
            action = agent.select_action(state, eps, add_noise=True)
            next_state, reward, done, info = agent.step(env, action)

            memory.push(state, action, reward, next_state, done)

            # Learn, if enough samples are available in memory
            if len(memory) >= batch_size:
                critic_loss, actor_loss = agent.update_parameters(memory, batch_size)

                tb_writer.add_scalar('loss/critic', critic_loss, i_episode)
                tb_writer.add_scalar('loss/actor', actor_loss, i_episode)

            episode_steps += 1
            total_num_steps += 1
            episode_reward += reward
            state = next_state

            all_x.append(next_state[1])
            all_y.append(next_state[2])

            if total_num_steps >= num_steps:
                break

        if total_num_steps >= num_steps:
            break

        reward_window.append(episode_reward)

        eps = max(epsilon_end, epsilon_decay*eps)

        tb_writer.add_scalar('reward/episode', episode_reward, i_episode)

        plotter(reward_window, env, all_x, all_y)

        print(
            f"Episode: {i_episode}, total_steps: {total_num_steps}, \
                    episode steps: {episode_steps}, \
                    return: {round(episode_reward, 2)}"
        )

        if eval_every_N_episodes is not None and (i_episode + 1) % eval_every_N_episodes == 0:
            reward_average = np.mean(reward_window)
            reward_max = np.max(reward_window)
            print('Episode {}\tAverage Score: {:.2f}\tMax: {:.1f}'.format(
                i_episode, reward_average, reward_max))
            tb_writer.add_scalar('reward/average', reward_average, i_episode)

    env.close()


if __name__ == '__main__':
    main()