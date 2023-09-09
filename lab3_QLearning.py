"""
Code within all the 'COMPLETE THIS METHOD/TODO' sections was written by Xingchen (Estella) Ye.
"""

from typing import Dict, Any, Tuple, Callable
from collections import defaultdict

import gym
import numpy as np
import numpy.typing as npt
from rlhw.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, build_env: Callable[[bool], gym.Env], params: Dict[str, Any]):
        BaseAgent.__init__(self, build_env, params)
        self.env = build_env(False)
        self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def get_action(self) -> int:
        return self.env.action_space.sample()

    def step(self, action: int) -> Tuple[npt.NDArray, float, bool]:
        next_state, reward, done, _, _ = self.env.step(action)
        return next_state, reward, done

    def learn(self):
        """A random agent does not need to learn"""
        pass

    def run(self, max_episodes: int, max_steps: int, train: bool):
        if train:
            self.env = self.build_env(render=False)
        else:
            self.env = self.build_env(render=True)

        episode_rewards = dict()

        for ne in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            for nt in range(max_steps):
                action = self.get_action()
                next_state, reward, done = self.step(action)
                total_reward += reward

                if train:
                    self.learn()

                if done:
                    break

                state = next_state

            episode_rewards[ne] = total_reward
            print(f"episode {ne}: {total_reward}")

        return episode_rewards


class QLearningAgent(BaseAgent):
    def __init__(self, build_env: Callable[[bool], gym.Env], params: Dict[str, Any]):
        BaseAgent.__init__(self, build_env, params)
        self.env = build_env(render=False)
        self.render = self.params["render"]
        self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        self.lr = self.params["learning_rate"]
        self.min_lr = self.params["min_learning_rate"]
        self.lr_decay = self.params["learning_rate_decay"]
        self.gamma = self.params["discount_factor"]
        self.eps = self.params["epsilon"]
        self.eps_decay = self.params["epsilon_decay"]
        self.min_eps = self.params["min_epsilon"]

    def get_action(self, state: npt.NDArray, train: bool) -> int:
        """Return an action given the current state"""
        # YOUR CODE HERE
        num = self.env.action_space.n
        if train is True: # epsilon greedy
            if np.random.rand() < self.eps:
                return np.random.randint(low=0, high=num)

        # greedy
        return np.argmax(self.policy[state])



    def step(self, action: int) -> Tuple[npt.NDArray, float, bool]:
        next_state, reward, done, _, _ = self.env.step(action)
        return next_state, reward, done

    def learn(
        self, state: npt.NDArray, action: int, reward: float, next_state: npt.NDArray
    ) -> float:
        """Update Agent's policy using the Q-learning algorithm and return the update delta"""
        # YOUR CODE HERE
        error = reward + self.gamma * np.max(self.policy[next_state]) - self.policy[state][action]
        self.policy[state][action] += self.lr * error

        return error


    def run(self, max_episodes: int, max_steps: int, train: bool):
        """Run simulations of the environment with the agent's policy"""
        if self.render and not train:
            self.env = self.build_env(render=True)

        # YOUR CODE HERE
        epi_rewards = dict()
        for episode in range(max_episodes):
            state,_ = self.env.reset()
            epi_reward = 0
            n_step = 0
            done = False
            while not done and n_step < max_steps:
                action = self.get_action(state, train)
                next_state, reward, done = self.step(action)
                epi_reward += reward
                if train:
                    self.learn(state, action, reward, next_state)
                state = next_state
                n_step += 1
            epi_rewards[episode] = epi_reward
            self.eps = max(self.min_eps, self.eps_decay * self.eps)
            self.lr = max(self.min_lr, self.lr * self.lr_decay)
        
        return epi_rewards


