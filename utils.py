import numpy as np
import random
import copy

from collections import deque


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
                [random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class PrioritisedReplayMemory:
    def __init__(self, maxlen, seed=None):
        """Initialize a ReplayBuffer object.
        """
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.seed = random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.append((state, action, reward, next_state, done))
        self.prioritis.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalised = importance / max(importance)
        return importance_normalised

    def sample(self, batch_size, priority_scale=1.0):
        sample_probs = self.get_probabilities(priority_scale)
        experiences_idx = random.sample(range(len(self.buffer)), batch_size, weights=sample_probs)
        experiences = np.array(self.buffer)[experiences_idx]
        states, actions, rewards, next_states, dones = map(
                np.vstack, zip(*experiences))
        importance = self.get_importance(sample_probs[experiences_idx])

        return states, actions, rewards, next_states, dones, importance


class ReplayMemory(deque):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed=None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        super().__init__(maxlen=buffer_size)
        self.seed = random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self, batch_size)
        states, actions, rewards, next_states, dones = map(
                np.vstack, zip(*experiences))

        return states, actions, rewards, next_states, dones


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (
            1.0-tau)*target_param.data)
