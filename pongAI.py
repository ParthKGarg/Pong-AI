import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

UP = 0
STAY = 1
DOWN = 2

class PongNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 3)
        nn.init.uniform_(self.fc1.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc2.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x


class PongAI():
    def __init__(self, alpha=0.0001, epsilon=0.3, gamma = 0.99):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
        - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
        - `action` is a tuple `(i, j)` for an action
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.online_network = PongNetwork()
        self.target_network = PongNetwork()
        self.replay_buffer = deque(maxlen=100000)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=alpha)
        self.actions = [UP, STAY, DOWN]
        self.batch_size = 64
        self.update_target_network()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # random sampling from deque
        sample = random.sample(list(self.replay_buffer), k=self.batch_size)

        # unpack in tensors
        states = []
        actions = []
        new_states = []
        rewards = []
        for s, a, ns, r in sample:
            states.append(s)
            actions.append(a)
            new_states.append(ns)
            rewards.append(r)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # get current q values
        current_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # get the target values using bellman
        with torch.no_grad():
            target_q = self.target_network(new_states).max(dim=1).values
        targets = rewards + self.gamma * target_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1.0)
        self.optimizer.step()

    def remember(self, state, action, new_state, reward):
        self.replay_buffer.append((state, action, new_state, reward))
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def avg_q_value(self):
        state = torch.tensor([[400, 300, 5, 0, 250, 250]], dtype=torch.float32)
        vals = self.online_network(state)
        print(f"Q values: {vals}")
        return vals.mean().item()

    def get_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.online_network(state)

    def choose_action(self, state, epsilon=True):
        """
        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """
        
        if epsilon and random.random()<=self.epsilon:
            return random.choice(self.actions)
        
        ind = int(torch.argmax(self.get_q_value(state)))
        bestAction = self.actions[ind]
        
        return bestAction