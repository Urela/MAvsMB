import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps_done = 0

class Network(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dims)
        #self.optimizer = optim.AdamW(self.parameters(), lr=LR, amsgrad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x
  
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done'))
class DQN:
    def __init__(self, in_dim, out_dim):
        self.policy = Network(in_dim, out_dim).to(device)
        self.target = Network(in_dim, out_dim).to(device) 
        self.target.load_state_dict(self.policy.state_dict())

        self.action_size = out_dim
        self.memory = deque([], maxlen=100000)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=LR, amsgrad=True)

    def store(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def select_action(self, state):
        global steps_done
        steps_done += 1
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
    def update_target(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_state_dict = self.target.state_dict()
        policy_state_dict = self.policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key]*TAU + target_state_dict[key]*(1-TAU)
        self.target.load_state_dict(target_state_dict)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_pred = self.policy(state_batch).gather(1, action_batch)
        q_targ = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad(): q_targ[non_final_mask] = self.target(non_final_next_states).max(1)[0]
        q_targ = reward_batch + q_targ*GAMMA

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_pred, q_targ.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100) # In-place gradient clipping
        self.optimizer.step()

obs, _ = env.reset()
agent = DQN(in_dim=len(obs), out_dim=env.action_space.n)

scores = []
for epi in range(600):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while True:
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        agent.store(state, action, next_state, reward, done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        agent.train()
        agent.update_target()
        if done:
            scores.append( info['episode']['r'] )
            print(f"Episode {epi}, Return: {info['episode']['r']}")
            break

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.arange(len(scores)), scores)

ax.set(xlabel='Episodes', 
       ylabel='Episodic return',
       title='CartPole-v1')
ax.grid()
#fig.savefig("test.png")
plt.show()  
