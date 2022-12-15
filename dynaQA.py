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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def store(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(10000)
# representation:  s_0 = h(o_1, ..., o_t)
# dynamics:        r_k, s_k = g(s_km1, a_k)
# prediction:      p_k, v_k = f(s_k)
class Model(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims):
        super().__init__()

        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.representation = nn.Sequential(
          nn.Linear(in_dims, 128), nn.ReLU(),
          nn.Linear(128, 128),     nn.ReLU(),
          nn.Linear(128, hid_dims)
        ).to(device)

        self.dynamics = nn.Sequential(
          nn.Linear(hid_dims+out_dims, 128), nn.ReLU(),
          nn.Linear(128, 128),     nn.ReLU(),
          nn.Linear(128, hid_dims+1)
        ).to(device)

        self.prediction = nn.Sequential(
          nn.Linear(hid_dims, 128), nn.ReLU(),
          nn.Linear(128, 128),     nn.ReLU(),
          nn.Linear(128, out_dims)
        ).to(device)

        self.optimizer = optim.AdamW(
          list(self.representation.parameters()) +\
          list(self.dynamics.parameters()) +\
          list(self.prediction.parameters()),
          lr=LR, 
          amsgrad=True
        )

    def ht(self, x):
        return self.representation(x)

    def gt(self, x):
        out = self.dynamics(x)
        reward = out[:, -1]
        nstate = out[:, 0:self.hid_dims]
        return nstate, reward

    def ft(self, x):
        action = self.prediction(x)
        return action

    def predict(self, obs):
        state = self.ht( obs )
        action = self.ft( state )
        return action

    def forward(self, obs):
        state = self.ht( obs )
        action = self.ft( state )
        next_state, reward = self.gt( torch.cat([state, action],dim=1) )
        action = action.max(1)[1].view(1, 1)
        return action, next_state, reward 

  
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class DQN:
    def __init__(self, in_dim, out_dim):
        self.policy = Model(in_dim, in_dim, out_dim)
        self.target = Model(in_dim, in_dim, out_dim)
        self.target.load_state_dict(self.policy.state_dict())

        self.action_size = out_dim

    def select_action(self, obs):
        global steps_done
        steps_done += 1
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        with torch.no_grad():
            action, next_state, reward = self.policy(state)
        if sample > eps_threshold:
            return action, next_state, reward 
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), next_state, reward 
    
    def update_target(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_state_dict = self.target.state_dict()
        policy_state_dict = self.policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key]*TAU + target_state_dict[key]*(1-TAU)
        self.target.load_state_dict(target_state_dict)

    def train(self):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_pred = self.policy.predict(state_batch).gather(1, action_batch)
        q_targ = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad(): q_targ[non_final_mask] = self.target.predict(non_final_next_states).max(1)[0]
        q_targ = reward_batch + q_targ*GAMMA

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_pred, q_targ.unsqueeze(1))

        # Optimize the model
        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100) # In-place gradient clipping
        self.policy.optimizer.step()

obs, _ = env.reset()
agent = DQN(in_dim=len(obs), out_dim=env.action_space.n)

scores = []
for epi in range(600):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while True:
        action, nstate, rew = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if np.random.random() < 0.6:
            memory.store(state, action, next_state, reward)
        else:
            rew = torch.tensor([rew], device=device)
            memory.store(state, action, nstate, rew)

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
