# heavily inspired : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# works only for openai gym 0.26 <=
import gym
import random
import collections 
from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# params
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON = 1
EPS_MIN = 0.05
EPS_DECAY = 0.9
TAU = 0.005
LR = 1e-3

env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dims)
        self.optimizer = optim.AdamW(self.parameters(), lr=LR, amsgrad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x
  
Transition = namedtuple('Transition',('obs', 'action', 'next_obs', 'reward','done'))
class DQN:
    def __init__(self, in_dim, out_dim):
        self.memory = collections.deque([], maxlen=100000)
        self.policy = Network(in_dim, out_dim).to(device)
        self.target = Network(in_dim, out_dim).to(device) 
        self.action_size = out_dim

    def store(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def update_epsilon(self):
        global EPSILON
        EPSILON = max(EPS_MIN, EPSILON*EPS_DECAY)

    def update_target(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_state_dict = self.target.state_dict()
        policy_state_dict = self.policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key]*TAU + target_state_dict[key]*(1-TAU)
        self.target.load_state_dict(target_state_dict)


    def select_action(self, obs):
        with torch.no_grad():
            if np.random.random() > EPSILON:
                action = self.policy(obs)
                #action = action.argmax().item()
                action = action.max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[np.random.randint( self.action_size, size=1)[0]]])
        return action
        
    def train(self):
        if len(self.memory) >= BATCH_SIZE:
            for i in range(1):
                transitions = random.sample(self.memory, BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                #namedtuple('Transition',('obs', 'action', 'next_obs', 'reward'))
                obs_batch    = torch.cat(batch.obs)
                action_batch = torch.cat(batch.action)
                n_obs_batch  = torch.cat(batch.next_obs)
                reward_batch = torch.cat(batch.reward)[:,  None]
                dones = batch.done

                q_pred = self.policy(obs_batch).gather(1, action_batch)
                q_targ = self.target( n_obs_batch ).max(1)[0].unsqueeze(1)
                q_targ[dones] = 0.0  # set all terminal states' value to zero
                q_targ = reward_batch + GAMMA * q_targ 

                # Compute Huber loss
                criterion = nn.SmoothL1Loss()
                #loss = criterion(q_pred, q_targ.unsqueeze(1))
                #loss = criterion(q_pred, q_targ)
                loss = F.smooth_l1_loss(q_pred, q_targ).to(device)

                # Optimize the model
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()
        pass


obs, _ = env.reset()
agent = DQN(in_dim=len(obs), out_dim=env.action_space.n)

scores = []
for epi in range(600):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    while True:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        if not terminated:
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            agent.store(obs, action, next_obs, reward, done)
        agent.train()


        obs = next_obs
        if "episode" in info.keys():
            scores.append( info['episode']['r'] )
            print(f"Episode {epi}, Return: {info['episode']['r']}")
            break
    agent.update_epsilon()
    agent.update_target()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.arange(len(scores)), scores)

ax.set(xlabel='Episodes', 
       ylabel='Episodic return',
       title='CartPole-v1')
ax.grid()
#fig.savefig("test.png")
plt.show()  
