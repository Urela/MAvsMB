import gym
import torch 
import numpy as np
import collections 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# minimalRL
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class MockModel(nn.Module):
    def __init__(self, input_dims, output_dims):
      pass

    def forward(self, x):
        raise NotImplementedError


class Agent():
    def __init__(self, input_dims, output_dims):
        self.model  = MockModel(input_dims, output_dims)
        self.memory = ReplayBuffer(10000)
        pass

    def selection_action(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

env = gym.make('CartPole-v1')
#env = gym.make("CartPole-v1", render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = Agent( env.observation_space.shape[0], env.action_space.n )

scores = []
for epi in range(1000):
  obs = env.reset()
  while True:

    action = env.action_space.sample()
    _obs, reward, truncated, terminated, info = env.step(action)
    done = truncated or terminated


    agent.memory.put((obs, action, reward/100.0, _obs, done))

    obs = _obs
    if "episode" in info.keys():
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
