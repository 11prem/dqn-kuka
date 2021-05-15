import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from models.dql.network import QNet, ReplayBuffer
from envs.kukaDivObjEnv import KukaDiverseObjectEnv

batch_size = 64  # 32
gamma = 0.99
epsilon = 0.9
# learning_rate = 1e-4
learning_rate = 0.0001
target_update = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = KukaDiverseObjectEnv(renders=True, isDiscrete=True, dv=0.05, maxSteps=50)  # , maxSteps=75)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# Define policy and target network
policy_net = QNet(input_dims=n_states, n_actions=n_actions, batch_size=batch_size,
                  gamma=gamma, epsilon=epsilon, device=device)

PATH = 'models_saved/dqn_72.pt'
PATH2 = 'dqn.pt'
chk_pnt = torch.load(PATH2)
policy_net.load_state_dict(chk_pnt['policy_net_state_dict'])
num_episodes = 25
# Main loop
for episode in range(num_episodes):
    state = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    step = 0
    done = False
    while not done:
        q = policy_net(state_tensor)
        action = torch.argmax(q, dim=1).item()

        next_state, reward, done, _ = env.step(action)
        # transition to new state
        state = next_state
        step += 1
