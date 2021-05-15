"""
Code inspired by the YouTube tutorial: https://www.youtube.com/watch?v=ZhFO8EWADmY&t=3016s
and the OpenAI Spinning Up documentation: https://spinningup.openai.com/en/latest/algorithms/td3.html
"""
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.network import StateNetwork, BaseNetwork
from models.policy import GenericPolicy
import os


class Critic(BaseNetwork):
    """
    Critic Network.
    """
    '''def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir='tmp/td3'):'''

    def __init__(self, critic_lr, out_channels, n_actions, name, chkpt_dir='saved_models/td3'):
        super(Critic, self).__init__(out_channels=out_channels,
                                     action_size=n_actions)
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cuda:0')

        # Model is not large. Save GPU for training
        self.to(self.device)  # send model to GPU

    def save_checkpoint(self):
        """Saves model to directory"""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # print('Save checkpoint to <%s>' % self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Loads model from directory"""
        print('Load checkpoint from <%s>' % self.checkpoint_dir)
        self.load_state_dict(T.load(self.checkpoint_file))


class Actor(nn.Module):
    """
    Actor Network.
    """
    def __init__(self, actor_lr, out_channels, n_actions, name, chkpt_dir='saved_models/td3'):
        super(Actor, self).__init__()
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        # Layers
        self.state_net = StateNetwork(out_channels)
        self.fc1 = nn.Linear(7 * 7 * (out_channels + 1), out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.mu = nn.Linear(out_channels, n_actions)

        # Initialize weights of network (Xavier initialization)
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cuda:0')

        # Model is not large. Save GPU for training
        self.to(self.device)  # send model to GPU

    def forward(self, img_obs, current_time_step):
        prob = self.state_net(img_obs, current_time_step)
        prob = prob.view(img_obs.shape[0], -1)  # reshape state representation

        prob = self.fc1(prob)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))
        return mu

    def save_checkpoint(self):
        """Saves model to directory"""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # print('Save checkpoint to <%s>' % self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Loads model from directory"""
        print('Load checkpoint from <%s>' % self.checkpoint_dir)
        self.load_state_dict(T.load(self.checkpoint_file))


class TD3(GenericPolicy):
    def __init__(self, actor_lr, critic_lr, out_channels,
                 min_action, max_action, update_actor_interval=2, warmup=1000,
                 n_actions=4, batch_size=512, gamma=0.90, noise=0.1, tau=1e-2):
        self.gamma = gamma
        self.tau = tau
        self.min_action = min_action
        self.max_action = max_action

        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        # Instantiate actor and critics
        self.actor = Actor(actor_lr, out_channels, n_actions=n_actions, name='actor')

        self.critic_1 = Critic(critic_lr, out_channels,
                               n_actions=n_actions, name='critic_1')

        self.critic_2 = Critic(critic_lr, out_channels,
                               n_actions=n_actions, name='critic_2')

        # Instantiate target networks
        self.target_actor = Actor(actor_lr, out_channels, n_actions=n_actions, name='target_actor')
        self.target_actor.eval()

        self.target_critic_1 = Critic(critic_lr, out_channels,
                                      n_actions=n_actions, name='target_critic_1')
        self.target_critic_1.eval()

        self.target_critic_2 = Critic(critic_lr, out_channels,
                                      n_actions=n_actions, name='target_critic_2')
        self.target_critic_2.eval()

        self.noise = noise
        # Copy the weights from networks to target networks (tau=1 for direct copying).
        self.target_critic_1 = self.soft_update(self.critic_1, self.target_critic_1, tau=1)
        self.target_critic_2 = self.soft_update(self.critic_2, self.target_critic_2, tau=1)
        self.target_actor = self.soft_update(self.actor, self.target_actor, tau=1)

    def get_weights(self):
        return (self.actor.state_dict(),
                self.critic_1.state_dict(),
                self.critic_2.state_dict(),
                self.target_actor.state_dict(),
                self.target_critic_1.state_dict(),
                self.target_critic_2.state_dict())

    def set_weights(self, weights):
        self.actor.load_state_dict(weights[0])
        self.critic_1.load_state_dict(weights[1])
        self.critic_2.load_state_dict(weights[2])
        self.target_actor.load_state_dict(weights[3])
        self.target_critic_1.load_state_dict(weights[4])
        self.target_critic_2.load_state_dict(weights[5])

    @T.no_grad()
    def sample_action(self, state, timestep):
        # warmup are how many steps the agent should explore, before it starts to take deterministic actions
        if timestep < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to('cuda:0')
        else:
            self.actor.eval()
            if isinstance(state, np.ndarray):
                state = T.from_numpy(state).to('cuda:0')
            if isinstance(timestep, float):
                # timestep = T.tensor([timestep], device=self.actor.device)
                timestep = T.tensor([timestep]).to('cuda:0')
            mu = self.actor(state, timestep).to('cuda:0')

        # mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to('cuda:0')
        mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        # return mu_prime.detach()  # problem w/o cpu()
        # return mu_prime.cpu().detach()  # before
        # return mu_prime.cpu().detach().numpy()
        return mu_prime.detach()

    def train(self, memory, gamma, batch_size, tau=1e-2):
        if memory.cur_idx < batch_size:
            if memory.cur_idx % 10 == 0:
                print('memory.cur_idx:', memory.cur_idx)
            return

        # Randomly sample a batch of experiences from replay buffer, and convert elements
        # from numpy ndarrays to Torch tensors, and send them to GPU.
        self.actor.train()

        state, action, reward, next_state, done, step_ctr = memory.sample(batch_size)

        state = T.from_numpy(state).to(self.critic_1.device)
        action = T.from_numpy(action).to(self.critic_1.device)
        reward = T.from_numpy(reward).to(self.critic_1.device)
        next_state = T.from_numpy(next_state).to(self.critic_1.device)
        done = T.from_numpy(done).to(self.critic_1.device)

        timestep = T.from_numpy(step_ctr).to(self.critic_1.device)
        next_timestep = T.from_numpy(step_ctr + 1.0).to(self.critic_1.device)

        with T.no_grad():
            # Compute target actions
            target_actions = self.target_actor(next_state, next_timestep)
            target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
            target_actions = T.clamp(target_actions, self.min_action, self.max_action)

            # Compute targets
            pred_target_q1 = self.target_critic_1(next_state, next_timestep, target_actions).view(-1)
            pred_target_q2 = self.target_critic_2(next_state, next_timestep, target_actions).view(-1)
            target_q = T.min(pred_target_q1, pred_target_q2)  # min of the two predicted values from target critics

            target = reward + (1 - done) * gamma * target_q

        # Train the critics
        pred_q1 = self.critic_1(state, timestep, action)
        pred_q2 = self.critic_2(state, timestep, action)

        # Critic 1
        self.critic_1.optimizer.zero_grad()
        q1_loss = T.mean((pred_q1 - target) ** 2)  # MSE loss
        # print('q1_loss:', q1_loss.item())
        q1_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 10.)  # gradient clipping
        self.critic_1.optimizer.step()

        # Critic 2
        self.critic_2.optimizer.zero_grad()
        q2_loss = T.mean((pred_q2 - target) ** 2)  # MSE loss
        q2_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 10.)  # gradient clipping
        self.critic_2.optimizer.step()

        # Train the actor
        if self.learn_step_cntr % self.update_actor_iter == 0:
            self.actor.optimizer.zero_grad()
            pred_action = self.actor(state, timestep)
            pred_q_value = self.critic_1(state, timestep, pred_action)
            actor_loss = -T.mean(pred_q_value)

            actor_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)  # gradient clipping
            self.actor.optimizer.step()

            # Update target networks with soft update
            self.target_critic_1 = self.soft_update(self.critic_1, self.target_critic_1, tau)
            self.target_critic_2 = self.soft_update(self.critic_2, self.target_critic_2, tau)
            self.target_actor = self.soft_update(self.actor, self.target_actor, tau)

        self.learn_step_cntr += 1

    @staticmethod
    def soft_update(network, target_network, tau=None):
        """
        Soft update of target networks.
        Function borrowed from:
        https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/td3.py
        :param network:
        :param target_network:
        :param tau:
        :return:
        """
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        return target_network

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
