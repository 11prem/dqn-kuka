import os
import numpy as np
import torch

from models.td3.td3 import TD3


def make_env(max_steps, render=False, dv=0.06, remove_height_hack=False, is_test=False):
    from kuka_env import KukaDiverseObjectEnv

    '''
    env = KukaDiverseObjectEnv(renders=True, isDiscrete=False, maxSteps=15, dv=0.05,
                               removeHeightHack=False, width=64, height=64)
    '''

    env_config = {'actionRepeat': 80,
                  'isEnableSelfCollision': True,
                  'renders': render,
                  'isDiscrete': False,
                  'maxSteps': max_steps,
                  'dv': dv,
                  'removeHeightHack': remove_height_hack,
                  'blockRandom': 0.3,
                  'cameraRandom': 0,
                  'width': 64,
                  'height': 64,
                  'numObjects': 5,
                  'isTest': is_test}

    def create():
        return KukaDiverseObjectEnv(**env_config)
    return create


def make_model():
    '''
    self, actor_lr, critic_lr, out_channels,
    min_action, max_action, update_actor_interval = 2, warmup = 1000,
    n_actions = 4, batch_size = 512, gamma = 0.90, noise = 0.1, tau = 1e-2
    '''
    # warmups = [1000, 500]
    '''
    For continuous actions:
    Height hack: n_actions =    4
    No height hack: n_actions = 3
    noise = 0.5 seems to work better for rotation than 0.1
    
    return TD3(actor_lr=1e-3, critic_lr=1e-3, out_channels=32,
                   min_action=-1, max_action=1, update_actor_interval=3,
                   warmup=1000, n_actions=3, batch_size=512, gamma=0.90,
                   noise=0.5, tau=1e-2)
                   
    return TD3(actor_lr=1e-3, critic_lr=1e-3, out_channels=32,
                   min_action=-1, max_action=1, update_actor_interval=2,
                   warmup=10000, n_actions=3, batch_size=32, gamma=0.98,
                   noise=0.40, tau=1e-2)
    
    
    
    
    return TD3(actor_lr=1e-3, critic_lr=1e-3, out_channels=32,
                   min_action=-1, max_action=1, update_actor_interval=2,
                   warmup=1000, n_actions=3, batch_size=128, gamma=0.98,
                   noise=0.35, tau=1e-2)
    
    '''
    def create():
        return TD3(actor_lr=1e-3, critic_lr=1e-3, out_channels=32,
                   min_action=-1, max_action=1, update_actor_interval=2,
                   warmup=0, n_actions=3, batch_size=128, gamma=0.98,
                   noise=0.25, tau=1e-2)
    return create


class EnvWrapper:
    def __init__(self, env_creator, model_creator, seed=None):
        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = env_creator()
        self.policy = model_creator()

    def step(self, action):
        """Takes a step in the environment."""
        if not isinstance(action, np.ndarray):
            action = action.cpu().numpy().flatten()
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


def main():
    # Model parameters
    # model_name = 'td3'
    buffer_size = 100000  # 200000 OOM error
    # max_epochs = 200

    # Hyperparameters
    seed = 1234  # For reproducability
    # out_channels = 32
    gamma = 0.98
    # lr = 1e-3
    tau = 1e-2
    batch_size = 128  # [512, 256, 128, 64, 32, 16, 8], 512
    # noise = 0.35
    # update_iter = 50
    # max_epochs = 200
    num_episodes = 10000000

    # Environment parameters
    max_steps = 15
    render = True
    test = False

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Make the environment and model
    '''env_fn = make_env(max_steps=max_steps, render=render, dv=0.05,
                      remove_height_hack=False, is_test=False)
    '''
    env_fn = make_env(max_steps=30, render=render, dv=0.04,
                      remove_height_hack=False, is_test=False)
    model_fn = make_model()

    # Environment wrapper
    env = EnvWrapper(env_fn, model_fn, seed=None)
    model = make_model()()

    # Make replay buffer
    from models.memory import BaseMemory as ReplayBuffer
    replay_buffer = ReplayBuffer(buffer_size)

    # score_file = 'plots/scores.png'
    load_from_checkpoint = False

    ######################### Train model #########################
    # Load model
    if load_from_checkpoint:
        model.load_models()

    # Stats
    # writer = SummaryWriter()
    total_rewards = []
    ten_rewards = 0
    best_mean_reward = None

    # Main loop
    for episode in range(num_episodes):
        print('episode: ', episode + 1)
        state = env.reset().transpose(2, 0, 1)[np.newaxis]
        step = 0.
        done = False

        score = 0

        while not done:
            state = state.astype(np.float32) / 255.  # Normalize state to [0, 1]. Grayscale image, and norm. time.
            action = model.sample_action(state, step)
            action = action.cpu().numpy().flatten()
            next_state, reward, done, _ = env.step(action)

            # store experience in replay buffer
            next_state = next_state.transpose(2, 0, 1)[np.newaxis]
            replay_buffer.add(state, action, reward, next_state, done, step)

            # learn
            model.train(replay_buffer, gamma, batch_size, tau=tau)
            score += reward

            # transition to new state
            state = next_state

            step = step + 1.

        # Stats.
        ten_rewards += reward
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-100:]) * 100
        # if (best_mean_reward is None or best_mean_reward < mean_reward) and episode > 100:
        if (best_mean_reward is None or best_mean_reward < mean_reward) and episode > 2:
            # Save model
            model.save_models()
            if best_mean_reward is not None:
                print("Best mean reward updated %.1f -> %.1f, model saved" % (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward

        if episode % 10 == 0:
            ten_rewards = 0
        if episode >= 1000 and mean_reward > 90:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode + 1, mean_reward))
            break

    # Print avg. score
    print('Average Score: {:.2f}'.format(mean_reward))

    ######################### Test model #########################


if __name__ == '__main__':
    main()
