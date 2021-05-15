import numpy as np

from envs.kukaGymEnv import KukaGymEnv
# from envs.kukaDivObjEnv import KukaDiverseObjectEnv


env = KukaGymEnv(actionRepeat=1, renders=True, isDiscrete=True)  # , maxSteps=75)
# env = KukaDiverseObjectEnv(renders=True, isDiscrete=True, dv=0.05, maxSteps=50)  # , maxSteps=75)
n_states = env.observation_space.shape[0]
print('n_states:', n_states)
n_actions = env.action_space.n
print('n_actions:', n_actions)

num_episodes = 5
# Main loop
for episode in range(num_episodes):
    print('episode: ', episode + 1)
    state = env.reset()
    step = 0.
    done = False

    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        state = next_state  # transition to new state
        step = step + 1

