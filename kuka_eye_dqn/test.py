from kuka_env import KukaDiverseObjectEnv
import pybullet as p
from itertools import count

# actionRepeat=150
# maxSteps=20

env = KukaDiverseObjectEnv(actionRepeat=80, renders=True, isDiscrete=False, maxSteps=30, dv=0.02,
                           removeHeightHack=False, width=64, height=64)

for episode in range(10):
    print('episode:', episode)
    env.reset()

    for t in count():
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        if done:
            print('done!')
            break
        obs = next_obs

'''
for episode in range(10):
    print('episode:', episode)
    obs = env.reset()

    for t in range(50):
        # env.render()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        if done:
            print('Episode finished after {} timesteps'.format(t+1))
            break

env.close()'''

'''
for episode in range(10):
    print('episode:', episode)
    env.reset()

    for step in range(40):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        if done:
            print('done!')
            break
        obs = next_obs
'''