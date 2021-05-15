from kuka_env import KukaDiverseObjectEnv
# from kukaGymEnv import KukaGymEnv
import pybullet as p
from itertools import count

# from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv


# actionRepeat=150
# maxSteps=20

env = KukaDiverseObjectEnv(actionRepeat=80, renders=True, isDiscrete=True, maxSteps=40, dv=0.02,
                           removeHeightHack=True, width=64, height=64)

# env = KukaGymEnv(isDiscrete=True, renders=True)

for episode in range(7):
    print('episode:', episode)
    o = env.reset()
    print('obs suze', len(o))

    for t in count():
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        if done:
            print('done!')
            break
        obs = next_obs