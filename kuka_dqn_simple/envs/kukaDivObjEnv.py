from envs.kukaGymEnv import KukaGymEnv
import random
import os
from gym import spaces
import time
import pybullet as p
import kuka
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym


class KukaDiverseObjectEnv(KukaGymEnv):
  """
  Class for Kuka environment with diverse objects.
  In each episode some objects are chosen from a set of 1000 diverse objects.
  These 1000 objects are split 90/10 into a train and test set.
  """

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=8,
               dv=0.06,
               removeHeightHack=False,
               blockRandom=0.3,
               cameraRandom=0,
               width=48,
               height=48,
               numObjects=5,
               isTest=False):
    """Initializes the KukaDiverseObjectEnv.
    Args:
      urdfRoot: The diretory from which to load environment URDF's.
      actionRepeat: The number of simulation steps to apply for each action.
      isEnableSelfCollision: If true, enable self-collision.
      renders: If true, render the bullet GUI.
      isDiscrete: If true, the action space is discrete. If False, the
        action space is continuous.
      maxSteps: The maximum number of actions per episode.
      dv: The velocity along each dimension for each action.
      removeHeightHack: If false, there is a "height hack" where the gripper
        automatically moves down for each action. If true, the environment is
        harder and the policy chooses the height displacement.
      blockRandom: A float between 0 and 1 indicated block randomness. 0 is
        deterministic.
      cameraRandom: A float between 0 and 1 indicating camera placement
        randomness. 0 is deterministic.
      width: The image width.
      height: The observation image height.
      numObjects: The number of objects in the bin.
      isTest: If true, use the test set of objects. If false, use the train
        set of objects.
    """

    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._dv = dv
    self._p = p
    self._removeHeightHack = removeHeightHack
    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._numObjects = numObjects
    self._isTest = isTest

    if self._renders:
      self.cid = p.connect(p.SHARED_MEMORY)
      if (self.cid < 0):
        self.cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      self.cid = p.connect(p.DIRECT)
    self.seed()

    # observationDim = len(self.getExtendedObservation())
    observationDim = 9
    observation_high = np.array([100] * observationDim)

    if (self._isDiscrete):
      if self._removeHeightHack:
        self.action_space = spaces.Discrete(9)
      else:
        self.action_space = spaces.Discrete(7)
    else:
      self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
      if self._removeHeightHack:
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da

    self.observation_space = spaces.Box(-observation_high, observation_high)
    '''self.observation_space = spaces.Box(low=0, high=255, shape=(self._height,
                                                                self._width,
                                                                3))'''
    self.viewer = None

  def reset(self):
    """Environment reset called at the beginning of an episode.
    """
    # Set the camera settings.
    look = [0.23, 0.2, 0.54]
    distance = 1.
    pitch = -56 + self._cameraRandom * np.random.uniform(-3, 3)
    yaw = 245 + self._cameraRandom * np.random.uniform(-3, 3)
    roll = 0
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
    aspect = self._width / self._height
    near = 0.01
    far = 10
    self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    # p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
               0.000000, 0.000000, 0.0, 1.0)

    xpos = 0.55 + 0.12 * random.random()
    ypos = 0 + 0.2 * random.random()
    ang = 3.14 * 0.5 + 3.1415925438 * random.random()
    #xpos = 0.55
    #ypos = 0.04
    #ang = 3.14 * 0.5 + 3.1415925438 * random.random()
    # ang = 3.14 * 0.5 + 3.1415925438 * random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                               orn[0], orn[1], orn[2], orn[3])

    p.setGravity(0, 0, -10)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()

    # Choose the objects in the bin.
    # urdfList = self._get_random_object(self._numObjects, self._isTest)
    # self._objectUids = self._randomly_place_objects(urdfList)

    # Spawn block objects
    # self._get_block_object()

    # Move gripper closer to tray
    for down in range(75):
        down = [0, 0, -0.001, 0, 0]
        self._kuka.applyAction(down)
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)


    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def _randomly_place_objects(self, urdfList):
    """Randomly places the objects in the bin.
    Args:
      urdfList: The list of urdf files to place in the bin.
    Returns:
      The list of object unique ID's.
    """

    # Randomize positions of each object urdf.
    objectUids = []
    for urdf_name in urdfList:
      xpos = 0.4 + self._blockRandom * random.random()
      ypos = self._blockRandom * (random.random() - .5)
      angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
      orn = p.getQuaternionFromEuler([0, 0, angle])
      urdf_path = os.path.join(self._urdfRoot, urdf_name)
      uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])
      objectUids.append(uid)
      # Let each object fall to the tray individual, to prevent object
      # intersection.
      for _ in range(500):
        p.stepSimulation()
    return objectUids

  '''
  def _get_observation(self):
    """Return the observation as an image.
    """
    img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=self._view_matrix,
                               projectionMatrix=self._proj_matrix)
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    return np_img_arr[:, :, :3]
  '''

  def getExtendedObservation(self):
    self._observation = self._kuka.getObservation()
    gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
    dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
    dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

    gripperEul = p.getEulerFromQuaternion(gripperOrn)
    # print("gripperEul")
    # print(gripperEul)
    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
    projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
    # print("projectedBlockPos2D")
    # print(projectedBlockPos2D)
    # print("blockEulerInGripper")
    # print(blockEulerInGripper)

    # we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

    self._observation.extend(list(blockInGripperPosXYEulZ))
    return self._observation

  def step(self, action):
    """Environment step.
    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    dv = self._dv  # velocity per physics step.
    if self._isDiscrete:
      # Static type assertion for integers.
      assert isinstance(action, int)
      if self._removeHeightHack:
        dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
        dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
        da = [0, 0, 0, 0, 0, 0, 0, -0.25, 0.25][action]
      else:
        dx = [0, -dv, dv, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0][action]
        dz = -dv
        # da = [0, 0, 0, 0, 0, -0.25, 0.25][action]  # original
        da = [0, 0, 0, 0, 0, -0.25, 0.25][action]
        # da = [0, 0, 0, 0, 0, -0.4, 0.4][action]
    else:
      dx = dv * action[0]
      dy = dv * action[1]
      if self._removeHeightHack:
        dz = dv * action[2]
        da = 0.25 * action[3]
      else:
        dz = -dv
        da = 0.25 * action[2]

    return self._step_continuous([dx, dy, dz, da, 0.3])

  def _step_continuous(self, action):
    """Applies a continuous velocity-control action.
    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    # Perform commanded action.
    self._env_step += 1
    self._kuka.applyAction(action)
    for _ in range(self._actionRepeat):
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      if self._termination():
        break

    # If we are close to the bin, attempt grasp.
    state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    end_effector_pos = state[0]
    # if end_effector_pos[2] <= 0.1:
    if end_effector_pos[2] <= 0.09:
      finger_angle = 0.3
      for _ in range(500):  # 500, 550
        grasp_action = [0, 0, 0, 0, finger_angle]
        self._kuka.applyAction(grasp_action)
        p.stepSimulation()
        #if self._renders:
        #  time.sleep(self._timeStep)
        finger_angle -= 0.3 / 100.
        if finger_angle < 0:
          finger_angle = 0

      for _ in range(500):  # 500, 450
        grasp_action = [0, 0, 0.001, 0, finger_angle]
        self._kuka.applyAction(grasp_action)
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)
        finger_angle -= 0.3 / 100.
        if finger_angle < 0:
          finger_angle = 0
      self._attempted_grasp = True
    # observation = self._get_observation()
    self._observation = self.getExtendedObservation()
    done = self._termination()
    reward = self._reward()
    # print('reward:', reward)
    return np.array(self._observation), reward, done, {}

  def _reward(self):
    # rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                       self._kuka.kukaEndEffectorIndex)

    reward = -1000
    numPt = len(closestPoints)
    if (numPt > 0):
      reward = -closestPoints[0][8] * 10
    if (blockPos[2] > 0.2):
      reward = reward + 10000
      # print("successfully grasped a block!!!")
    return reward

  def _termination(self):
    """Terminates the episode if we have tried to grasp or if we are above
    maxSteps steps.
    """
    return self._attempted_grasp or self._env_step >= self._maxSteps

  def _get_random_object(self, num_objects, test):
    """Randomly choose an object urdf from the random_urdfs directory.
    Args:
      num_objects:
        Number of graspable objects.
    Returns:
      A list of urdf filenames.
    """
    if test:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
    else:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[1-9]/*.urdf')
    found_object_directories = glob.glob(urdf_pattern)
    total_num_objects = len(found_object_directories)
    selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
    selected_objects_filenames = []
    for object_index in selected_objects:
      selected_objects_filenames += [found_object_directories[object_index]]
    return selected_objects_filenames

  def _get_block_object(self):
    xpos = 0.55 + 0.12 * random.random()
    ypos = 0 + 0.2 * random.random()
    ang = 3.14 * 0.5 + 3.1415925438 * random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                               orn[0], orn[1], orn[2], orn[3])
    self._objectUids = [self.blockUid]

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _reset = reset
    _step = step