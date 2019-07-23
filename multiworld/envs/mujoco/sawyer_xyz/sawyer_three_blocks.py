from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_block_stacking_camera


class SawyerThreeBlocksXYZEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,

            block_low=(-0.2, 0.55, 0.02),
            block_high=(0.2, 0.75, 0.02),

            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.75, 0.3),

            stack_goal_low=(-0.2, 0.55, 0.02),
            stack_goal_high=(0.2, 0.75, 0.02),

            hand_goal_low=(-0.2, 0.55, 0.3),
            hand_goal_high=(0.2, 0.75, 0.3),

            fix_goal=False,
            fixed_stack_goal=(0.0, 0.55, 0.02),
            fixed_hand_goal=(0.0, 0.75, 0.3),

            use_sparse_reward=False,
            sparse_reward_threshold=0.05,

            reset_free=False,
            hide_goal_markers=False,
            oracle_reset_prob=0.0,

            xml_path='sawyer_xyz/three_blocks.xml',

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=get_asset_full_path(xml_path),
            **kwargs)

        self.block_low = np.array(block_low)
        self.block_high = np.array(block_high)
        self.block_radius = 0.02

        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)

        self.stack_goal_low = np.array(stack_goal_low)
        self.stack_goal_high = np.array(stack_goal_high)

        self.hand_goal_low = np.array(hand_goal_low)
        self.hand_goal_high = np.array(hand_goal_high)

        self.fix_goal = fix_goal
        self.fixed_stack_goal = np.array(fixed_stack_goal)
        self.fixed_hand_goal = np.array(fixed_hand_goal)

        self.use_sparse_reward = use_sparse_reward
        self.sparse_reward_threshold = sparse_reward_threshold

        self.reset_free = reset_free
        self.hide_goal_markers = hide_goal_markers
        self.oracle_reset_prob = oracle_reset_prob

        self.action_space = Box(
            np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)

        diff = (self.block_high[0] - self.block_low[0]) / 3.0
        midpoint = diff + self.block_low[0]
        block_one_high = [midpoint, *self.block_high[1:]]
        block_two_low = [midpoint, *self.block_low[1:]]
        midpoint = midpoint + diff
        block_two_high = [midpoint, *self.block_high[1:]]
        block_three_low = [midpoint, *self.block_low[1:]]

        self.sampling_space = Box(
            np.hstack(([0.0], self.hand_low, self.block_low, block_two_low, block_three_low)),
            np.hstack(([0.04], self.hand_high, block_one_high, block_two_high, self.block_high)),
            dtype=np.float32)

        self.gripper_and_hand_and_blocks_space = Box(
            np.hstack(([0.0], self.hand_low, self.block_low, self.block_low, self.block_low)),
            np.hstack(([0.04], self.hand_high, self.block_high, self.block_high, self.block_high)),
            dtype=np.float32)

        self.observation_space = Dict([
            ('observation', self.gripper_and_hand_and_blocks_space),
            ('desired_goal', self.gripper_and_hand_and_blocks_space),
            ('achieved_goal', self.gripper_and_hand_and_blocks_space),
            ('state_observation', self.gripper_and_hand_and_blocks_space),
            ('state_desired_goal', self.gripper_and_hand_and_blocks_space),
            ('state_achieved_goal', self.gripper_and_hand_and_blocks_space),
            ('proprio_observation', self.gripper_and_hand_and_blocks_space),
            ('proprio_desired_goal', self.gripper_and_hand_and_blocks_space),
            ('proprio_achieved_goal', self.gripper_and_hand_and_blocks_space),])

        self.block_one_id = self.model.get_joint_qpos_addr("blockOneJoint")
        self.block_two_id = self.model.get_joint_qpos_addr("blockTwoJoint")
        self.block_three_id = self.model.get_joint_qpos_addr("blockThreeJoint")
        self.reset()

    def get_block_positions(self):
        return np.hstack([
            self.data.get_body_xpos('blockOne').copy(),
            self.data.get_body_xpos('blockTwo').copy(),
            self.data.get_body_xpos('blockThree').copy()])

    def set_block_positions(self, data):
        self.data.qpos[self.block_one_id[0]:self.block_one_id[1]] = np.append(data[:3], [1, 0, 1, 0])
        self.data.qpos[self.block_two_id[0]:self.block_two_id[1]] = np.append(data[3:6], [1, 0, 1, 0])
        self.data.qpos[self.block_three_id[0]:self.block_three_id[1]] = np.append(data[6:9], [1, 0, 1, 0])

    def viewer_setup(self):
        sawyer_block_stacking_camera(self.viewer.cam)

    def update_goal_markers(self, goal):
        self.data.site_xpos[self.model.site_name2id('handGoal')] = goal[1:4]
        self.data.site_xpos[self.model.site_name2id('blockOneGoal')] = goal[4:7]
        self.data.site_xpos[self.model.site_name2id('blockTwoGoal')] = goal[7:10]
        self.data.site_xpos[self.model.site_name2id('blockThreeGoal')] = goal[10:13]

        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('handGoal'), 2] = -1000
            self.data.site_xpos[self.model.site_name2id('blockOneGoal'), 2] = -1000
            self.data.site_xpos[self.model.site_name2id('blockThreeGoal'), 2] = -1000

    def get_observation(self):
        hand_position = self.get_endeff_pos()
        block_positions = self.get_block_positions()
        gripper_position = self.get_gripper_pos()

        flat_obs = np.concatenate((hand_position, block_positions, gripper_position))
        previous_goal = self._state_goal
        return dict(
            observation=flat_obs,
            desired_goal=previous_goal,
            achieved_goal=flat_obs,

            state_observation=flat_obs,
            state_desired_goal=previous_goal,
            state_achieved_goal=flat_obs,

            proprio_observation=flat_obs,
            proprio_desired_goal=previous_goal,
            proprio_achieved_goal=flat_obs)

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation(action[3:])
        new_block_positions = self.get_block_positions()

        new_block_positions[:3] = np.clip(
            new_block_positions[:3], self.block_low, self.block_high)
        new_block_positions[3:6] = np.clip(
            new_block_positions[3:6], self.block_low, self.block_high)
        new_block_positions[6:] = np.clip(
            new_block_positions[6:], self.block_low, self.block_high)

        self.set_block_positions(new_block_positions)
        self.last_block_positions = new_block_positions.copy()

        self.update_goal_markers(self._state_goal)
        observation = self.get_observation()
        reward = self.compute_reward(action, observation)
        return observation, reward, False, {}

    def reset_model(self):
        if self.reset_free:
            initial_pose = self.get_observation()["observation"]
        else:
            initial_pose = self.sampling_space.sample()

        # TODO: is this for loop really necessary?
        for _ in range(10):
            self.data.set_mocap_pos('mocap', initial_pose[1:4])
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

        self.set_block_positions(initial_pose[4:])
        if self.oracle_reset_prob > np.random.random():
            self.set_to_goal(self.sample_goal())

        self.set_goal(self.sample_goal())
        self.update_goal_markers(self._state_goal)
        return self.get_observation()

    def set_to_goal(self, goal):
        initial_pose = goal['state_desired_goal']
        for _ in range(30):
            self.data.set_mocap_pos('mocap', initial_pose[1:4])
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(np.array([-1]))

        self.set_block_positions(initial_pose[4:])
        for _ in range(10):
            self.do_simulation(None, self.frame_skip)
        self.sim.forward()

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal}

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        self.update_goal_markers(self._state_goal)

    def sample_goals(self, batch_size):
        if self.fix_goal:
            hand_goals = np.repeat(self.fixed_hand_goal.copy()[None], batch_size, 0)
            stack_goals = np.repeat(self.fixed_stack_goal.copy()[None], batch_size, 0)
        else:
            hand_goals = np.random.uniform(
                self.hand_goal_low,
                self.hand_goal_high,
                size=(batch_size, self.hand_goal_low.size))
            stack_goals = np.random.uniform(
                self.stack_goal_low,
                self.stack_goal_high,
                size=(batch_size, self.stack_goal_low.size))
        goals = np.hstack((
            np.repeat([[0.0]], batch_size, 0),
            hand_goals,
            stack_goals,
            stack_goals + np.array([[0.0, 0.0, 0.0 + 2.0 * self.block_radius]]),
            stack_goals + np.array([[0.0, 0.0, 0.0 + 4.0 * self.block_radius]])))
        return {
            'desired_goal': goals,
            'state_desired_goal': goals}

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']

        hand_positions = achieved_goals[:, 1:4]
        block_one_positions = achieved_goals[:, 4:7]
        block_two_positions = achieved_goals[:, 7:10]
        block_three_positions = achieved_goals[:, 10:13]

        hand_goals = desired_goals[:, 1:4]
        block_one_goals = desired_goals[:, 4:7]
        block_two_goals = desired_goals[:, 7:10]
        block_three_goals = desired_goals[:, 10:13]

        hand_goal_distances = np.linalg.norm(hand_goals - hand_positions, axis=1)
        hand_block_one_distances = np.linalg.norm(
            block_one_positions - hand_positions, axis=1)
        hand_block_two_distances = np.linalg.norm(
            block_two_positions - hand_positions, axis=1)
        hand_block_three_distances = np.linalg.norm(
            block_three_positions - hand_positions, axis=1)

        block_one_goal_distances = np.linalg.norm(
            block_one_goals - block_one_positions, axis=1)
        block_two_goal_distances = np.linalg.norm(
            block_two_goals - block_two_positions, axis=1)
        block_three_goal_distances = np.linalg.norm(
            block_three_goals - block_three_positions, axis=1)

        if self.use_sparse_reward:
            hand_reward = -(hand_goal_distances >
                            self.sparse_reward_threshold).astype(float)
            block_one_reward = -(block_one_goal_distances >
                                 self.sparse_reward_threshold).astype(float)
            block_two_reward = -(block_two_goal_distances >
                                 self.sparse_reward_threshold).astype(float)
            block_three_reward = -(block_three_goal_distances >
                                 self.sparse_reward_threshold).astype(float)
            additional_reward = 0.0
        else:
            hand_reward = -hand_goal_distances
            block_one_reward = -block_one_goal_distances
            block_two_reward = -block_two_goal_distances
            block_three_reward = -block_three_goal_distances
            additional_reward = -(hand_block_one_distances +
                                  hand_block_two_distances +
                                  hand_block_three_distances)

        return (hand_reward + block_one_reward +
                block_two_reward + block_three_reward +
                additional_reward)

    def get_diagnostics(self, paths, prefix=''):
        return OrderedDict()

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)

if __name__ == "__main__":

    import multiworld.envs.mujoco as m
    m.register_mujoco_envs()
    import gym
    x = gym.make("SawyerThreeBlocksXYZEnv-v0")
    import time
    while True:
        x.reset()
        for i in range(100):
            time.sleep(0.05)
            x.render()
            x.step(x.action_space.sample())
