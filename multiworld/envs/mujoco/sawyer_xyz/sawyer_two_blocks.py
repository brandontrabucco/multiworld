from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_block_stacking_camera


class SawyerTwoBlocksXYZEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,

            block_low=(-0.2, 0.55, 0.02),
            block_high=(0.2, 0.75, 0.02),

            hand_low=(0.0, 0.55, 0.3),
            hand_high=(0.0, 0.55, 0.3),

            stack_goal_low=(-0.2, 0.55, 0.02),
            stack_goal_high=(0.2, 0.75, 0.02),

            fix_goal=False,
            fixed_stack_goal=(0.0, 0.55, 0.02),

            use_sparse_reward=False,
            sparse_reward_threshold=0.05,

            reset_free=False,
            hide_goal_markers=False,
            oracle_reset_prob=0.0,

            xml_path='sawyer_xyz/two_blocks.xml',

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

        self.max_place_distance = max(
            np.linalg.norm((self.stack_goal_high - self.block_low), ord=2),
            np.linalg.norm((self.block_high - self.stack_goal_low), ord=2),
        )

        self.fix_goal = fix_goal
        self.fixed_stack_goal = np.array(fixed_stack_goal)

        self.use_sparse_reward = use_sparse_reward
        self.sparse_reward_threshold = sparse_reward_threshold

        self.reset_free = reset_free
        self.hide_goal_markers = hide_goal_markers
        self.oracle_reset_prob = oracle_reset_prob

        self.action_space = Box(
            np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)

        midpoint = (self.block_low[0] + self.block_high[0]) / 2.0
        block_one_high = [midpoint, *self.block_high[1:]]
        block_two_low = [midpoint, *self.block_low[1:]]

        self.sampling_space = Box(
            np.hstack(([0.0], self.hand_low, self.block_low, block_two_low)),
            np.hstack(([0.01], self.hand_high, block_one_high, self.block_high)),
            dtype=np.float32)

        self.gripper_and_hand_and_blocks_space = Box(
            np.hstack(([0.0], self.hand_low, self.block_low, self.block_low)),
            np.hstack(([0.01], self.hand_high, self.block_high, self.block_high)),
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
        self.reset()

    def get_block_positions(self):
        return np.hstack([
            self.data.get_body_xpos('blockOne').copy(),
            self.data.get_body_xpos('blockTwo').copy()])

    def set_block_positions(self, data):
        self.data.qpos[self.block_one_id[0]:self.block_one_id[1]] = np.append(data[:3], [1, 0, 1, 0])
        self.data.qpos[self.block_two_id[0]:self.block_two_id[1]] = np.append(data[3:], [1, 0, 1, 0])

    def viewer_setup(self):
        sawyer_block_stacking_camera(self.viewer.cam)

    def update_goal_markers(self, goal):
        self.data.site_xpos[self.model.site_name2id('handGoal')] = goal[1:4]
        self.data.site_xpos[self.model.site_name2id('blockOneGoal')] = goal[4:7]
        self.data.site_xpos[self.model.site_name2id('blockTwoGoal')] = goal[7:]

        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('handGoal'), 2] = -1000
            self.data.site_xpos[self.model.site_name2id('blockOneGoal'), 2] = -1000
            self.data.site_xpos[self.model.site_name2id('blockTwoGoal'), 2] = -1000

    def get_observation(self):
        hand_position = self.get_endeff_pos()
        block_positions = self.get_block_positions()
        gripper_position = self.get_gripper_pos()

        flat_obs = np.concatenate((gripper_position, hand_position, block_positions))
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
        self.set_xyz_action(action[1:4])
        self.do_simulation([action[0], -action[0]])
        new_block_positions = self.get_block_positions()

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
            self.do_simulation(np.array([-1, 1]))

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
            stack_goals = np.repeat(self.fixed_stack_goal.copy()[None], batch_size, 0)
        else:
            stack_goals = np.random.uniform(
                self.stack_goal_low,
                self.stack_goal_high,
                size=(batch_size, self.stack_goal_low.size))
        hand_goals = stack_goals + np.array([0.0, 0.0, 4.0 * self.block_radius])
        goals = np.hstack((
            np.repeat([[0.0]], batch_size, 0),
            hand_goals,
            stack_goals,
            stack_goals + np.array([[0.0, 0.0, 0.0 + 2.0 * self.block_radius]])))
        return {
            'desired_goal': goals,
            'state_desired_goal': goals}

    def compute_rewards(self, actions, obs):
        # Required by HER-TD3
        assert isinstance(obs, dict) == True
        rewards = [
            self.compute_reward(actions, {
                key: value[i] for key, value in obs.items()})[0]
            for i in range(obs["observation"].shape[0])
        ]
        return np.array(rewards)

    def compute_reward(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']

        gripper_position = achieved_goals[0]
        hand_position = achieved_goals[1:4]
        block_one_position = achieved_goals[4:7]
        block_two_position = achieved_goals[7:10]

        gripper_goal = desired_goals[0]
        hand_goal = desired_goals[1:4]
        block_one_goal = desired_goals[4:7]
        block_two_goal = desired_goals[7:10]

        block_one_distance = np.linalg.norm((block_one_position - block_one_goal), ord=2)
        block_two_distance = np.linalg.norm((block_two_position - block_two_goal), ord=2)

        block_one_stacked = block_one_distance < 0.05
        block_two_stacked = block_two_distance < 0.05

        base_reward = 1000.0
        selected_block = block_two_position
        selected_goal = block_two_goal
        if not block_one_stacked:
            selected_block = block_one_position
            selected_goal = block_one_goal
            base_reward = base_reward - 500.0
        elif not block_two_stacked:
            selected_block = block_two_position
            selected_goal = block_two_goal
            base_reward = base_reward - 500.0

        reach_distance_xy = np.linalg.norm((selected_block[:2] - hand_position[:2]), ord=2)
        reach_distance = np.linalg.norm((selected_block - hand_position),  ord=2)

        above_block = reach_distance_xy < 0.05
        arrived_to_block = reach_distance < 0.05

        target_height = 0.3
        z_distance = np.linalg.norm(hand_position[2:] - target_height, ord=2)

        if not above_block and not arrived_to_block:
            reach_reward = -reach_distance_xy - 2.0 * z_distance
        elif above_block and not arrived_to_block:
            reach_reward = -reach_distance
        else:
            reach_reward = -reach_distance + max(gripper_position, 0)

        is_grasping = (self.data.sensordata[0] > 0) and (self.data.sensordata[1] > 0)
        is_raised = z_distance < 0.01
        pick_reward = min(target_height, selected_block[2]) if is_grasping else 0

        place_distance = np.linalg.norm((selected_block - selected_goal), ord=2)

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if is_raised and is_grasping:
            place_reward = 1000 * (self.max_place_distance - place_distance) + c1 * (
                        np.exp(-(place_distance ** 2) / c2) + np.exp(-(place_distance ** 2) / c3))
            place_reward = max(place_reward, 0)
        else:
            place_reward = 0

        return (base_reward + place_reward +
                pick_reward + reach_reward)

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
        self.update_goal_markers(goal)


if __name__ == "__main__":

    import multiworld.envs.mujoco as m
    m.register_mujoco_envs()
    import gym
    x = gym.make("SawyerTwoBlocksXYZEnv-v0")
    import time
    while True:
        x.reset()
        for i in range(100):
            time.sleep(0.05)
            x.step(x.action_space.sample())
            x.render()
