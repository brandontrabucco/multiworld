from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.envs.mujoco.sawyer_xyz.sawyer_two_blocks import SawyerTwoBlocksXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_block_stacking_camera


class SawyerTwoBlocksShelfXYZEnv(SawyerTwoBlocksXYZEnv):
    def __init__(
            self,

            block_low=(-0.2, 0.65, 0.15),
            block_high=(0.2, 0.75, 0.15),

            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.65, 0.3),

            stack_goal_low=(-0.2, 0.55, 0.02),
            stack_goal_high=(0.2, 0.65, 0.02),

            hand_goal_low=(-0.2, 0.55, 0.3),
            hand_goal_high=(0.2, 0.65, 0.3),

            fix_goal=False,
            fixed_stack_goal=(0.0, 0.55, 0.02),
            fixed_hand_goal=(0.0, 0.75, 0.3),

            use_sparse_reward=False,
            sparse_reward_threshold=0.05,

            reset_free=False,
            hide_goal_markers=False,
            oracle_reset_prob=0.0,

            xml_path='sawyer_xyz/two_blocks_shelf.xml',

            **kwargs
    ):
        super(SawyerTwoBlocksShelfXYZEnv, self).__init__(
            block_low=block_low,
            block_high=block_high,

            hand_low=hand_low,
            hand_high=hand_high,

            stack_goal_low=stack_goal_low,
            stack_goal_high=stack_goal_high,

            hand_goal_low=hand_goal_low,
            hand_goal_high=hand_goal_high,

            fix_goal=fix_goal,
            fixed_stack_goal=fixed_stack_goal,
            fixed_hand_goal=fixed_hand_goal,

            use_sparse_reward=use_sparse_reward,
            sparse_reward_threshold=sparse_reward_threshold,

            reset_free=reset_free,
            hide_goal_markers=hide_goal_markers,
            oracle_reset_prob=oracle_reset_prob,

            xml_path=xml_path,

            **kwargs
        )


if __name__ == "__main__":

    import multiworld.envs.mujoco as m
    m.register_mujoco_envs()
    import gym
    x = gym.make("SawyerTwoBlocksShelfXYZEnv-v0")
    import time
    while True:
        x.reset()
        for i in range(100):
            time.sleep(0.05)
            x.step(x.action_space.sample())
            x.render()
