# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

first_stage=True

class G1FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):

        pos = [0.0, 0.0, 0.78]  # x,y,z [m]
        
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.1,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        n_scan = 132
        n_priv = 3 + 3 + 3 # = 9 base velocity 3个
        # n_priv_latent = 4 + 1 + 12 +12
        n_priv_latent = 4 + 1 + 12 + 12 # mass, fraction, motor strength1 and 2
        
        n_proprio = 51 # 所有本体感知信息，即obs_buf
        history_len = 10

        # num obs = 53+132+10*53+43+9 = 187+47+530+43+9 = 816
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_privileged_obs = num_observations+4
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 

        contact_buf_len = 100

    class depth( LeggedRobotCfg.depth ):
        position = [0.1, 0, 0.77]  # front camera
        angle = [-5, 5]  # positive pitch down
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
        name = "g1_fix_upper"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.1, 0.6]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]
        cycletime = 0.02 * 40 # frequence * frames

  
    class rewards:
        min_dist = 0.2
        max_dist = 0.40
        # high_knees_target = -0.20 # 0.75 - 0.95
        # high_feet_target = -0.45  # 0.50 - 0.95
        # squat_height_target = 0.82 # h1: 0.75
        feet_min_lateral_distance_target = 0.22 # [h1] 0.22   # [g1]0.17 
        class scales:
            if first_stage:
                # [NOTE]  first stage
                termination = -0.0
                # tracking_lin_vel = 2.0
                # tracking_ang_vel = 0.8
                tracking_goal_vel = 2.5#1.5
                tracking_yaw = 1.2 # 0.7
                lin_vel_z = -2.0
                ang_vel_xy = -0.05
                orientation = -2.0
                torques = -0.00001
                dof_vel = -1e-3 # -0.
                dof_acc = -2.5e-7
                base_height = -0. 
                feet_air_time =  5.0
                collision = -1.
                feet_stumble = -1.0 
                action_rate = -0.01
                g1_hip_joint_deviation = -1.2

                feet_contact_number = 4.0 
                # feet_distance = 0.2
                # knee_distance = 0.2

            ######################
            if not first_stage:
                termination = -0.0
                tracking_goal_vel = 1.5
                tracking_yaw = 0.7
                lin_vel_z = -0.5
                ang_vel_xy = -0.05
                orientation = -2.0
                torques = -0.00001
                # dof_vel = -1e-3 # -0.
                dof_acc = -2.5e-8
                base_height = -0. 
                # feet_air_time =  1.0


                g1_hip_joint_deviation = -0.5
                # dof_pos_limits = -2.0
                # dof_vel_limits = -1.0
                # torque_limits = -1.0
                
                # feet_slippage = -0.25
                # feet_contact_force = -2.5e-4
                
                collision = -10.
                feet_stumble = -1.5 
                action_rate = -0.01
                # feet_contact_number = 0.0 

                feet_edge = -1.0

                # encourage higher 
                # high_knees_height = 2
                # high_feet_height = 2
                # base_height = 2.0
                # feet_lateral_distance = 0.5 # 2 # 0.5

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        is_play = False

class G1FixCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'g1_fix'
        experiment_name = 'g1_fix'
        max_iterations = 50001 # number of policy updates
        save_interval = 500

    class estimator(LeggedRobotCfgPPO.estimator):
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = G1FixCfg.env.n_priv
        num_prop = G1FixCfg.env.n_proprio
        num_scan = G1FixCfg.env.n_scan

