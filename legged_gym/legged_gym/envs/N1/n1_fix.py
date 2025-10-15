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

first_stage=False

class N1FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.72]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "left_hip_roll_joint": 0.,
            "left_hip_yaw_joint": 0.,
            "left_hip_pitch_joint": 0.,
            "left_knee_pitch_joint": 0.,
            "left_ankle_pitch_joint": 0.,
            "left_ankle_roll_joint": 0.,
            "right_hip_roll_joint": 0.,
            "right_hip_yaw_joint": 0.,
            "right_hip_pitch_joint": 0.,
            "right_knee_pitch_joint": 0.,
            "right_ankle_pitch_joint": 0.,
            "right_ankle_roll_joint": 0.,
            'torso_joint' : 0.
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        n_scan = 132
        n_priv = 3 + 3 + 3 
        n_priv_latent = 4 + 1 + 12 + 12 
        n_proprio = 43+3 
        n_proprio_priv = 49+12 
        history_len = 10
        num_observations = n_proprio + n_scan 
        # num_observations = n_proprio_priv + n_scan + history_len*n_proprio + n_priv_latent + n_priv 

        num_privileged_obs = 839 #709 +4 # 4 for contact & stance mask  #731

        num_actions = 12
        env_spacing = 3.

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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N1/N1_rotor.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N1/N1_rotor_origin.urdf'
        name = "N1"
        foot_name = "foot_roll"
        knee_name = "shank"
        penalize_contacts_on = ["thigh", "shank"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.6, 0.8]  # min max [m/s]
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
                tracking_lin_vel = 2.0
                tracking_ang_vel = 0.8
                tracking_goal_vel = 5 # 15 # 1.5
                tracking_yaw = 2 # 7 # 0.7
                lin_vel_z = -2.0
                ang_vel_xy = -0.05
                orientation = -2.0
                torques = -0.00001
                dof_vel = -1e-3 # -0.
                dof_acc = -2.5e-7
                base_height = -0. 
                feet_air_time =  1.0
                collision = -1.
                feet_stumble = -1.0 
                action_rate = -0.01
                n1_hip_joint_deviation = -2.0
                stand_still = -0.
                feet_contact_number = 2.0 
                single_foot_contact = 2.0
                # 先训练一个第一阶段的 -- 不要phase（priv的obs和rewards）*********** TODO
                feet_distance = 0.2
                knee_distance = 0.2

                # feet_clearance = 1.0
                joint_hip_pitch = 1.5
                joint_knee = 1.5
                joint_ankle_pitch = 0.8
                # 约束脚踝不要过分运动
                # ankle_torque = -1e-02 #-5e-6 #-5e-5
                # ankle_action_rate = -0.02 #-0.005 #-0.02

            ######################
            if not first_stage:
                termination = -0.0
                tracking_lin_vel = 3.0
                # tracking_goal_vel = 2.0 #2.5
                tracking_yaw = 2.0 #1.2  
                lin_vel_z = -0.5 
                ang_vel_xy = -0.05 
                orientation = -2.0
                torques = -0.00001
                dof_acc = -2.5e-8
                # base_height = -0. 
                n1_hip_joint_deviation = -0.8 /2 
                collision = -10.
                feet_stumble = -2.0
                action_rate = -0.01
                feet_edge = -3.0
                stuck = -2.0

    

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        is_play = False

class N1FixCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'n1_fix'
        max_iterations = 1000001 # number of policy updates
        save_interval = 100

    class estimator(LeggedRobotCfgPPO.estimator):
        train_with_estimated_states = False
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = N1FixCfg.env.n_priv
        num_prop = N1FixCfg.env.n_proprio
        num_scan = N1FixCfg.env.n_scan

