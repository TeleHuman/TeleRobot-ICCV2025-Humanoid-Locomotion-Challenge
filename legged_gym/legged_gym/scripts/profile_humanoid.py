#!/usr/bin/env python3

"""
专门分析 HumanoidRobot 类性能的脚本
使用方法:
1. 安装 line_profiler: pip install line_profiler
2. 运行分析: kernprof -l -v profile_humanoid.py
3. 查看详细结果: python -m line_profiler profile_humanoid.py.lprof
"""

import numpy as np
import os
import sys
import time
from datetime import datetime
import torch

# 添加路径
sys.path.append('/home/lxz/ICCV2025-Challenge/legged_gym')
sys.path.append('/home/lxz/ICCV2025-Challenge/rsl_rl')

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math import quat_rotate_inverse

def setup_environment():
    """设置测试环境"""
    # 创建简单的参数
    class SimpleArgs:
        def __init__(self):
            self.task = 'gr1'
            self.num_envs = 256  # 较小的环境数量便于分析
            self.headless = True
            self.no_wandb = True
            self.device = 'cuda:0'
            self.proj_name = 'profile_test'
            self.exptid = 'performance_analysis'
    
    args = SimpleArgs()
    
    # 创建环境
    print("创建环境...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"环境创建完成，环境数量: {env.num_envs}")
    
    return env, args

@profile
def step_analysis(env, num_steps=50):
    """分析环境step方法的详细性能"""
    print(f"开始分析 {num_steps} 步的性能...")
    
    start_time = time.time()
    
    for step in range(num_steps):
        # 生成随机动作
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        
        # 执行step
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        
        if step % 10 == 0:
            print(f"完成步骤 {step}/{num_steps}")
    
    total_time = time.time() - start_time
    print(f"总耗时: {total_time:.3f}秒")
    print(f"平均每步耗时: {total_time/num_steps:.6f}秒")
    print(f"FPS: {num_steps/total_time:.1f}")

@profile
def post_physics_step_detailed(env):
    """详细分析post_physics_step方法"""
    # 刷新tensors
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    
    env.episode_length_buf += 1
    env.phase_length_buf += 1 
    env.common_step_counter += 1

    # 准备数量
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.base_lin_acc = (env.root_states[:, 7:10] - env.last_root_vel[:, :3]) / env.dt

@profile
def compute_reward_detailed(env):
    """详细分析compute_reward方法"""
    env.rew_buf[:] = 0.
    for i in range(len(env.reward_functions)):
        name = env.reward_names[i]
        rew = env.reward_functions[i]() * env.reward_scales[name]
        env.rew_buf += rew
        if name != "success_rate" and name != "complete_rate":
            env.episode_sums[name] += rew

@profile
def compute_observations_detailed(env):
    """详细分析compute_observations方法"""
    # 获取相位信息
    phase = env._get_phase()
    
    # 计算接触信息
    contact_mask = env.contact_forces[:, env.feet_indices, 2] > 5 
    
    # IMU观测
    imu_obs = torch.stack((env.roll, env.pitch), dim=1)
    
    # 构建观测向量
    obs_buf = torch.cat((
        env.base_ang_vel * env.obs_scales.ang_vel,
        imu_obs,
        0*env.delta_yaw[:, None],
        env.delta_yaw[:, None],
        env.delta_next_yaw[:, None],
        0*env.commands[:, 0:2],
        env.commands[:, 0:1],
        (env.env_class != 17).float()[:, None],
        (env.env_class == 17).float()[:, None],
        (env.dof_pos - env.default_dof_pos_all) * env.obs_scales.dof_pos,
        env.dof_vel * env.obs_scales.dof_vel,
        env.action_history_buf[:, -1],
        env.contact_filt.float()-0.5,
    ), dim=-1)

@profile
def reset_analysis_detailed(env):
    """分析环境重置的性能"""
    # 模拟一些环境需要重置
    reset_envs = torch.randint(0, 2, (env.num_envs,), device=env.device).bool()
    env_ids = reset_envs.nonzero(as_tuple=False).flatten()
    
    if len(env_ids) > 0:
        print(f"重置 {len(env_ids)} 个环境")
        env.reset_idx(env_ids)

def run_detailed_analysis():
    """运行详细的性能分析"""
    print("=" * 60)
    print("开始详细性能分析")
    print("=" * 60)
    
    # 设置环境
    env, args = setup_environment()
    
    # 让环境运行几步以初始化
    print("初始化环境...")
    for _ in range(5):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)
    
    print("开始性能分析...")
    
    # 1. 分析完整的step过程
    step_analysis(env, num_steps=20)
    
    print("分析完成！")
    print("要查看详细结果，请运行: python -m line_profiler profile_humanoid.py.lprof")

if __name__ == '__main__':
    try:
        run_detailed_analysis()
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
