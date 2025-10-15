#!/usr/bin/env python3

"""
测试 h1_2_fix 环境 4096 个环境的性能
使用方法: python profile_h1_2_4096.py
"""

import cProfile
import pstats
import io
import numpy as np
import os
import sys
import time
from datetime import datetime

# 添加路径
sys.path.append('/home/lxz/ICCV2025-Challenge/legged_gym')
sys.path.append('/home/lxz/ICCV2025-Challenge/rsl_rl')

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch  # 必须在isaacgym之后导入

def setup_h1_2_environment():
    """设置 h1_2_fix 测试环境，4096个环境"""
    # 使用真实的参数解析器
    args = get_args()
    
    # 覆盖一些参数以便于性能分析
    args.num_envs = 4096  # 设置为4096个环境
    args.headless = True
    args.task = 'h1_2_fix'
    args.no_wandb = True
    
    # 创建环境
    print("创建 h1_2_fix 环境...")
    print(f"目标环境数量: {args.num_envs}")
    
    start_time = time.time()
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    creation_time = time.time() - start_time
    
    print(f"环境创建完成，实际环境数量: {env.num_envs}")
    print(f"环境创建耗时: {creation_time:.2f}秒")
    
    return env, args

def benchmark_h1_2_performance(env, num_steps=50):
    """基准测试 h1_2_fix 环境的性能"""
    print(f"\n=== h1_2_fix 环境性能基准测试 ===")
    print(f"环境数量: {env.num_envs}")
    print(f"测试步数: {num_steps}")
    
    step_times = []
    physics_times = []
    reward_times = []
    obs_times = []
    reset_times = []
    
    # 预热几步
    print("预热环境...")
    for _ in range(3):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)
    
    print(f"开始性能测试...")
    total_start = time.time()
    
    for step in range(num_steps):
        # 生成随机动作
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1
        
        # 测量完整step时间
        step_start = time.time()
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # 分别测量各个组件的时间
        if step % 10 == 0:
            # 测量post_physics_step
            physics_start = time.time()
            env.post_physics_step()
            physics_time = time.time() - physics_start
            physics_times.append(physics_time)
            
            # 测量compute_reward
            reward_start = time.time()
            env.compute_reward()
            reward_time = time.time() - reward_start
            reward_times.append(reward_time)
            
            # 测量compute_observations
            obs_start = time.time()
            env.compute_observations()
            obs_time = time.time() - obs_start
            obs_times.append(obs_time)
            
            print(f"步骤 {step}/{num_steps}: {step_time:.6f}s (物理: {physics_time:.6f}s)")
    
    total_time = time.time() - total_start
    
    # 统计信息
    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)
    max_step_time = np.max(step_times)
    min_step_time = np.min(step_times)
    
    avg_physics_time = np.mean(physics_times) if physics_times else 0
    avg_reward_time = np.mean(reward_times) if reward_times else 0
    avg_obs_time = np.mean(obs_times) if obs_times else 0
    
    print(f"\n=== 性能统计结果 ===")
    print(f"总测试时间: {total_time:.2f}秒")
    print(f"总步数: {num_steps}")
    print(f"环境数量: {env.num_envs}")
    print(f"")
    print(f"=== 每步耗时统计 ===")
    print(f"平均每步耗时: {avg_step_time:.6f}秒 (±{std_step_time:.6f})")
    print(f"最快/最慢步: {min_step_time:.6f}s / {max_step_time:.6f}s")
    print(f"")
    print(f"=== 性能指标 ===")
    print(f"理论FPS: {1/avg_step_time:.1f}")
    print(f"总环境FPS: {env.num_envs/avg_step_time:.1f}")
    print(f"每秒处理环境步数: {env.num_envs * num_steps / total_time:.1f}")
    print(f"")
    print(f"=== 组件耗时 ===")
    print(f"物理步骤后处理: {avg_physics_time:.6f}s")
    print(f"奖励计算: {avg_reward_time:.6f}s") 
    print(f"观测计算: {avg_obs_time:.6f}s")
    
    return {
        'avg_step_time': avg_step_time,
        'total_env_fps': env.num_envs/avg_step_time,
        'env_count': env.num_envs,
        'total_time': total_time
    }

def detailed_profiling(env):
    """详细的性能分析"""
    print(f"\n=== 详细性能分析 ===")
    
    actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1
    
    # 使用cProfile分析
    pr = cProfile.Profile()
    pr.enable()
    
    # 执行几次step来获得稳定的分析结果
    for _ in range(5):
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
    
    pr.disable()
    
    # 输出分析结果 - 只显示前20个最耗时的函数
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    
    print("前20个最耗时函数:")
    print(s.getvalue())

def memory_usage_analysis(env):
    """内存使用分析"""
    print(f"\n=== 内存使用分析 ===")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_cached = torch.cuda.memory_reserved() / 1024**3      # GB
        
        print(f"GPU内存已分配: {memory_allocated:.2f} GB")
        print(f"GPU内存已缓存: {memory_cached:.2f} GB")
        
        # 检查各个tensor的大小
        print(f"\n主要Tensor大小:")
        print(f"root_states: {env.root_states.shape} - {env.root_states.numel() * 4 / 1024**2:.1f} MB")
        print(f"dof_state: {env.dof_state.shape} - {env.dof_state.numel() * 4 / 1024**2:.1f} MB")
        print(f"contact_forces: {env.contact_forces.shape} - {env.contact_forces.numel() * 4 / 1024**2:.1f} MB")
        print(f"obs_buf: {env.obs_buf.shape} - {env.obs_buf.numel() * 4 / 1024**2:.1f} MB")
        if hasattr(env, 'privileged_obs_buf') and env.privileged_obs_buf is not None:
            print(f"privileged_obs_buf: {env.privileged_obs_buf.shape} - {env.privileged_obs_buf.numel() * 4 / 1024**2:.1f} MB")

def run_h1_2_benchmark():
    """运行 h1_2_fix 环境的完整基准测试"""
    print("=" * 80)
    print("h1_2_fix 环境 4096 个环境性能测试")
    print("=" * 80)
    
    try:
        # 设置环境
        env, args = setup_h1_2_environment()
        
        # 内存使用分析
        memory_usage_analysis(env)
        
        # 基准性能测试
        results = benchmark_h1_2_performance(env, num_steps=30)
        
        # 详细性能分析
        detailed_profiling(env)
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        
        # 总结关键指标
        print(f"\n=== 关键性能指标总结 ===")
        print(f"环境数量: {results['env_count']}")
        print(f"平均每步耗时: {results['avg_step_time']:.6f}秒")
        print(f"总环境FPS: {results['total_env_fps']:.1f}")
        print(f"单环境等效FPS: {results['total_env_fps']/results['env_count']:.1f}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_h1_2_benchmark()
