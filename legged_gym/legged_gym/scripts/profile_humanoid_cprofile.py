#!/usr/bin/env python3

"""
使用Python内置的cProfile分析 HumanoidRobot 类性能
使用方法: python profile_humanoid_cprofile.py
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

def setup_environment():
    """设置测试环境"""
    # 使用真实的参数解析器
    args = get_args()
    
    # 覆盖一些参数以便于性能分析
    args.num_envs = 256  # 较小的环境数量便于分析
    args.headless = True
    args.task = 'gr1'
    
    # 创建环境
    print("创建环境...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"环境创建完成，环境数量: {env.num_envs}")
    
    return env, args

def step_analysis(env, num_steps=30):
    """分析环境step方法的详细性能"""
    print(f"开始分析 {num_steps} 步的性能...")
    
    step_times = []
    
    for step in range(num_steps):
        # 生成随机动作
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        
        # 测量单步时间
        start_time = time.time()
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        step_time = time.time() - start_time
        step_times.append(step_time)
        
        if step % 10 == 0:
            print(f"完成步骤 {step}/{num_steps}, 用时: {step_time:.6f}秒")
    
    # 统计信息
    avg_time = np.mean(step_times)
    std_time = np.std(step_times)
    max_time = np.max(step_times)
    min_time = np.min(step_times)
    
    print(f"\n=== 性能统计 ===")
    print(f"总步数: {num_steps}")
    print(f"平均每步耗时: {avg_time:.6f}秒 (±{std_time:.6f})")
    print(f"最快/最慢: {min_time:.6f}s / {max_time:.6f}s")
    print(f"理论FPS: {1/avg_time:.1f}")
    print(f"总环境FPS: {env.num_envs/avg_time:.1f}")

def profile_main_functions(env):
    """使用cProfile分析主要函数"""
    print("\n=== 开始详细性能分析 ===")
    
    # 1. 分析一次完整的step
    actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
    
    pr = cProfile.Profile()
    pr.enable()
    
    # 执行多次step来获得稳定的分析结果
    for _ in range(5):
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
    
    pr.disable()
    
    # 输出分析结果
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # 显示前30个最耗时的函数
    
    print("前30个最耗时函数:")
    print(s.getvalue())
    
    # 2. 专门分析HumanoidRobot的方法
    print("\n=== HumanoidRobot 方法分析 ===")
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2)
    ps2.sort_stats('cumulative')
    ps2.print_stats('humanoid_robot')  # 只显示包含humanoid_robot的函数
    print(s2.getvalue())
    
    # 3. 分析torch相关操作
    print("\n=== PyTorch 操作分析 ===")
    s3 = io.StringIO()
    ps3 = pstats.Stats(pr, stream=s3)
    ps3.sort_stats('cumulative')
    ps3.print_stats('torch')  # 只显示torch相关函数
    print(s3.getvalue())

def analyze_individual_methods(env):
    """分析各个方法的耗时"""
    print("\n=== 单独方法分析 ===")
    
    # 准备数据
    actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
    
    # 分析post_physics_step
    print("分析 post_physics_step...")
    times = []
    for _ in range(10):
        start = time.time()
        env.post_physics_step()
        times.append(time.time() - start)
    print(f"post_physics_step 平均耗时: {np.mean(times):.6f}s (±{np.std(times):.6f})")
    
    # 分析compute_reward
    print("分析 compute_reward...")
    times = []
    for _ in range(10):
        start = time.time()
        env.compute_reward()
        times.append(time.time() - start)
    print(f"compute_reward 平均耗时: {np.mean(times):.6f}s (±{np.std(times):.6f})")
    
    # 分析compute_observations
    print("分析 compute_observations...")
    times = []
    for _ in range(10):
        start = time.time()
        env.compute_observations()
        times.append(time.time() - start)
    print(f"compute_observations 平均耗时: {np.mean(times):.6f}s (±{np.std(times):.6f})")
    
    # 分析check_termination
    print("分析 check_termination...")
    times = []
    for _ in range(10):
        start = time.time()
        env.check_termination()
        times.append(time.time() - start)
    print(f"check_termination 平均耗时: {np.mean(times):.6f}s (±{np.std(times):.6f})")

def run_comprehensive_analysis():
    """运行全面的性能分析"""
    print("=" * 60)
    print("开始全面性能分析")
    print("=" * 60)
    
    # 设置环境
    env, args = setup_environment()
    
    # 让环境运行几步以初始化
    print("初始化环境...")
    for _ in range(3):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)
    
    # 1. 基本性能测试
    step_analysis(env, num_steps=20)
    
    # 2. 详细的函数分析
    profile_main_functions(env)
    
    # 3. 单独方法分析
    analyze_individual_methods(env)
    
    print("\n" + "=" * 60)
    print("性能分析完成！")
    print("=" * 60)

if __name__ == '__main__':
    try:
        run_comprehensive_analysis()
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
