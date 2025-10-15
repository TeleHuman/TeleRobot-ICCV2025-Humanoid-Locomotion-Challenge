#!/usr/bin/env python3

"""
使用 line_profiler 分析训练代码的性能瓶颈
使用方法:
1. 运行: kernprof -l -v profile_train.py --task=gr1 --num_envs=512 --headless
2. 查看结果: python -m line_profiler profile_train.py.lprof
"""

import numpy as np
import os
from datetime import datetime
import sys
import torch

# 添加路径
sys.path.append('/home/lxz/ICCV2025-Challenge/legged_gym')
sys.path.append('/home/lxz/ICCV2025-Challenge/rsl_rl')

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import wandb

# 在函数上添加 @profile 装饰器来进行分析
@profile
def train_with_profiling(args):
    """主训练函数 - 添加性能分析"""
    # args.headless = True
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + datetime.now().strftime('%b%d_%H-%M-%S--') + args.exptid + "_profile"
    
    try:
        os.makedirs(log_pth)
    except:
        pass
    
    # 禁用 wandb 以减少干扰
    wandb.init(mode="disabled")
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args)
    
    # 只运行少量迭代进行分析
    num_iterations = 5  # 减少迭代次数以便分析
    print(f"开始性能分析，运行 {num_iterations} 次迭代...")
    
    ppo_runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)

@profile 
def env_step_analysis(env, actions):
    """分析环境step方法的性能"""
    return env.step(actions)

@profile
def post_physics_step_analysis(env):
    """分析post_physics_step的性能"""
    env.post_physics_step()

@profile
def compute_reward_analysis(env):
    """分析reward计算的性能"""
    env.compute_reward()

@profile
def compute_observations_analysis(env):
    """分析observation计算的性能"""
    env.compute_observations()

@profile
def reset_analysis(env, env_ids):
    """分析环境重置的性能"""
    if len(env_ids) > 0:
        env.reset_idx(env_ids)

if __name__ == '__main__':
    # 解析参数
    args = get_args()
    
    # 设置较小的环境数量以便分析
    if not hasattr(args, 'num_envs') or args.num_envs > 1024:
        args.num_envs = 512
    
    # 确保headless模式
    args.headless = True
    args.no_wandb = True
    
    print(f"开始性能分析，环境数量: {args.num_envs}")
    print("请使用以下命令运行:")
    print("kernprof -l -v profile_train.py --task=gr1 --num_envs=512 --headless")
    
    train_with_profiling(args)
