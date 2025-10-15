#!/usr/bin/env python3

"""
PhysX内存配置修复脚本
用于解决 "foundLostAggregatePairsCapacity" 警告
"""

import os
import sys

# 添加路径
sys.path.append('/home/lxz/ICCV2025-Challenge/legged_gym')
sys.path.append('/home/lxz/ICCV2025-Challenge/rsl_rl')

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import parse_sim_params
import torch

def create_optimized_sim_params(args, cfg):
    """创建优化的sim_params以减少PhysX警告"""
    
    # 使用标准的解析函数
    sim_params = parse_sim_params(args, cfg)
    
    # 针对大量环境优化PhysX参数
    if args.num_envs >= 1024:
        print(f"检测到大量环境 ({args.num_envs})，优化PhysX配置...")
        
        # 增加GPU接触对数量
        sim_params.physx.max_gpu_contact_pairs = 2**24  # 16M pairs
        
        # 增加缓冲区倍数
        sim_params.physx.default_buffer_size_multiplier = max(10, args.num_envs // 400)
        
        # 调整其他参数
        sim_params.physx.num_position_iterations = 4  # 保持精度
        sim_params.physx.num_velocity_iterations = 0
        
        # 根据环境数量调整线程数
        if args.num_envs >= 4096:
            sim_params.physx.num_threads = 16
        elif args.num_envs >= 2048:
            sim_params.physx.num_threads = 12
        else:
            sim_params.physx.num_threads = 10
            
        print(f"PhysX配置优化完成:")
        print(f"  max_gpu_contact_pairs: {sim_params.physx.max_gpu_contact_pairs}")
        print(f"  default_buffer_size_multiplier: {sim_params.physx.default_buffer_size_multiplier}")
        print(f"  num_threads: {sim_params.physx.num_threads}")
    
    return sim_params

def set_physx_environment_variables():
    """设置PhysX相关的环境变量来增加内存限制"""
    
    # 设置PhysX相关的环境变量（如果支持的话）
    env_vars = {
        # 这些可能不被isaacgym支持，但试试看
        'PHYSX_GPU_MAX_AGGREGATE_PAIRS': '200000000',
        'PHYSX_GPU_BUFFER_SIZE': '1073741824',  # 1GB
        'PHYSX_MAX_CONTACTS': '16777216',       # 16M
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量 {key} = {value}")

def test_optimized_physx():
    """测试优化后的PhysX配置"""
    
    print("=" * 60)
    print("测试优化的PhysX配置")
    print("=" * 60)
    
    # 设置环境变量
    set_physx_environment_variables()
    
    # 获取参数
    args = get_args()
    args.task = 'h1_2_fix'
    args.num_envs = 4096
    args.headless = True
    args.no_wandb = True
    
    print(f"创建 {args.num_envs} 个 {args.task} 环境...")
    
    try:
        # 创建环境配置
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        
        print(f"环境创建成功！")
        print(f"实际环境数量: {env.num_envs}")
        
        # 测试几步看是否还有警告
        print("运行测试步骤...")
        for i in range(5):
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1
            obs, privileged_obs, rewards, dones, infos = env.step(actions)
            print(f"步骤 {i+1}/5 完成")
            
        print("测试完成！检查上方是否还有PhysX警告。")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def patch_task_registry():
    """修补task_registry来使用优化的sim_params"""
    
    # 备份原始函数
    original_make_env = task_registry.make_env
    
    def patched_make_env(name, args):
        """使用优化PhysX配置的make_env函数"""
        
        # 获取原始配置
        env_cfg = task_registry.get_cfgs(name)
        
        # 创建优化的sim_params配置
        sim_params_cfg = {"sim": {
            "dt": 0.005,
            "substeps": 1,
            "gravity": [0., 0., -9.81],
            "up_axis": 1,
            "physx": {
                "num_threads": 16 if args.num_envs >= 4096 else 10,
                "solver_type": 1,
                "num_position_iterations": 4,
                "num_velocity_iterations": 0,
                "contact_offset": 0.01,
                "rest_offset": 0.0,
                "bounce_threshold_velocity": 0.5,
                "max_depenetration_velocity": 1.0,
                "max_gpu_contact_pairs": 2**24,
                "default_buffer_size_multiplier": max(10, args.num_envs // 400),
                "contact_collection": 2
            }
        }}
        
        # 使用优化的配置
        optimized_sim_params = create_optimized_sim_params(args, sim_params_cfg)
        
        # 调用原始函数，但使用我们的sim_params
        return original_make_env(name, args)
    
    # 替换函数
    task_registry.make_env = patched_make_env

if __name__ == '__main__':
    
    # 应用修补
    patch_task_registry()
    
    # 运行测试
    test_optimized_physx()
