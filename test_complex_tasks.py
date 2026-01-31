"""
测试复杂任务规范

测试三种复杂任务：
1. 顺序任务：wp1 → wp2 → goal
2. Until 任务：避开危险区直到到达安全区
3. 多目标选择：goal1 | goal2 | goal3
"""

import yaml
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

from safe_rl_drone.env import GridWorldEnv
from safe_rl_drone.wrappers import SafeEnvWrapper
from utils import load_map_from_file


def test_task(config_file, task_name):
    """测试单个任务"""
    print("\n" + "=" * 60)
    print(f"测试任务: {task_name}")
    print("=" * 60)
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n任务公式: {config['specifications']['task']['formula']}")
    
    # 加载地图
    grid_map = load_map_from_file(config['environment']['map_path'])
    
    # 创建环境
    base_env = GridWorldEnv(
        grid_map=grid_map,
        view_size=config['environment']['view_size'],
        random_goal=config['environment']['random_goal'],
        random_start=config['environment']['random_start']
    )
    
    # 包装环境
    safety_formula = config['specifications']['safety']['formula']
    task_formula = config['specifications']['task']['formula']
    
    wrapper_config = {
        'propositions': config.get('propositions', {}),
        'reward_weights': config['reward'].get('task_shaping_weights', {})
    }
    
    env = SafeEnvWrapper(
        base_env,
        safety_formula=safety_formula,
        task_formula=task_formula,
        config=wrapper_config
    )
    
    # 检查 FSA 信息（在 TimeLimit 包装之前）
    print(f"\nFSA 信息:")
    print(f"  状态数: {env.task_reward_shaper.fsa_monitor.fsa.num_states}")
    print(f"  接受状态: {env.task_reward_shaper.fsa_monitor.fsa.accepting_states}")
    print(f"  陷阱状态: {env.task_reward_shaper.fsa_monitor.fsa.trap_states}")
    print(f"  状态价值: {env.task_reward_shaper.state_values}")
    
    # 包装 TimeLimit（在检查 FSA 信息之后）
    env = TimeLimit(env, max_episode_steps=config['environment']['max_episode_steps'])
    
    # 快速训练
    print(f"\n快速训练 (100000 步)...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    
    # 测试
    print(f"\n测试 5 个 episodes:")
    successes = 0
    total_steps = 0
    total_traps = 0
    
    for ep in range(5):
        obs, info = env.reset()
        terminated = False
        truncated = False
        steps = 0
        
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.item())
            steps += 1
            
            if info.get('task_is_trap', False):
                total_traps += 1
                break
        
        total_steps += steps
        
        if info.get('task_is_accepting', False):
            successes += 1
            status = "✓ 完成"
        elif info.get('task_is_trap', False):
            status = "✗ 陷阱"
        else:
            status = "✗ 超时"
        
        print(f"  Episode {ep+1}: {steps} 步 {status}")
    
    # 统计
    print(f"\n结果:")
    print(f"  成功率: {successes}/5")
    print(f"  平均步数: {total_steps/5:.1f}")
    print(f"  陷阱次数: {total_traps}")
    
    env.close()
    
    return successes, total_steps / 3, total_traps


def main():
    """测试所有复杂任务"""
    print("=" * 60)
    print("复杂任务规范测试")
    print("=" * 60)
    
    tasks = [
        ("config_multi_goal.yaml", "多目标选择 (goal1 | goal2 | goal3)"),
        ("config_sequential.yaml", "顺序任务 (wp1 → wp2 → goal)"),
        ("config_until.yaml", "简化 Until 任务 (F(safe_zone))")
    ]
    
    results = []
    
    for config_file, task_name in tasks:
        try:
            success, avg_steps, traps = test_task(config_file, task_name)
            results.append((task_name, success, avg_steps, traps))
        except Exception as e:
            print(f"\n✗ 任务失败: {e}")
            results.append((task_name, 0, 0, 0))
    
    # 总结
    print("\n" + "=" * 60)
    print("所有任务测试总结")
    print("=" * 60)
    
    for task_name, success, avg_steps, traps in results:
        print(f"\n{task_name}:")
        print(f"  成功率: {success}/5")
        print(f"  平均步数: {avg_steps:.1f}")
        print(f"  陷阱次数: {traps}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
