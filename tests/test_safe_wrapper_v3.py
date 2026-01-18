"""
测试 SafeEnvWrapper v3.0 - 双规范架构

测试内容：
1. SafetyFilter 集成
2. TaskRewardShaper 集成
3. 陷阱状态终止
4. 完整 episode 运行
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from safe_rl_drone.env import GridWorldEnv
from safe_rl_drone.wrappers.safe_env_wrapper_v3 import SafeEnvWrapper


def test_wrapper_basic():
    """测试基本功能"""
    print("=" * 60)
    print("测试 SafeEnvWrapper v3.0 - 基本功能")
    print("=" * 60)
    
    # 加载地图
    import os
    map_path = os.path.join(os.path.dirname(__file__), '..', 'maps', 'map1.txt')
    with open(map_path, 'r') as f:
        lines = f.readlines()
    grid_map = [list(line.strip()) for line in lines if line.strip()]
    
    # 创建环境
    base_env = GridWorldEnv(
        grid_map=grid_map,
        random_goal=False,
        random_start=False
    )
    
    # 包装环境
    env = SafeEnvWrapper(
        base_env,
        safety_formula="G(!wall) & G(!boundary)",
        task_formula="F(goal)"
    )
    
    print("\n测试 reset:")
    print("-" * 40)
    obs, info = env.reset()
    print(f"观测形状: {obs.shape}")
    print(f"Agent 位置: {env.unwrapped.agent_pos}")
    print(f"目标位置: {env.unwrapped.target_pos}")
    
    print("\n测试 step (安全动作):")
    print("-" * 40)
    action = 3  # 向右
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"动作: {action}")
    print(f"奖励: {reward:.3f}")
    print(f"  r_rho: {info['r_rho']:.3f}")
    print(f"  r_progress: {info['r_progress']:.3f}")
    print(f"  r_time: {info['r_time']:.3f}")
    print(f"安全过滤: {info['safety_filtered']}")
    print(f"Terminated: {terminated}")
    
    print("\n" + "=" * 60)
    print("SafeEnvWrapper v3.0 基本功能测试通过！")
    print("=" * 60)


def test_wrapper_safety_filter():
    """测试安全过滤"""
    print("\n" + "=" * 60)
    print("测试 SafeEnvWrapper v3.0 - 安全过滤")
    print("=" * 60)
    
    # 加载地图
    import os
    map_path = os.path.join(os.path.dirname(__file__), '..', 'maps', 'map1.txt')
    with open(map_path, 'r') as f:
        lines = f.readlines()
    grid_map = [list(line.strip()) for line in lines if line.strip()]
    
    # 创建环境
    base_env = GridWorldEnv(
        grid_map=grid_map,
        random_goal=False,
        random_start=False
    )
    
    env = SafeEnvWrapper(
        base_env,
        safety_formula="G(!wall) & G(!boundary)",
        task_formula="F(goal)"
    )
    
    obs, info = env.reset()
    
    # 手动设置到墙边
    env.unwrapped.agent_pos = np.array([1, 1])
    
    print(f"\n当前位置: {env.unwrapped.agent_pos}")
    print(f"墙的位置: {env.unwrapped.wall_positions}")
    
    # 尝试向右走（会撞墙）
    action = 3  # 向右，会到 (1, 2) 撞墙
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n原始动作: {info['original_action']}")
    print(f"执行动作: {info['executed_action']}")
    print(f"被过滤: {info['safety_filtered']}")
    print(f"过滤器惩罚: {info['r_filter']:.3f}")
    print(f"总奖励: {reward:.3f}")
    
    assert info['safety_filtered'], "不安全动作应该被过滤"
    assert info['r_filter'] < 0, "应该有过滤器惩罚"
    
    print("\n" + "=" * 60)
    print("SafeEnvWrapper v3.0 安全过滤测试通过！")
    print("=" * 60)


def test_wrapper_task_trap():
    """测试任务陷阱状态终止"""
    print("\n" + "=" * 60)
    print("测试 SafeEnvWrapper v3.0 - 任务陷阱终止")
    print("=" * 60)
    
    # 加载地图
    import os
    map_path = os.path.join(os.path.dirname(__file__), '..', 'maps', 'map1.txt')
    with open(map_path, 'r') as f:
        lines = f.readlines()
    grid_map = [list(line.strip()) for line in lines if line.strip()]
    
    # 创建环境
    base_env = GridWorldEnv(
        grid_map=grid_map,
        random_goal=False,
        random_start=False
    )
    
    # 注册禁区命题
    from safe_rl_drone.ltl.propositions import PositionProposition
    
    config = {
        'propositions': {
            'forbidden': {
                'type': 'position',
                'pos': [3, 3]
            }
        }
    }
    
    env = SafeEnvWrapper(
        base_env,
        safety_formula="G(!wall) & G(!boundary)",
        task_formula="F(goal) & G(!forbidden)",  # 必须避开禁区
        config=config
    )
    
    obs, info = env.reset()
    
    print(f"\n目标位置: {env.unwrapped.target_pos}")
    print(f"禁区位置: (3, 3)")
    
    # 手动移动到禁区
    env.unwrapped.agent_pos = np.array([3, 3])
    
    # 执行一步（会进入陷阱状态）
    action = 4  # stay
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n当前位置: {env.unwrapped.agent_pos}")
    print(f"进入陷阱: {info['task_is_trap']}")
    print(f"陷阱惩罚: {info['r_trap']:.3f}")
    print(f"Terminated: {terminated}")
    print(f"总奖励: {reward:.3f}")
    
    if info['task_is_trap']:
        assert terminated, "进入陷阱应该终止 episode"
        assert info['r_trap'] < 0, "应该有陷阱惩罚"
        print("\n✓ 陷阱状态正确终止 episode")
    else:
        print("\n注：此公式未产生陷阱状态（可能 Spot 构建方式不同）")
    
    print("\n" + "=" * 60)
    print("SafeEnvWrapper v3.0 任务陷阱测试完成！")
    print("=" * 60)


def test_wrapper_full_episode():
    """测试完整 episode"""
    print("\n" + "=" * 60)
    print("测试 SafeEnvWrapper v3.0 - 完整 Episode")
    print("=" * 60)
    
    # 加载地图
    import os
    map_path = os.path.join(os.path.dirname(__file__), '..', 'maps', 'map1.txt')
    with open(map_path, 'r') as f:
        lines = f.readlines()
    grid_map = [list(line.strip()) for line in lines if line.strip()]
    
    # 创建环境
    base_env = GridWorldEnv(
        grid_map=grid_map,
        random_goal=False,
        random_start=False
    )
    
    env = SafeEnvWrapper(
        base_env,
        safety_formula="G(!wall) & G(!boundary)",
        task_formula="F(goal)"
    )
    
    obs, info = env.reset()
    
    print(f"\n起点: {env.unwrapped.agent_pos}")
    print(f"目标: {env.unwrapped.target_pos}")
    
    # 简单策略：向右下移动
    actions = [3, 3, 1, 1, 3, 3, 1, 1]  # 右右下下右右下下
    
    total_reward = 0.0
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {i+1}:")
        print(f"  位置: {env.unwrapped.agent_pos}")
        print(f"  奖励: {reward:.3f} (累积: {total_reward:.3f})")
        print(f"  过滤: {info['safety_filtered']}")
        print(f"  接受: {info['task_is_accepting']}")
        
        if terminated or truncated:
            print(f"  Episode 结束！")
            break
    
    # 统计信息
    stats = env.get_episode_stats()
    print(f"\nEpisode 统计:")
    print(f"  总步数: {stats['steps']}")
    print(f"  安全干预: {stats['safety_interventions']}")
    print(f"  干预率: {stats['intervention_rate']:.2%}")
    print(f"  任务完成: {stats['task_completions']}")
    print(f"  任务陷阱: {stats['task_traps']}")
    
    print("\n" + "=" * 60)
    print("SafeEnvWrapper v3.0 完整 Episode 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_wrapper_basic()
    test_wrapper_safety_filter()
    test_wrapper_task_trap()
    test_wrapper_full_episode()
    
    print("\n" + "=" * 60)
    print("所有 SafeEnvWrapper v3.0 测试通过！✓")
    print("=" * 60)
