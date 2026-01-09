#!/usr/bin/env python3
"""
安全包装器测试

测试内容：
1. SafetyMonitor 安全监控
2. ActionFilter 动作过滤
3. SafeEnvWrapper 完整集成
"""

import sys
import os
import numpy as np

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_map_from_file


def test_safety_monitor():
    """测试安全监控器"""
    print("\n" + "=" * 60)
    print("1. 测试 SafetyMonitor")
    print("=" * 60)
    
    from safe_rl_drone.safety.monitor import SafetyMonitor
    
    # 模拟环境信息
    env_info = {
        'wall_positions': {(0, 0), (0, 1), (1, 0)},
        'goal_position': (4, 4),
        'grid_size': (6, 6)
    }
    
    # 创建监控器
    monitor = SafetyMonitor(
        safety_formula="G(!wall)",
        env_info=env_info
    )
    
    # 测试状态安全性
    test_states = [
        (np.array([2, 2]), "安全位置"),
        (np.array([0, 0]), "墙上"),
        (np.array([4, 4]), "目标"),
    ]
    
    print("\n状态安全性测试:")
    for state, desc in test_states:
        is_safe, rho = monitor.is_state_safe(state)
        print(f"  {desc} {tuple(state)}: 安全={is_safe}, rho={rho:.2f}")
    
    # 测试动作安全性
    action_map = {
        0: np.array([-1, 0]),  # 上
        1: np.array([1, 0]),   # 下
        2: np.array([0, -1]),  # 左
        3: np.array([0, 1]),   # 右
        4: np.array([0, 0]),   # 不动
    }
    
    print("\n动作安全性测试 (从位置 (1, 1)):")
    current = np.array([1, 1])
    action_names = ['上', '下', '左', '右', '不动']
    for action in range(5):
        is_safe, rho, reason = monitor.is_action_safe(current, action, action_map)
        print(f"  动作 {action}({action_names[action]}): 安全={is_safe}, rho={rho:.2f}, 原因={reason}")
    
    print("\n[OK] SafetyMonitor 测试通过")
    return monitor, action_map


def test_action_filter(monitor, action_map):
    """测试动作过滤器"""
    print("\n" + "=" * 60)
    print("2. 测试 ActionFilter")
    print("=" * 60)
    
    from safe_rl_drone.safety.action_filter import ActionFilter
    
    # 创建过滤器
    action_filter = ActionFilter(
        safety_monitor=monitor,
        action_map=action_map,
        num_actions=5
    )
    
    # 测试动作过滤
    print("\n动作过滤测试 (从位置 (1, 1)):")
    current = np.array([1, 1])
    action_names = ['上', '下', '左', '右', '不动']
    
    for action in range(5):
        filtered, was_modified, info = action_filter.filter_action(current, action)
        
        if was_modified:
            print(f"  动作 {action}({action_names[action]}) -> "
                  f"{filtered}({action_names[filtered]}) [干预: {info.get('reason', '')}]")
        else:
            print(f"  动作 {action}({action_names[action]}) -> 通过")
    
    # 获取安全动作列表
    safe_actions = action_filter.get_safe_actions(current)
    print(f"\n安全动作列表: {safe_actions}")
    
    # 统计
    stats = action_filter.get_stats()
    print(f"干预统计: {stats}")
    
    print("\n[OK] ActionFilter 测试通过")


def test_safe_env_wrapper():
    """测试完整的安全环境包装器"""
    print("\n" + "=" * 60)
    print("3. 测试 SafeEnvWrapper")
    print("=" * 60)
    
    from safe_rl_drone.env import GridWorldEnv
    from safe_rl_drone.wrappers.safe_env_wrapper import SafeEnvWrapper
    
    # 加载地图
    grid_map = load_map_from_file("maps/map1.txt")
    
    # 创建环境
    base_env = GridWorldEnv(grid_map, view_size=5)
    
    # 包装环境
    env = SafeEnvWrapper(
        base_env,
        safety_formula="G(!wall)",
        unsafe_penalty=-1.0,
        use_robustness_reward=True,
        robustness_weight=0.1
    )
    
    # 运行一个 episode
    print("\n运行测试 Episode:")
    obs, info = env.reset()
    agent_pos = env.unwrapped.agent_pos
    print(f"初始位置: {tuple(agent_pos)}")
    print(f"观测维度: {obs.shape} (局部视野 5x5 + 目标方向 2)")
    
    total_reward = 0
    interventions = 0
    action_names = ['上', '下', '左', '右', '不动']
    
    # 执行一些动作
    actions = [3, 3, 1, 1, 3, 3, 1, 1, 3, 1]  # 尝试到达目标
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent_pos = env.unwrapped.agent_pos
        
        executed = info.get('executed_action', action)
        
        status = ""
        if info.get('safety_intervention', False):
            status = f" [干预: {info.get('intervention_reason', '')}]"
            interventions += 1
        
        print(f"  步骤 {i+1}: 动作={action_names[action]}, "
              f"执行={action_names[executed]}, "
              f"位置={tuple(agent_pos)}, "
              f"奖励={reward:.2f}, "
              f"rho={info.get('robustness', 0):.2f}{status}")
        
        if terminated:
            print(f"  到达目标!")
            break
    
    # 获取统计
    stats = env.get_episode_stats()
    print(f"\nEpisode 统计:")
    print(f"  总步数: {stats['steps']}")
    print(f"  干预次数: {stats['interventions']}")
    print(f"  干预率: {stats['intervention_rate']:.2%}")
    print(f"  平均 rho: {stats['avg_robustness']:.2f}")
    print(f"  轨迹 rho: {stats['trajectory_robustness']:.2f}")
    
    env.close()
    print("\n[OK] SafeEnvWrapper 测试通过")


def test_unsafe_actions():
    """测试不安全动作的处理"""
    print("\n" + "=" * 60)
    print("4. 测试不安全动作处理")
    print("=" * 60)
    
    from safe_rl_drone.env import GridWorldEnv
    from safe_rl_drone.wrappers.safe_env_wrapper import SafeEnvWrapper
    
    # 加载地图
    grid_map = load_map_from_file("maps/map1.txt")
    
    # 创建环境
    base_env = GridWorldEnv(grid_map, view_size=5)
    env = SafeEnvWrapper(
        base_env,
        safety_formula="G(!wall)",
        unsafe_penalty=-2.0
    )
    
    obs, _ = env.reset()
    agent_pos = env.unwrapped.agent_pos
    print(f"初始位置: {tuple(agent_pos)}")
    print(f"墙位置: {env.env_info['wall_positions']}")
    
    # 故意尝试撞墙
    print("\n尝试撞墙测试:")
    
    # 先移动到墙边
    for action in [3, 3, 3]:  # 向右移动
        obs, _, _, _, _ = env.step(action)
    
    agent_pos = env.unwrapped.agent_pos
    print(f"当前位置: {tuple(agent_pos)}")
    print(f"安全动作: {env.get_safe_actions()}")
    
    # 尝试向上（可能撞墙）
    obs, reward, _, _, info = env.step(0)  # 上
    agent_pos = env.unwrapped.agent_pos
    print(f"尝试向上: 位置={tuple(agent_pos)}, 奖励={reward:.2f}")
    if info.get('safety_intervention'):
        print(f"  被干预! 原因: {info.get('intervention_reason')}")
    
    stats = env.get_episode_stats()
    print(f"\n干预次数: {stats['interventions']}")
    
    env.close()
    print("\n[OK] 不安全动作处理测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("安全包装器测试")
    print("=" * 60)
    
    try:
        # 测试新观测空间
        test_observation_space()
        
        # 测试安全监控器
        monitor, action_map = test_safety_monitor()
        
        # 测试动作过滤器
        test_action_filter(monitor, action_map)
        
        # 测试完整包装器
        test_safe_env_wrapper()
        
        # 测试不安全动作
        test_unsafe_actions()
        
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def test_observation_space():
    """测试新的观测空间（局部视野 + 目标方向）"""
    print("\n" + "=" * 60)
    print("0. 测试观测空间")
    print("=" * 60)
    
    from safe_rl_drone.env import GridWorldEnv
    
    # 加载地图
    grid_map = load_map_from_file("maps/map1.txt")
    
    # 创建环境
    env = GridWorldEnv(grid_map, view_size=5)
    
    print(f"\n环境信息:")
    print(f"  地图大小: {env.rows} x {env.cols}")
    print(f"  局部视野: {env.view_size} x {env.view_size}")
    print(f"  观测维度: {env.observation_space.shape}")
    
    obs, _ = env.reset()
    print(f"\n观测结构:")
    print(f"  总维度: {len(obs)}")
    print(f"  局部视野: {env.view_size * env.view_size} 维")
    print(f"  目标方向: 2 维")
    
    # 解析观测
    local_view = obs[:-2].reshape(env.view_size, env.view_size)
    goal_dir = obs[-2:]
    
    print(f"\n局部视野 (0=空地, 1=墙, 2=目标, 3=边界外):")
    print(local_view)
    
    print(f"\n目标方向 (归一化): [{goal_dir[0]:.3f}, {goal_dir[1]:.3f}]")
    
    # 移动几步看看视野变化
    print("\n移动测试:")
    actions = [3, 3, 1]  # 右、右、下
    action_names = ['上', '下', '左', '右', '不动']
    
    for action in actions:
        obs, _, _, _, _ = env.step(action)
        local_view = obs[:-2].reshape(env.view_size, env.view_size)
        goal_dir = obs[-2:]
        print(f"\n  动作: {action_names[action]}, 位置: {tuple(env.agent_pos)}")
        print(f"  目标方向: [{goal_dir[0]:.3f}, {goal_dir[1]:.3f}]")
    
    env.close()
    print("\n[OK] 观测空间测试通过")


if __name__ == '__main__':
    sys.exit(main())
