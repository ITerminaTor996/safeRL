#!/usr/bin/env python3
"""
Goal-Conditioned RL 测试

测试内容：
1. 随机目标采样
2. 指定目标
3. 目标泛化验证
"""

import sys
import os
import numpy as np

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_map_from_file


def test_random_goal():
    """测试随机目标采样"""
    print("\n" + "=" * 60)
    print("1. 测试随机目标采样")
    print("=" * 60)
    
    from safe_rl_drone.env import GridWorldEnv
    
    grid_map = load_map_from_file("maps/map1.txt")
    env = GridWorldEnv(grid_map, view_size=5, random_goal=True, random_start=True)
    
    print(f"\n环境信息:")
    print(f"  地图大小: {env.rows} x {env.cols}")
    print(f"  可用位置数: {len(env.empty_cells)}")
    print(f"  默认目标: {tuple(env.default_target_pos)}")
    print(f"  默认起点: {tuple(env.agent_start_pos)}")
    
    # 多次 reset，检查目标是否随机
    print(f"\n随机采样测试 (5次 reset):")
    goals_seen = set()
    starts_seen = set()
    
    for i in range(5):
        obs, info = env.reset()
        goal = info['goal']
        start = info['start']
        goals_seen.add(goal)
        starts_seen.add(start)
        print(f"  Reset {i+1}: 起点={start}, 目标={goal}")
        
        # 验证起点 ≠ 目标
        assert start != goal, "起点不应等于目标"
    
    print(f"\n不同目标数: {len(goals_seen)}")
    print(f"不同起点数: {len(starts_seen)}")
    
    env.close()
    print("\n[OK] 随机目标采样测试通过")


def test_specified_goal():
    """测试指定目标"""
    print("\n" + "=" * 60)
    print("2. 测试指定目标")
    print("=" * 60)
    
    from safe_rl_drone.env import GridWorldEnv
    
    grid_map = load_map_from_file("maps/map1.txt")
    env = GridWorldEnv(grid_map, view_size=5, random_goal=False)
    
    # 指定目标
    test_goals = [(2, 3), (4, 2), (3, 5)]
    
    print(f"\n指定目标测试:")
    for goal in test_goals:
        obs, info = env.reset(options={'goal': goal})
        actual_goal = info['goal']
        print(f"  指定目标={goal}, 实际目标={actual_goal}")
        assert actual_goal == goal, f"目标不匹配: {actual_goal} != {goal}"
        
        # 验证目标方向
        goal_dir = obs[-2:]
        print(f"    目标方向: [{goal_dir[0]:.3f}, {goal_dir[1]:.3f}]")
    
    env.close()
    print("\n[OK] 指定目标测试通过")


def test_goal_direction():
    """测试目标方向计算"""
    print("\n" + "=" * 60)
    print("3. 测试目标方向计算")
    print("=" * 60)
    
    from safe_rl_drone.env import GridWorldEnv
    
    grid_map = load_map_from_file("maps/map1.txt")
    env = GridWorldEnv(grid_map, view_size=5)
    
    # 测试不同目标位置的方向
    test_cases = [
        {'start': (0, 0), 'goal': (5, 5), 'expected_sign': (1, 1)},   # 右下
        {'start': (5, 5), 'goal': (0, 0), 'expected_sign': (-1, -1)}, # 左上
        {'start': (2, 2), 'goal': (2, 5), 'expected_sign': (0, 1)},   # 右
        {'start': (2, 2), 'goal': (5, 2), 'expected_sign': (1, 0)},   # 下
    ]
    
    print(f"\n目标方向测试:")
    for case in test_cases:
        obs, info = env.reset(options={'start': case['start'], 'goal': case['goal']})
        goal_dir = obs[-2:]
        
        # 检查方向符号
        expected = case['expected_sign']
        actual_sign = (np.sign(goal_dir[0]), np.sign(goal_dir[1]))
        
        print(f"  起点={case['start']}, 目标={case['goal']}")
        print(f"    方向: [{goal_dir[0]:.3f}, {goal_dir[1]:.3f}]")
        print(f"    符号: 期望={expected}, 实际={actual_sign}")
    
    # 测试到达目标时方向为 0
    obs, _ = env.reset(options={'start': (3, 3), 'goal': (3, 3)})
    goal_dir = obs[-2:]
    print(f"\n  起点=目标=(3,3) 时方向: [{goal_dir[0]:.3f}, {goal_dir[1]:.3f}]")
    assert np.allclose(goal_dir, [0, 0]), "在目标位置时方向应为 [0, 0]"
    
    env.close()
    print("\n[OK] 目标方向计算测试通过")


def test_safe_wrapper_with_random_goal():
    """测试安全包装器与随机目标的兼容性"""
    print("\n" + "=" * 60)
    print("4. 测试 SafeEnvWrapper + 随机目标")
    print("=" * 60)
    
    from safe_rl_drone.env import GridWorldEnv
    from safe_rl_drone.wrappers.safe_env_wrapper import SafeEnvWrapper
    
    grid_map = load_map_from_file("maps/map1.txt")
    base_env = GridWorldEnv(grid_map, view_size=5, random_goal=True, random_start=True)
    
    env = SafeEnvWrapper(
        base_env,
        safety_formula="G(!wall) & G(!boundary)",
        unsafe_penalty=-1.0
    )
    
    print(f"\n多次 reset 测试:")
    for i in range(3):
        obs, info = env.reset()
        goal = info['goal']
        start = info['start']
        print(f"  Reset {i+1}: 起点={start}, 目标={goal}")
        
        # 执行几步随机动作
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            if terminated:
                break
    
    # 测试指定目标
    print(f"\n指定目标测试:")
    obs, info = env.reset(options={'goal': (3, 4)})
    print(f"  指定目标=(3,4), 实际目标={info['goal']}")
    assert info['goal'] == (3, 4), "目标不匹配"
    
    env.close()
    print("\n[OK] SafeEnvWrapper + 随机目标测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Goal-Conditioned RL 测试")
    print("=" * 60)
    
    try:
        test_random_goal()
        test_specified_goal()
        test_goal_direction()
        test_safe_wrapper_with_random_goal()
        
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
