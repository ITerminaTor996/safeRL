"""
测试 SafetyFilter 类
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from safe_rl_drone.safety import SafetyFilter
from safe_rl_drone.ltl.propositions import AtomicPropositionManager


def test_safety_filter_basic():
    """测试基本的安全过滤功能"""
    print("=" * 60)
    print("测试 SafetyFilter - 基本功能")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    # 设置环境信息
    env_info = {
        'wall_positions': {(1, 1), (2, 2), (3, 3)},
        'goal_position': (5, 5),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    # 动作映射
    action_map = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
        4: (0, 0)    # stay
    }
    
    # 创建安全过滤器
    safety_filter = SafetyFilter(
        safety_formula="G(!wall) & G(!boundary)",
        prop_manager=prop_manager,
        action_map=action_map,
        num_actions=5
    )
    
    print("\n测试场景 1: 安全动作（不过滤）")
    print("-" * 40)
    current_state = np.array([0, 0])
    action = 3  # 向右，安全
    
    safe_action, filtered, info = safety_filter.filter_action(current_state, action)
    print(f"当前位置: {tuple(current_state)}")
    print(f"原始动作: {action} (向右)")
    print(f"过滤结果: {safe_action}")
    print(f"是否被过滤: {filtered}")
    print(f"FSA 状态: {info.get('fsa_state')}")
    
    assert not filtered, "安全动作不应该被过滤"
    assert safe_action == action, "安全动作应该保持不变"
    
    print("\n测试场景 2: 不安全动作（会撞墙）")
    print("-" * 40)
    current_state = np.array([1, 0])
    action = 3  # 向右，会到 (1,1) 撞墙
    
    # DEBUG: 添加调试输出
    print(f"[DEBUG] 测试前 FSA 状态: {safety_filter.fsa_monitor.current_state}")
    print(f"[DEBUG] 当前位置: {tuple(current_state)}")
    print(f"[DEBUG] 动作: {action}")
    
    # 预测下一状态
    next_state = safety_filter._predict_next_state(current_state, action)
    print(f"[DEBUG] 预测的下一状态: {tuple(next_state)}")
    print(f"[DEBUG] 下一状态是墙: {prop_manager.evaluate('wall', next_state)}")
    
    # 手动测试 FSA 转移
    saved = safety_filter.fsa_monitor.current_state
    test_state, test_acc, test_trap = safety_filter.fsa_monitor.step(next_state)
    print(f"[DEBUG] 手动测试 step: state={test_state}, accepting={test_acc}, trap={test_trap}")
    safety_filter.fsa_monitor.current_state = saved
    
    safe_action, filtered, info = safety_filter.filter_action(current_state, action)
    print(f"当前位置: {tuple(current_state)}")
    print(f"原始动作: {action} (向右，会到 (1,1) 撞墙)")
    print(f"过滤结果: {safe_action}")
    print(f"是否被过滤: {filtered}")
    print(f"原因: {info.get('reason')}")
    print(f"FSA 状态: {info.get('fsa_state')}")
    
    assert filtered, "不安全动作应该被过滤"
    assert safe_action == 4, "应该替换为 stay 动作"
    
    print("\n测试场景 3: 边界检查")
    print("-" * 40)
    current_state = np.array([0, 0])
    action = 0  # 向上，会越界
    
    safe_action, filtered, info = safety_filter.filter_action(current_state, action)
    print(f"当前位置: {tuple(current_state)}")
    print(f"原始动作: {action} (向上，会越界)")
    print(f"过滤结果: {safe_action}")
    print(f"是否被过滤: {filtered}")
    
    assert filtered, "越界动作应该被过滤"
    
    # 统计信息
    print("\n统计信息:")
    print("-" * 40)
    stats = safety_filter.get_stats()
    print(f"总动作数: {stats['total_actions']}")
    print(f"干预次数: {stats['intervention_count']}")
    print(f"干预率: {stats['intervention_rate']:.2%}")
    
    print("\n" + "=" * 60)
    print("SafetyFilter 基本功能测试通过！")
    print("=" * 60)


def test_safety_filter_trajectory():
    """测试轨迹中的连续过滤"""
    print("\n" + "=" * 60)
    print("测试 SafetyFilter - 轨迹过滤")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    # 设置环境信息（简单场景）
    env_info = {
        'wall_positions': {(2, 2)},
        'goal_position': (4, 4),
        'grid_size': (6, 6)
    }
    prop_manager.update_env_info(env_info)
    
    # 动作映射
    action_map = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
        4: (0, 0)    # stay
    }
    
    # 创建安全过滤器
    safety_filter = SafetyFilter(
        safety_formula="G(!wall)",
        prop_manager=prop_manager,
        action_map=action_map,
        num_actions=5
    )
    
    # 模拟轨迹
    trajectory = [
        (np.array([0, 0]), 3),  # 向右，安全
        (np.array([0, 1]), 1),  # 向下，安全
        (np.array([1, 1]), 3),  # 向右，安全
        (np.array([1, 2]), 1),  # 向下，会到 (2,2) 撞墙！
        (np.array([2, 2]), 3),  # 从安全位置继续
    ]
    
    print("\n模拟轨迹:")
    print("-" * 40)
    
    for i, (state, action) in enumerate(trajectory):
        safe_action, filtered, info = safety_filter.filter_action(state, action)
        
        action_names = ['up', 'down', 'left', 'right', 'stay']
        print(f"Step {i}: pos={tuple(state)}, "
              f"action={action_names[action]}, "
              f"filtered={filtered}, "
              f"result={action_names[safe_action]}")
    
    # 统计信息
    stats = safety_filter.get_stats()
    print(f"\n总动作数: {stats['total_actions']}")
    print(f"干预次数: {stats['intervention_count']}")
    print(f"干预率: {stats['intervention_rate']:.2%}")
    
    assert stats['intervention_count'] > 0, "应该有至少一次干预"
    
    print("\n" + "=" * 60)
    print("SafetyFilter 轨迹过滤测试通过！")
    print("=" * 60)


def test_get_safe_actions():
    """测试获取所有安全动作"""
    print("\n" + "=" * 60)
    print("测试 SafetyFilter - 获取安全动作集合")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    # 设置环境信息
    env_info = {
        'wall_positions': {(1, 1), (1, 2), (2, 1)},
        'goal_position': (5, 5),
        'grid_size': (6, 6)
    }
    prop_manager.update_env_info(env_info)
    
    # 动作映射
    action_map = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
        4: (0, 0)    # stay
    }
    
    # 创建安全过滤器
    safety_filter = SafetyFilter(
        safety_formula="G(!wall)",
        prop_manager=prop_manager,
        action_map=action_map,
        num_actions=5
    )
    
    # 测试位置 (1, 0)：右边是墙
    state = np.array([1, 0])
    safe_actions = safety_filter.get_safe_actions(state)
    
    action_names = ['up', 'down', 'left', 'right', 'stay']
    print(f"\n位置 {tuple(state)} 的安全动作:")
    print(f"墙的位置: {env_info['wall_positions']}")
    print(f"安全动作: {[action_names[a] for a in safe_actions]}")
    
    # 向右 (3) 会到 (1,1) 撞墙，应该不安全
    # 向左 (2) 会到 (1,-1) 越界，应该不安全（因为公式是 G(!wall)，不检查边界）
    # 实际上这个测试只用了 G(!wall)，所以只检查墙，不检查边界
    assert 3 not in safe_actions, "向右会撞墙，不应该安全"
    assert 4 in safe_actions, "stay 应该总是安全的"
    
    print("\n" + "=" * 60)
    print("SafetyFilter 安全动作集合测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_safety_filter_basic()
    test_safety_filter_trajectory()
    test_get_safe_actions()
    
    print("\n" + "=" * 60)
    print("所有 SafetyFilter 测试通过！✓")
    print("=" * 60)
