"""
测试 FSA 陷阱状态检测的修复
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from safe_rl_drone.ltl.fsa import FSAMonitor
from safe_rl_drone.ltl.propositions import AtomicPropositionManager


def test_trap_detection_safety_formula():
    """测试安全公式的陷阱状态检测"""
    print("=" * 60)
    print("测试陷阱状态检测 - 安全公式")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': {(1, 1)},
        'goal_position': (5, 5),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    # 测试安全公式
    safety_formulas = [
        "G(!wall)",
        "G(!wall) & G(!boundary)",
        "G(!wall & !boundary)",
    ]
    
    for formula in safety_formulas:
        print(f"\n公式: {formula}")
        
        monitor = FSAMonitor(formula)
        monitor.set_prop_manager(prop_manager)
        
        if monitor.fsa:
            print(f"  状态数: {monitor.fsa.num_states}")
            print(f"  初始状态: {monitor.fsa.initial_state}")
            print(f"  接受状态: {monitor.fsa.accepting_states}")
            print(f"  陷阱状态: {monitor.fsa.trap_states}")
            
            # 验证：对于安全公式，非接受状态应该是陷阱状态
            non_accepting = set(range(monitor.fsa.num_states)) - monitor.fsa.accepting_states
            print(f"  非接受状态: {non_accepting}")
            
            if monitor.fsa.trap_states == non_accepting:
                print("  ✓ 陷阱状态检测正确")
            else:
                print("  ✗ 陷阱状态检测可能有问题")
    
    print("\n" + "=" * 60)


def test_trap_detection_liveness_formula():
    """测试活性公式的陷阱状态检测"""
    print("\n" + "=" * 60)
    print("测试陷阱状态检测 - 活性公式")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (5, 5),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    # 测试活性公式（使用 ba 模式）
    liveness_formulas = [
        "F(goal)",
        "F(goal) & G(!wall)",
    ]
    
    for formula in liveness_formulas:
        print(f"\n公式: {formula}")
        
        # 对比 monitor 和 ba 模式
        for mode in ['monitor', 'ba']:
            print(f"\n  模式: {mode}")
            monitor = FSAMonitor(formula, mode=mode)
            monitor.set_prop_manager(prop_manager)
            
            if monitor.fsa:
                print(f"    状态数: {monitor.fsa.num_states}")
                print(f"    初始状态: {monitor.fsa.initial_state}")
                print(f"    接受状态: {monitor.fsa.accepting_states}")
                print(f"    陷阱状态: {monitor.fsa.trap_states}")
                
                # 对于活性公式，可能有中间状态（既不接受也不是陷阱）
                non_accepting = set(range(monitor.fsa.num_states)) - monitor.fsa.accepting_states
                print(f"    非接受状态: {non_accepting}")
                
                if monitor.fsa.trap_states <= non_accepting:
                    print("    ✓ 陷阱状态是非接受状态的子集")
                else:
                    print("    ✗ 陷阱状态检测有问题")
    
    print("\n" + "=" * 60)


def test_action_safety_with_trap():
    """测试动作安全性检查（使用修复后的陷阱检测）"""
    print("\n" + "=" * 60)
    print("测试动作安全性检查")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': {(1, 1), (2, 2)},
        'goal_position': (5, 5),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    # 创建 FSA 监控器
    monitor = FSAMonitor("G(!wall)")
    monitor.set_prop_manager(prop_manager)
    
    print(f"\n公式: G(!wall)")
    print(f"墙的位置: {env_info['wall_positions']}")
    
    # 测试场景
    test_cases = [
        (np.array([0, 0]), np.array([0, 1]), True, "安全移动"),
        (np.array([0, 0]), np.array([1, 1]), False, "会撞墙"),
        (np.array([1, 0]), np.array([1, 1]), False, "会撞墙"),
        (np.array([3, 3]), np.array([3, 4]), True, "安全移动"),
    ]
    
    print("\n测试动作安全性:")
    print("-" * 40)
    
    for current, next_state, expected_safe, description in test_cases:
        monitor.reset()  # 重置到初始状态
        is_safe = monitor.check_action_safety(current, next_state)
        
        status = "✓" if is_safe == expected_safe else "✗"
        print(f"{status} {tuple(current)} -> {tuple(next_state)}: "
              f"safe={is_safe} (expected={expected_safe}) - {description}")
        
        if is_safe != expected_safe:
            print(f"   错误！FSA 状态: {monitor.get_state_info()}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_trap_detection_safety_formula()
    test_trap_detection_liveness_formula()
    test_action_safety_with_trap()
    
    print("\n" + "=" * 60)
    print("FSA 陷阱状态检测测试完成！")
    print("=" * 60)
