#!/usr/bin/env python3
"""
LTL 模块综合测试脚本

测试内容：
1. 原子命题模块
2. LTL 解析器
3. Robustness 计算
4. FSA 监控器
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_propositions():
    """测试原子命题模块"""
    print("\n" + "=" * 60)
    print("1. 测试原子命题模块")
    print("=" * 60)
    
    from safe_rl_drone.ltl.propositions import (
        AtomicPropositionManager, 
        PositionProposition,
        RegionProposition
    )
    
    # 创建管理器
    manager = AtomicPropositionManager()
    manager.register_auto_propositions()
    
    # 添加自定义命题
    manager.register_proposition(
        'checkpoint1', 
        PositionProposition('checkpoint1', (2, 3))
    )
    manager.register_proposition(
        'danger_zone',
        RegionProposition('danger_zone', [(1, 1), (1, 2), (2, 1)], avoid=True)
    )
    
    # 设置环境信息
    env_info = {
        'wall_positions': {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
        'goal_position': (4, 4),
        'grid_size': (6, 6)
    }
    manager.update_env_info(env_info)
    
    print(f"注册的命题: {manager.get_all_names()}")
    
    # 测试不同位置
    test_positions = [
        (np.array([3, 3]), "空地"),
        (np.array([4, 4]), "目标"),
        (np.array([1, 1]), "危险区"),
        (np.array([2, 3]), "checkpoint1"),
        (np.array([0, 0]), "墙"),
    ]
    
    for pos, desc in test_positions:
        print(f"\n位置 {tuple(pos)} ({desc}):")
        for prop_name in ['wall', 'goal', 'checkpoint1', 'danger_zone']:
            val = manager.evaluate(prop_name, pos)
            rho = manager.robustness(prop_name, pos)
            print(f"  {prop_name}: 值={val}, ρ={rho:.2f}")
    
    print("\n✓ 原子命题模块测试通过")
    return manager, env_info


def test_parser():
    """测试 LTL 解析器"""
    print("\n" + "=" * 60)
    print("2. 测试 LTL 解析器")
    print("=" * 60)
    
    from safe_rl_drone.ltl.parser import LTLParser, SPOT_AVAILABLE
    
    print(f"Spot 库可用: {SPOT_AVAILABLE}")
    
    test_formulas = [
        "G(!wall)",
        "F(goal)",
        "G(!wall) & F(goal)",
        "F(checkpoint1) & F(goal)",
        "G(!wall) & G(!boundary) & F(goal)",
    ]
    
    parser = LTLParser()
    
    for formula in test_formulas:
        print(f"\n公式: {formula}")
        parser.parse(formula)
        print(f"  原子命题: {parser.get_atomic_propositions()}")
        print(f"  语法树: {parser.syntax_tree}")
    
    print("\n✓ LTL 解析器测试通过")
    return parser


def test_robustness(prop_manager, env_info):
    """测试 Robustness 计算"""
    print("\n" + "=" * 60)
    print("3. 测试 Robustness 计算")
    print("=" * 60)
    
    from safe_rl_drone.ltl.parser import LTLParser
    from safe_rl_drone.ltl.robustness import RobustnessCalculator
    
    parser = LTLParser()
    
    # 测试用例：(公式, 轨迹, 预期结果描述)
    test_cases = [
        (
            "G(!wall)",
            [np.array([3, 3]), np.array([3, 4]), np.array([4, 4])],
            "安全轨迹，应该 ρ > 0"
        ),
        (
            "G(!wall)",
            [np.array([3, 3]), np.array([0, 0]), np.array([4, 4])],
            "经过墙，应该 ρ < 0"
        ),
        (
            "F(goal)",
            [np.array([0, 0]), np.array([2, 2]), np.array([4, 4])],
            "到达目标，应该 ρ >= 0"
        ),
        (
            "F(goal)",
            [np.array([0, 0]), np.array([1, 0]), np.array([2, 0])],
            "未到达目标，应该 ρ < 0"
        ),
        (
            "G(!wall) & F(goal)",
            [np.array([3, 3]), np.array([3, 4]), np.array([4, 4])],
            "安全到达目标，应该 ρ > 0"
        ),
    ]
    
    for formula, trajectory, description in test_cases:
        print(f"\n公式: {formula}")
        print(f"轨迹: {[tuple(s) for s in trajectory]}")
        print(f"预期: {description}")
        
        parser.parse(formula)
        calculator = RobustnessCalculator(parser, prop_manager)
        
        rho = calculator.compute(trajectory[-1], trajectory)
        satisfied = calculator.is_satisfied(trajectory[-1], trajectory)
        
        print(f"结果: ρ = {rho:.2f}, 满足 = {satisfied}")
    
    print("\n✓ Robustness 计算测试通过")


def test_fsa(prop_manager):
    """测试 FSA 监控器"""
    print("\n" + "=" * 60)
    print("4. 测试 FSA 监控器")
    print("=" * 60)
    
    from safe_rl_drone.ltl.fsa import FSAMonitor, SPOT_AVAILABLE
    
    print(f"Spot 库可用: {SPOT_AVAILABLE}")
    
    # 创建监控器
    monitor = FSAMonitor("G(!wall)")
    monitor.set_prop_manager(prop_manager)
    
    if monitor.fsa:
        print(f"\n自动机信息:")
        print(f"  状态数: {monitor.fsa.num_states}")
        print(f"  初始状态: {monitor.fsa.initial_state}")
        print(f"  接受状态: {monitor.fsa.accepting_states}")
        print(f"  Trap 状态: {monitor.fsa.trap_states}")
    
    # 模拟轨迹
    print(f"\n运行时监控测试:")
    trajectory = [
        (np.array([3, 3]), "安全位置"),
        (np.array([3, 4]), "安全位置"),
        (np.array([4, 4]), "目标位置"),
    ]
    
    monitor.reset()
    for state, desc in trajectory:
        fsa_state, is_acc, is_trap = monitor.step(state)
        print(f"  位置 {tuple(state)} ({desc}): "
              f"FSA状态={fsa_state}, 接受={is_acc}, trap={is_trap}")
    
    # 测试动作安全检查
    print(f"\n动作安全检查测试:")
    current = np.array([3, 3])
    safe_next = np.array([3, 4])
    unsafe_next = np.array([0, 0])  # 墙
    
    monitor.reset()
    print(f"  当前位置: {tuple(current)}")
    print(f"  安全动作 -> {tuple(safe_next)}: "
          f"安全={monitor.check_action_safety(current, safe_next)}")
    print(f"  不安全动作 -> {tuple(unsafe_next)}: "
          f"安全={monitor.check_action_safety(current, unsafe_next)}")
    
    print("\n✓ FSA 监控器测试通过")


def test_integration():
    """集成测试"""
    print("\n" + "=" * 60)
    print("5. 集成测试")
    print("=" * 60)
    
    from safe_rl_drone.ltl import (
        AtomicPropositionManager,
        LTLParser,
        RobustnessCalculator,
        FSAMonitor
    )
    from safe_rl_drone.ltl.propositions import PositionProposition
    
    # 1. 设置原子命题
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    prop_manager.register_proposition(
        'checkpoint', PositionProposition('checkpoint', (2, 2))
    )
    
    env_info = {
        'wall_positions': {(1, 1), (1, 2), (2, 1)},
        'goal_position': (4, 4),
        'grid_size': (6, 6)
    }
    prop_manager.update_env_info(env_info)
    
    # 2. 解析公式
    formula = "G(!wall) & F(goal)"
    parser = LTLParser()
    parser.parse(formula)
    
    print(f"公式: {formula}")
    print(f"原子命题: {parser.get_atomic_propositions()}")
    
    # 3. 创建 Robustness 计算器
    rob_calc = RobustnessCalculator(parser, prop_manager)
    
    # 4. 创建 FSA 监控器
    fsa_monitor = FSAMonitor(formula)
    fsa_monitor.set_prop_manager(prop_manager)
    
    # 5. 模拟一个 episode
    print(f"\n模拟 Episode:")
    trajectory = [
        np.array([0, 0]),
        np.array([0, 3]),
        np.array([2, 3]),
        np.array([3, 3]),
        np.array([4, 4]),  # 目标
    ]
    
    fsa_monitor.reset()
    for t, state in enumerate(trajectory):
        # FSA 状态更新
        fsa_state, is_acc, is_trap = fsa_monitor.step(state)
        
        # 计算 robustness
        rho = rob_calc.compute(state, trajectory[:t+1])
        
        # 检查各命题
        at_wall = prop_manager.evaluate('wall', state)
        at_goal = prop_manager.evaluate('goal', state)
        
        print(f"  t={t}, pos={tuple(state)}: "
              f"wall={at_wall}, goal={at_goal}, "
              f"ρ={rho:.2f}, FSA={fsa_state}")
    
    print("\n✓ 集成测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("LTL 模块综合测试")
    print("=" * 60)
    
    try:
        # 测试原子命题
        prop_manager, env_info = test_propositions()
        
        # 测试解析器
        test_parser()
        
        # 测试 Robustness
        test_robustness(prop_manager, env_info)
        
        # 测试 FSA
        test_fsa(prop_manager)
        
        # 集成测试
        test_integration()
        
        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
