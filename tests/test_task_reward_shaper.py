"""
测试 TaskRewardShaper 模块 - 边条件 Robustness 计算

测试内容：
1. 基于真实 FSA 边条件计算 robustness
2. 排除自环边，只考虑能推进状态的边
3. AND 取 min，OR 取 max，NOT 取负
4. 进度奖励只在 FSA 状态改变时给
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from safe_rl_drone.safety.task_reward_shaper import TaskRewardShaper
from safe_rl_drone.ltl.fsa import FSAMonitor
from safe_rl_drone.ltl.propositions import (
    AtomicPropositionManager, 
    PositionProposition,
    RegionProposition
)


def test_simple_eventually():
    """测试简单公式 F(goal) - 排除自环边"""
    print("=" * 60)
    print("测试 1: F(goal) - 排除自环边")
    print("=" * 60)
    
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (5, 5),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    shaper = TaskRewardShaper("F(goal)", prop_manager)
    
    print(f"\n公式: F(goal)")
    print(f"目标: (5, 5)")
    print(f"FSA 转移边:")
    for t in shaper.fsa_monitor.fsa.transitions:
        is_self_loop = "（自环，排除）" if t.source == t.target else "（非自环，计算）"
        print(f"  {t} {is_self_loop}")
    
    # F(goal) 的 FSA：
    # q1 --[goal]--> q0 (接受)  ← 非自环，计算这个
    # q1 --[!goal]--> q1        ← 自环，排除
    # q0 --[1]--> q0            ← 自环，排除
    
    test_positions = [
        ((0, 0), "远离目标"),
        ((3, 3), "接近目标"),
        ((5, 5), "在目标上"),
    ]
    
    print(f"\n边条件 Robustness（只考虑非自环边 [goal]）:")
    print("-" * 40)
    
    for pos, desc in test_positions:
        shaper.reset()
        state = np.array(pos)
        rho = shaper.compute_edge_based_robustness(state)
        
        # 非自环边只有 [goal]，期望值就是 ρ(goal)
        expected = prop_manager.robustness('goal', state)
        
        print(f"位置 {pos} ({desc}): rho={rho:.2f}, 期望={expected:.2f}")
        assert abs(rho - expected) < 0.1, f"计算错误: {rho} != {expected}"
    
    print("\n✓ 测试通过!")


def test_multi_goal_or():
    """测试多目标选择 F(goal1 | goal2) - 真实 FSA 边条件"""
    print("\n" + "=" * 60)
    print("测试 2: F(goal1 | goal2) - OR 条件")
    print("=" * 60)
    
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    prop_manager.register_proposition('goal1', PositionProposition('goal1', (2, 2)))
    prop_manager.register_proposition('goal2', PositionProposition('goal2', (8, 8)))
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (5, 5),  # 默认 goal，不影响测试
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    # 使用真实公式构建 FSA
    shaper = TaskRewardShaper("F(goal1 | goal2)", prop_manager)
    
    print(f"\n公式: F(goal1 | goal2)")
    print(f"goal1: (2, 2), goal2: (8, 8)")
    print(f"FSA 转移边:")
    for t in shaper.fsa_monitor.fsa.transitions:
        is_self_loop = "（自环）" if t.source == t.target else "（非自环）"
        print(f"  {t} {is_self_loop}")
    
    test_positions = [
        ((0, 0), "更接近 goal1"),
        ((5, 5), "中间位置"),
        ((9, 9), "更接近 goal2"),
        ((2, 2), "在 goal1 上"),
    ]
    
    print(f"\n边条件 Robustness:")
    print("-" * 40)
    
    for pos, desc in test_positions:
        shaper.reset()
        state = np.array(pos)
        rho = shaper.compute_edge_based_robustness(state)
        
        # F(goal1 | goal2) 的非自环边条件是 "goal1 | goal2"
        # OR 取 max
        rho1 = prop_manager.robustness('goal1', state)
        rho2 = prop_manager.robustness('goal2', state)
        expected = max(rho1, rho2)
        
        print(f"位置 {pos} ({desc}):")
        print(f"  ρ(goal1)={rho1:.2f}, ρ(goal2)={rho2:.2f}")
        print(f"  max={expected:.2f}, 实际={rho:.2f}")
        
        assert abs(rho - expected) < 0.1, f"OR 计算错误"
    
    print("\n✓ 测试通过!")


def test_sequential_task():
    """测试顺序任务 F(wp1 & F(wp2)) - AND 条件"""
    print("\n" + "=" * 60)
    print("测试 3: F(wp1 & F(wp2)) - 顺序任务")
    print("=" * 60)
    
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    prop_manager.register_proposition('wp1', PositionProposition('wp1', (3, 3)))
    prop_manager.register_proposition('wp2', PositionProposition('wp2', (7, 7)))
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (9, 9),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    shaper = TaskRewardShaper("F(wp1 & F(wp2))", prop_manager)
    
    print(f"\n公式: F(wp1 & F(wp2))")
    print(f"wp1: (3, 3), wp2: (7, 7)")
    print(f"FSA 状态数: {shaper.fsa_monitor.fsa.num_states}")
    print(f"FSA 转移边:")
    for t in shaper.fsa_monitor.fsa.transitions:
        is_self_loop = "（自环）" if t.source == t.target else "（非自环）"
        print(f"  {t} {is_self_loop}")
    
    # 阶段 1：在初始状态，需要先到 wp1
    print(f"\n阶段 1：初始状态，目标是 wp1")
    print("-" * 40)
    
    shaper.reset()
    initial_fsa_state = shaper.fsa_monitor.current_state
    print(f"初始 FSA 状态: {initial_fsa_state}")
    
    # 找出初始状态的非自环边
    edges = shaper.fsa_monitor.fsa.get_transitions_from(initial_fsa_state)
    non_self_edges = [e for e in edges if e.source != e.target]
    print(f"非自环边: {[str(e) for e in non_self_edges]}")
    
    test_pos = (1, 1)
    state = np.array(test_pos)
    rho = shaper.compute_edge_based_robustness(state)
    print(f"位置 {test_pos}: rho={rho:.2f}")
    
    # 阶段 2：到达 wp1 后，FSA 状态改变
    print(f"\n阶段 2：到达 wp1，FSA 状态改变")
    print("-" * 40)
    
    state_wp1 = np.array([3, 3])
    reward, info = shaper.compute_reward(state_wp1)
    print(f"位置 (3, 3): FSA 状态 {initial_fsa_state} -> {info['fsa_state']}")
    print(f"  r_progress={info['r_progress']:.2f}")
    
    if info['fsa_state'] != initial_fsa_state:
        print("  *** FSA 状态改变! ***")
        
        # 现在目标是 wp2
        new_edges = shaper.fsa_monitor.fsa.get_transitions_from(info['fsa_state'])
        new_non_self_edges = [e for e in new_edges if e.source != e.target]
        print(f"  新的非自环边: {[str(e) for e in new_non_self_edges]}")
    
    print("\n✓ 测试通过!")


def test_until_formula():
    """测试 Until 公式 (!danger) U safe - 排除通向陷阱的边"""
    print("\n" + "=" * 60)
    print("测试 4: (!danger) U safe - Until 公式")
    print("=" * 60)
    
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    prop_manager.register_proposition('danger', RegionProposition('danger', [(3, 3), (3, 4), (4, 3)]))
    prop_manager.register_proposition('safe', PositionProposition('safe', (7, 7)))
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (9, 9),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    shaper = TaskRewardShaper("(!danger) U safe", prop_manager)
    
    print(f"\n公式: (!danger) U safe")
    print(f"danger 区域: [(3,3), (3,4), (4,3)]")
    print(f"safe 位置: (7, 7)")
    print(f"FSA 状态数: {shaper.fsa_monitor.fsa.num_states}")
    print(f"陷阱状态: {shaper.fsa_monitor.fsa.trap_states}")
    print(f"FSA 转移边:")
    for t in shaper.fsa_monitor.fsa.transitions:
        is_self_loop = "（自环）" if t.source == t.target else ""
        is_trap = "（→陷阱，排除）" if t.target in shaper.fsa_monitor.fsa.trap_states else ""
        is_valid = "" if is_self_loop or is_trap else "（有效）"
        print(f"  {t} {is_self_loop}{is_trap}{is_valid}")
    
    # 有效边只有: 1 --[safe]--> 0
    # 所以 robustness 应该只基于 ρ(safe)
    
    test_cases = [
        ((0, 0), "远离 danger 和 safe"),
        ((5, 5), "接近 safe"),
        ((7, 7), "在 safe 上"),
        ((3, 3), "在 danger 上"),
    ]
    
    print(f"\n边条件 Robustness（只考虑有效边 [safe]）:")
    print("-" * 40)
    
    for pos, desc in test_cases:
        shaper.reset()
        state = np.array(pos)
        rho = shaper.compute_edge_based_robustness(state)
        
        rho_safe = prop_manager.robustness('safe', state)
        rho_danger = prop_manager.robustness('danger', state)
        
        print(f"位置 {pos} ({desc}):")
        print(f"  ρ(safe)={rho_safe:.2f}, ρ(danger)={rho_danger:.2f}")
        print(f"  边条件 rho={rho:.2f}, 期望={rho_safe:.2f}")
        
        # 有效边只有 [safe]，所以期望值就是 ρ(safe)
        assert abs(rho - rho_safe) < 0.1, f"计算错误: {rho} != {rho_safe}"
    
    # 关键验证：在 danger 上时，robustness 应该是负的（引导远离）
    shaper.reset()
    state_danger = np.array([3, 3])
    rho_on_danger = shaper.compute_edge_based_robustness(state_danger)
    print(f"\n关键验证：在 danger 上时 rho={rho_on_danger:.2f}")
    assert rho_on_danger < 0, "在 danger 上时 robustness 应该是负的"
    
    print("\n✓ 测试通过!")


def test_progress_only_on_transition():
    """测试进度奖励只在 FSA 状态改变时给"""
    print("\n" + "=" * 60)
    print("测试 5: 进度奖励只在 FSA 状态改变时给")
    print("=" * 60)
    
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (5, 5),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    shaper = TaskRewardShaper("F(goal)", prop_manager)
    shaper.reset()
    
    print(f"\n公式: F(goal)")
    print(f"目标: (5, 5)")
    
    # 在目标外移动（FSA 状态不变）
    print(f"\n在目标外移动（FSA 状态不变）:")
    print("-" * 40)
    
    positions = [(0, 0), (1, 1), (2, 2), (3, 3)]
    
    for pos in positions:
        state = np.array(pos)
        reward, info = shaper.compute_reward(state)
        
        print(f"位置 {pos}: r_progress={info['r_progress']:.2f}, FSA={info['fsa_state']}")
        assert info['r_progress'] == 0.0, "FSA 状态不变时进度奖励应该为 0"
    
    # 到达目标（FSA 状态改变）
    print(f"\n到达目标（FSA 状态改变）:")
    print("-" * 40)
    
    state = np.array([5, 5])
    reward, info = shaper.compute_reward(state)
    
    print(f"位置 (5, 5): r_progress={info['r_progress']:.2f}, FSA={info['fsa_state']}")
    print(f"  is_accepting={info['is_accepting']}")
    
    assert info['r_progress'] > 0, "到达目标时应该有进度奖励"
    assert info['is_accepting'], "应该在接受状态"
    
    print("\n✓ 测试通过!")


def test_robustness_guides_toward_goal():
    """测试 robustness 能正确引导 agent 朝目标前进"""
    print("\n" + "=" * 60)
    print("测试 6: Robustness 引导方向")
    print("=" * 60)
    
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (5, 5),
        'grid_size': (10, 10)
    }
    prop_manager.update_env_info(env_info)
    
    shaper = TaskRewardShaper("F(goal)", prop_manager)
    
    print(f"\n公式: F(goal)")
    print(f"目标: (5, 5)")
    
    # 沿着朝目标的路径，robustness 应该递增
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    
    print(f"\n沿路径的 Robustness（应该递增）:")
    print("-" * 40)
    
    prev_rho = float('-inf')
    for pos in path:
        shaper.reset()
        state = np.array(pos)
        rho = shaper.compute_edge_based_robustness(state)
        
        increasing = "↑" if rho > prev_rho else "✗"
        print(f"位置 {pos}: rho={rho:.2f} {increasing}")
        
        assert rho > prev_rho, f"Robustness 应该递增: {prev_rho:.2f} -> {rho:.2f}"
        prev_rho = rho
    
    print("\n✓ 测试通过!")


if __name__ == '__main__':
    test_simple_eventually()
    test_multi_goal_or()
    test_sequential_task()
    test_until_formula()
    test_progress_only_on_transition()
    test_robustness_guides_toward_goal()
    
    print("\n" + "=" * 60)
    print("所有 TaskRewardShaper 测试通过！✓")
    print("=" * 60)
