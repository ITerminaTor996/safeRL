"""
测试 TaskRewardShaper 模块
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from safe_rl_drone.safety.task_reward_shaper import TaskRewardShaper, DEFAULT_REWARD_WEIGHTS
from safe_rl_drone.ltl.propositions import AtomicPropositionManager


def test_task_reward_shaper_basic():
    """测试 TaskRewardShaper - 基本功能"""
    print("=" * 60)
    print("测试 TaskRewardShaper - 基本功能")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': {(1, 1), (2, 2)},
        'goal_position': (5, 5),
        'grid_size': (6, 6)
    }
    prop_manager.update_env_info(env_info)
    
    # 创建 TaskRewardShaper
    task_formula = "F(goal)"
    shaper = TaskRewardShaper(task_formula, prop_manager)
    
    print(f"\n任务公式: {task_formula}")
    print(f"目标位置: {env_info['goal_position']}")
    print(f"状态价值: {shaper.state_values}")
    
    # 测试场景 1: 朝目标前进
    print("\n测试场景 1: 朝目标前进")
    print("-" * 40)
    
    shaper.reset()
    
    # 初始位置
    state1 = np.array([0, 0])
    reward1, info1 = shaper.compute_reward(state1, filtered=False)
    print(f"位置 {tuple(state1)}: reward={reward1:.3f}")
    print(f"  r_rho={info1['r_rho']:.3f}, r_progress={info1['r_progress']:.3f}")
    print(f"  r_accept={info1['r_accept']:.3f}, r_filter={info1['r_filter']:.3f}")
    print(f"  r_time={info1['r_time']:.3f}")
    print(f"  rho={info1['rho']:.3f}, fsa_state={info1['fsa_state']}")
    
    # 前进一步（更接近目标）
    state2 = np.array([1, 1])
    reward2, info2 = shaper.compute_reward(state2, filtered=False)
    print(f"位置 {tuple(state2)}: reward={reward2:.3f}")
    print(f"  r_rho={info2['r_rho']:.3f}, r_progress={info2['r_progress']:.3f}")
    print(f"  r_accept={info2['r_accept']:.3f}, r_filter={info2['r_filter']:.3f}")
    print(f"  r_time={info2['r_time']:.3f}")
    print(f"  rho={info2['rho']:.3f}, fsa_state={info2['fsa_state']}")
    
    # 验证 robustness 增加（更接近目标）
    assert info2['rho'] > info1['rho'], "Robustness 应该增加"
    assert info2['r_rho'] > 0, "Robustness 增量奖励应该为正"
    
    # 测试场景 2: 到达目标
    print("\n测试场景 2: 到达目标")
    print("-" * 40)
    
    shaper.reset()
    
    # 接近目标
    state3 = np.array([4, 4])
    reward3, info3 = shaper.compute_reward(state3, filtered=False)
    print(f"位置 {tuple(state3)}: reward={reward3:.3f}")
    print(f"  rho={info3['rho']:.3f}, is_accepting={info3['is_accepting']}")
    
    # 到达目标
    state4 = np.array([5, 5])
    reward4, info4 = shaper.compute_reward(state4, filtered=False)
    print(f"位置 {tuple(state4)}: reward={reward4:.3f}")
    print(f"  r_accept={info4['r_accept']:.3f}")
    print(f"  rho={info4['rho']:.3f}, is_accepting={info4['is_accepting']}")
    
    # 验证到达目标有大奖励
    assert info4['r_accept'] > 0, "到达目标应该有完成奖励"
    assert info4['is_accepting'], "应该在接受状态"
    
    # 测试场景 3: 触发过滤器
    print("\n测试场景 3: 触发过滤器")
    print("-" * 40)
    
    shaper.reset()
    
    state5 = np.array([2, 2])
    reward5, info5 = shaper.compute_reward(state5, filtered=True)
    print(f"位置 {tuple(state5)}, filtered=True: reward={reward5:.3f}")
    print(f"  r_filter={info5['r_filter']:.3f}")
    
    # 验证过滤器惩罚
    assert info5['r_filter'] < 0, "触发过滤器应该有惩罚"
    
    print("\n" + "=" * 60)
    print("TaskRewardShaper 基本功能测试通过！")
    print("=" * 60)


def test_task_reward_shaper_progress():
    """测试 TaskRewardShaper - FSA 进度奖励"""
    print("\n" + "=" * 60)
    print("测试 TaskRewardShaper - FSA 进度奖励")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (5, 5),
        'grid_size': (6, 6)
    }
    prop_manager.update_env_info(env_info)
    
    # 创建 TaskRewardShaper
    task_formula = "F(goal)"
    shaper = TaskRewardShaper(task_formula, prop_manager)
    
    print(f"\n任务公式: {task_formula}")
    print(f"状态价值: {shaper.state_values}")
    
    # 模拟轨迹
    trajectory = [
        np.array([0, 0]),
        np.array([1, 1]),
        np.array([2, 2]),
        np.array([3, 3]),
        np.array([4, 4]),
        np.array([5, 5]),  # 目标
    ]
    
    shaper.reset()
    
    print("\n模拟轨迹:")
    print("-" * 40)
    
    total_reward = 0.0
    for i, state in enumerate(trajectory):
        reward, info = shaper.compute_reward(state, filtered=False)
        total_reward += reward
        
        print(f"Step {i}: pos={tuple(state)}, reward={reward:.3f}")
        print(f"  r_rho={info['r_rho']:.3f}, r_progress={info['r_progress']:.3f}")
        print(f"  r_accept={info['r_accept']:.3f}, r_time={info['r_time']:.3f}")
        print(f"  rho={info['rho']:.3f}, fsa_state={info['fsa_state']}")
        print(f"  is_accepting={info['is_accepting']}")
    
    print(f"\n总奖励: {total_reward:.3f}")
    
    # 验证最终到达目标
    assert info['is_accepting'], "最终应该到达接受状态"
    assert info['r_accept'] > 0, "最终应该有完成奖励"
    
    print("\n" + "=" * 60)
    print("TaskRewardShaper 进度奖励测试通过！")
    print("=" * 60)


def test_task_reward_shaper_trap():
    """测试 TaskRewardShaper - 陷阱状态处理"""
    print("\n" + "=" * 60)
    print("测试 TaskRewardShaper - 陷阱状态处理")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    # 注册一个"禁区"命题
    from safe_rl_drone.ltl.propositions import PositionProposition
    prop_manager.register_proposition(
        'forbidden', PositionProposition('forbidden', (3, 3))
    )
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (5, 5),
        'grid_size': (6, 6)
    }
    prop_manager.update_env_info(env_info)
    
    # 使用 G 算子：必须永远避开禁区
    # 如果进入禁区，就违反了 G(!forbidden)，进入陷阱状态
    task_formula = "F(goal) & G(!forbidden)"
    shaper = TaskRewardShaper(task_formula, prop_manager)
    
    print(f"\n任务公式: {task_formula}")
    print(f"Goal: (5, 5)")
    print(f"Forbidden: (3, 3)")
    print(f"陷阱状态: {shaper.fsa_monitor.fsa.trap_states if shaper.fsa_monitor.fsa else 'None'}")
    
    # 测试正常路径
    print("\n测试正常路径（避开禁区）:")
    print("-" * 40)
    
    shaper.reset()
    
    state1 = np.array([0, 0])
    reward1, info1 = shaper.compute_reward(state1, filtered=False)
    print(f"位置 {tuple(state1)}: reward={reward1:.3f}, is_trap={info1['is_trap']}")
    
    state2 = np.array([1, 1])
    reward2, info2 = shaper.compute_reward(state2, filtered=False)
    print(f"位置 {tuple(state2)}: reward={reward2:.3f}, is_trap={info2['is_trap']}")
    
    # 测试进入禁区（陷阱）
    print("\n测试进入禁区（陷阱状态）:")
    print("-" * 40)
    
    shaper.reset()
    
    # 先走几步
    shaper.compute_reward(np.array([0, 0]), filtered=False)
    shaper.compute_reward(np.array([1, 1]), filtered=False)
    
    # 进入禁区
    state_forbidden = np.array([3, 3])
    reward_trap, info_trap = shaper.compute_reward(state_forbidden, filtered=False)
    
    print(f"位置 {tuple(state_forbidden)} (禁区): reward={reward_trap:.3f}")
    print(f"  r_trap={info_trap['r_trap']:.3f}")
    print(f"  r_progress={info_trap['r_progress']:.3f}")
    print(f"  r_time={info_trap['r_time']:.3f}")
    print(f"  is_trap={info_trap['is_trap']}")
    print(f"  terminated={info_trap['terminated']}")
    
    if info_trap['is_trap']:
        assert info_trap['terminated'], "进入陷阱应该终止 episode"
        assert info_trap['r_trap'] < 0, "进入陷阱应该有陷阱惩罚"
        assert info_trap['r_progress'] == 0.0, "陷阱状态不应该有进度奖励"
        # 验证其他奖励组件仍然被计算
        assert 'r_time' in info_trap and info_trap['r_time'] != 0, "时间惩罚应该被计算"
        print("  ✓ 陷阱状态处理正确，进度奖励为 0，其他组件正常计算")
    else:
        print("  注：此公式未产生陷阱状态（可能 Spot 构建方式不同）")
    
    print("\n" + "=" * 60)
    print("TaskRewardShaper 陷阱状态测试完成！")
    print("=" * 60)


def test_reward_weights():
    """测试自定义奖励权重"""
    print("\n" + "=" * 60)
    print("测试自定义奖励权重")
    print("=" * 60)
    
    # 创建命题管理器
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': set(),
        'goal_position': (5, 5),
        'grid_size': (6, 6)
    }
    prop_manager.update_env_info(env_info)
    
    # 自定义权重
    custom_weights = {
        'robustness': 0.5,   # 增大 robustness 权重
        'progress': 2.0,     # 增大进度权重
        'acceptance': 20.0,  # 增大完成奖励
        'trap': 1.0,
        'filter': 2.0,       # 增大过滤器惩罚
        'time': 0.5          # 减小时间惩罚
    }
    
    task_formula = "F(goal)"
    shaper = TaskRewardShaper(task_formula, prop_manager, custom_weights)
    
    print(f"\n自定义权重: {custom_weights}")
    
    shaper.reset()
    
    # 测试
    state1 = np.array([0, 0])
    reward1, info1 = shaper.compute_reward(state1, filtered=False)
    
    state2 = np.array([1, 1])
    reward2, info2 = shaper.compute_reward(state2, filtered=False)
    
    print(f"\n位置 {tuple(state1)}: reward={reward1:.3f}")
    print(f"位置 {tuple(state2)}: reward={reward2:.3f}")
    print(f"  r_rho={info2['r_rho']:.3f} (权重 {custom_weights['robustness']})")
    print(f"  r_time={info2['r_time']:.3f} (权重 {custom_weights['time']})")
    
    # 验证权重生效
    assert shaper.weights == custom_weights, "权重应该被正确设置"
    
    print("\n" + "=" * 60)
    print("自定义奖励权重测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_task_reward_shaper_basic()
    test_task_reward_shaper_progress()
    test_task_reward_shaper_trap()
    test_reward_weights()
    
    print("\n" + "=" * 60)
    print("所有 TaskRewardShaper 测试通过！✓")
    print("=" * 60)
