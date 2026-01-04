"""
Robustness（鲁棒度）计算模块

基于 TLTL 的量化语义，计算 LTL 公式的 robustness 值：
- ρ > 0: 满足公式
- ρ < 0: 违反公式
- |ρ| 越大，满足/违反程度越高
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from .parser import LTLNode, LTLOperator, LTLParser
from .propositions import AtomicPropositionManager


class RobustnessCalculator:
    """Robustness 计算器"""
    
    def __init__(self, 
                 parser: LTLParser,
                 prop_manager: AtomicPropositionManager):
        """
        Args:
            parser: LTL 解析器（已解析公式）
            prop_manager: 原子命题管理器
        """
        self.parser = parser
        self.prop_manager = prop_manager
        self.rho_max = 100.0  # 最大 robustness 值
    
    def compute(self, 
                state: np.ndarray,
                trajectory: Optional[List[np.ndarray]] = None) -> float:
        """
        计算当前状态（或轨迹）的 robustness
        
        Args:
            state: 当前状态
            trajectory: 状态轨迹（用于时序算子）
                       如果为 None，只计算当前状态的 robustness
        
        Returns:
            robustness 值
        """
        if self.parser.syntax_tree is None:
            raise ValueError("No formula parsed")
        
        if trajectory is None:
            trajectory = [state]
        
        return self._compute_node(self.parser.syntax_tree, trajectory, 0)
    
    def _compute_node(self, 
                      node: LTLNode, 
                      trajectory: List[np.ndarray],
                      t: int) -> float:
        """
        递归计算语法树节点的 robustness
        
        Args:
            node: 语法树节点
            trajectory: 状态轨迹
            t: 当前时间步
        
        Returns:
            robustness 值
        """
        T = len(trajectory)
        
        if t >= T:
            # 超出轨迹范围，返回边界值
            return -self.rho_max
        
        state = trajectory[t]
        
        # 原子命题
        if node.operator == LTLOperator.ATOM:
            if node.value == 'true':
                return self.rho_max
            elif node.value == 'false':
                return -self.rho_max
            else:
                return self.prop_manager.robustness(node.value, state)
        
        # 否定
        elif node.operator == LTLOperator.NOT:
            child_rho = self._compute_node(node.children[0], trajectory, t)
            return -child_rho
        
        # 与
        elif node.operator == LTLOperator.AND:
            rho_left = self._compute_node(node.children[0], trajectory, t)
            rho_right = self._compute_node(node.children[1], trajectory, t)
            return min(rho_left, rho_right)
        
        # 或
        elif node.operator == LTLOperator.OR:
            rho_left = self._compute_node(node.children[0], trajectory, t)
            rho_right = self._compute_node(node.children[1], trajectory, t)
            return max(rho_left, rho_right)
        
        # 蕴含: φ -> ψ ≡ !φ | ψ
        elif node.operator == LTLOperator.IMPLIES:
            rho_left = self._compute_node(node.children[0], trajectory, t)
            rho_right = self._compute_node(node.children[1], trajectory, t)
            return max(-rho_left, rho_right)
        
        # Next: X(φ) - 下一时刻 φ 成立
        elif node.operator == LTLOperator.NEXT:
            if t + 1 >= T:
                return -self.rho_max  # 没有下一时刻
            return self._compute_node(node.children[0], trajectory, t + 1)
        
        # Globally: G(φ) - 从 t 开始所有时刻 φ 都成立
        # ρ(G(φ)) = min_{t' >= t} ρ(φ, t')
        elif node.operator == LTLOperator.GLOBALLY:
            rho_values = []
            for t_prime in range(t, T):
                rho = self._compute_node(node.children[0], trajectory, t_prime)
                rho_values.append(rho)
            
            if not rho_values:
                return self.rho_max  # 空轨迹，默认满足
            return min(rho_values)
        
        # Eventually: F(φ) - 从 t 开始某个时刻 φ 成立
        # ρ(F(φ)) = max_{t' >= t} ρ(φ, t')
        elif node.operator == LTLOperator.EVENTUALLY:
            rho_values = []
            for t_prime in range(t, T):
                rho = self._compute_node(node.children[0], trajectory, t_prime)
                rho_values.append(rho)
            
            if not rho_values:
                return -self.rho_max  # 空轨迹，默认不满足
            return max(rho_values)
        
        # Until: φ U ψ - φ 一直成立直到 ψ 成立
        # ρ(φ U ψ) = max_{t' >= t} min(ρ(ψ, t'), min_{t'' in [t, t')} ρ(φ, t''))
        elif node.operator == LTLOperator.UNTIL:
            best_rho = -self.rho_max
            
            for t_prime in range(t, T):
                # ψ 在 t' 成立
                rho_psi = self._compute_node(node.children[1], trajectory, t_prime)
                
                # φ 在 [t, t') 都成立
                if t_prime == t:
                    rho_phi_min = self.rho_max  # 空区间
                else:
                    rho_phi_values = []
                    for t_double_prime in range(t, t_prime):
                        rho_phi = self._compute_node(node.children[0], trajectory, t_double_prime)
                        rho_phi_values.append(rho_phi)
                    rho_phi_min = min(rho_phi_values)
                
                rho_until = min(rho_psi, rho_phi_min)
                best_rho = max(best_rho, rho_until)
            
            return best_rho
        
        # 未知算子
        return 0.0
    
    def compute_safety(self, state: np.ndarray) -> float:
        """
        计算安全相关的 robustness（只看当前状态）
        
        用于实时安全检查，不需要完整轨迹
        """
        return self.compute(state, trajectory=[state])
    
    def is_satisfied(self, 
                     state: np.ndarray,
                     trajectory: Optional[List[np.ndarray]] = None) -> bool:
        """检查公式是否满足"""
        return self.compute(state, trajectory) > 0


class TrajectoryRobustness:
    """轨迹 Robustness 计算（用于奖励塑形）"""
    
    def __init__(self, calculator: RobustnessCalculator):
        self.calculator = calculator
        self.trajectory: List[np.ndarray] = []
    
    def reset(self):
        """重置轨迹"""
        self.trajectory = []
    
    def add_state(self, state: np.ndarray):
        """添加状态到轨迹"""
        self.trajectory.append(state.copy())
    
    def get_current_robustness(self) -> float:
        """获取当前轨迹的 robustness"""
        if not self.trajectory:
            return 0.0
        return self.calculator.compute(self.trajectory[-1], self.trajectory)
    
    def get_incremental_robustness(self) -> float:
        """
        获取增量 robustness（用于奖励）
        
        返回添加最新状态后 robustness 的变化
        """
        if len(self.trajectory) < 2:
            return 0.0
        
        # 当前 robustness
        current_rho = self.calculator.compute(
            self.trajectory[-1], self.trajectory
        )
        
        # 上一步 robustness
        prev_rho = self.calculator.compute(
            self.trajectory[-2], self.trajectory[:-1]
        )
        
        return current_rho - prev_rho
                                                      

# ============================================================
# 测试代码
# ============================================================

def test_robustness():
    """测试 Robustness 计算"""
    print("=" * 60)
    print("测试 Robustness 计算模块")
    print("=" * 60)
    
    # 设置原子命题
    from .propositions import AtomicPropositionManager, PositionProposition
    
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
    
    # 测试不同公式
    test_cases = [
        ("G(!wall)", [
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([0, 2]),
        ]),
        ("F(goal)", [
            np.array([0, 0]),
            np.array([1, 1]),  # 墙！
            np.array([2, 2]),
            np.array([3, 3]),
            np.array([4, 4]),  # 目标
        ]),
        ("G(!wall) & F(goal)", [
            np.array([0, 0]),
            np.array([0, 3]),
            np.array([3, 3]),
            np.array([4, 4]),
        ]),
        ("(!goal) U checkpoint", [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([2, 2]),  # checkpoint
            np.array([4, 4]),  # goal
        ]),
    ]
    
    parser = LTLParser()
    
    for formula, trajectory in test_cases:
        print(f"\n公式: {formula}")
        print(f"轨迹: {[tuple(s) for s in trajectory]}")
        
        parser.parse(formula)
        calculator = RobustnessCalculator(parser, prop_manager)
        
        # 计算每个时间步的 robustness
        for t, state in enumerate(trajectory):
            rho = calculator.compute(state, trajectory[:t+1])
            print(f"  t={t}, state={tuple(state)}, ρ={rho:.2f}")
        
        # 完整轨迹的 robustness
        final_rho = calculator.compute(trajectory[-1], trajectory)
        satisfied = calculator.is_satisfied(trajectory[-1], trajectory)
        print(f"  最终 ρ={final_rho:.2f}, 满足={satisfied}")
    
    print("\n" + "=" * 60)
    print("Robustness 计算模块测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    test_robustness()
