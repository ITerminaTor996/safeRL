"""
任务奖励塑形器模块

基于 Task FSA + Robustness 计算多层次奖励
"""

import numpy as np
from typing import Dict, Optional

from ..ltl.fsa import FSAMonitor
from ..ltl.parser import LTLParser
from ..ltl.robustness import RobustnessCalculator
from ..ltl.propositions import AtomicPropositionManager


# 默认权重配置
DEFAULT_REWARD_WEIGHTS = {
    'robustness': 0.1,   # 边条件 Robustness（稠密引导）
    'progress': 1.0,     # FSA 进度（稀疏里程碑）
    'acceptance': 10.0,  # 任务完成（最终目标）
    'trap': 1.0,         # 陷阱惩罚（任务失败，-10.0）
    'filter': 1.0,       # 过滤器惩罚（学习安全，-0.5）
}


class TaskRewardShaper:
    """
    任务奖励塑形器：基于 Task FSA + Robustness 计算多层次奖励
    
    参考：Xiao Li et al., "A formal methods approach to interpretable 
          reinforcement learning for robotic planning", Science Robotics, 2019
    
    奖励组件（5 个部分）：
    1. Robustness 奖励（稠密）：r_rho = w_rho * edge_based_robustness
       - 基于当前 FSA 状态的出边条件计算
       - AND 取 min，OR 取 max，NOT 取负
    2. FSA 进度奖励（稀疏）：r_progress = w_progress * (value_t - value_{t+1})
       - 只在 FSA 状态改变时给予
    3. 任务完成奖励（稀疏）：r_accept = w_accept * is_accepting
    4. 陷阱状态惩罚（稀疏）：r_trap = w_trap * (-10.0) * entered_trap
    5. 过滤器惩罚（稀疏）：r_filter = w_filter * (-0.5) * filtered
    """
    
    def __init__(self, 
                 task_formula: str,
                 prop_manager: AtomicPropositionManager,
                 reward_weights: Optional[Dict] = None):
        """
        Args:
            task_formula: 任务规范字符串（如 "F(goal)"）
            prop_manager: 原子命题管理器
            reward_weights: 奖励权重配置（dict），如果为 None 使用默认值
        """
        self.task_formula = task_formula
        self.prop_manager = prop_manager
        self.weights = reward_weights if reward_weights else DEFAULT_REWARD_WEIGHTS.copy()
        
        # 构建 Task FSA（使用 ba 模式）
        self.fsa_monitor = FSAMonitor(task_formula, mode='ba')
        self.fsa_monitor.set_prop_manager(prop_manager)
        
        # 构建 Robustness 计算器
        parser = LTLParser()
        parser.parse(task_formula)
        self.rho_calculator = RobustnessCalculator(parser, prop_manager)
        
        # 预计算 FSA 状态价值（BFS 距离）
        self.state_values = self._compute_state_values()
        
        print(f"[TaskRewardShaper] 初始化完成")
        print(f"[TaskRewardShaper] 任务公式: {task_formula}")
        if self.fsa_monitor.fsa:
            print(f"[TaskRewardShaper] FSA 状态数: {self.fsa_monitor.fsa.num_states}")
            print(f"[TaskRewardShaper] 接受状态: {self.fsa_monitor.fsa.accepting_states}")
            print(f"[TaskRewardShaper] 陷阱状态: {self.fsa_monitor.fsa.trap_states}")
            print(f"[TaskRewardShaper] 状态价值: {self.state_values}")
    
    def compute_reward(self, 
                       state: np.ndarray, 
                       filtered: bool = False) -> tuple:
        """
        计算任务奖励
        
        Args:
            state: 当前状态（np.ndarray）
            filtered: 是否触发了安全过滤器
            
        Returns:
            reward: 总奖励（float）
            info: 调试信息（dict，包含 terminated 标志）
        """
        # 1. 基于边条件的 Robustness 奖励（稠密）
        # 参考：Xiao Li et al., Science Robotics 2019
        rho_t = self.compute_edge_based_robustness(state)
        r_rho = self.weights['robustness'] * rho_t
        
        # 2. FSA 进度奖励（稀疏，只在状态改变时给）
        prev_fsa_state = self.fsa_monitor.current_state
        new_fsa_state, is_accepting, is_trap = self.fsa_monitor.step(state)
        
        # 只在 FSA 状态改变且非陷阱时给进度奖励
        if new_fsa_state != prev_fsa_state and not is_trap:
            prev_value = self.state_values.get(prev_fsa_state, 0)
            new_value = self.state_values.get(new_fsa_state, 0)
            r_progress = self.weights['progress'] * (prev_value - new_value)
        else:
            r_progress = 0.0
        
        # 3. 任务完成奖励
        r_accept = self.weights['acceptance'] * (1.0 if is_accepting else 0.0)
        
        # 4. 陷阱状态惩罚
        if is_trap:
            r_trap = self.weights['trap'] * (-10.0)
        else:
            r_trap = 0.0
        
        # 5. 过滤器惩罚
        r_filter = self.weights['filter'] * (-0.5 if filtered else 0.0)
        
        # 总奖励（去掉了时间惩罚）
        total_reward = r_rho + r_progress + r_accept + r_trap + r_filter
        
        # 返回结果（如果进入陷阱，标记 terminated）
        return total_reward, {
            'r_rho': r_rho,
            'r_progress': r_progress,
            'r_accept': r_accept,
            'r_trap': r_trap,
            'r_filter': r_filter,
            'rho': rho_t,
            'fsa_state': new_fsa_state,
            'is_accepting': is_accepting,
            'is_trap': is_trap,
            'terminated': is_trap,  # 进入陷阱时终止 episode
            'reason': 'task_trap_state' if is_trap else None
        }
    
    def compute_edge_based_robustness(self, state: np.ndarray) -> float:
        """
        基于 FSA 转移边计算 robustness
        
        参考：Xiao Li et al., Science Robotics 2019, Equation (11)
        
        r = max_{有效边i} condition_robustness(边i.label)
        
        有效边的定义：
        1. 非自环边（source != target）
        2. 目标状态不是陷阱状态
        
        对于每条出边，计算其条件的 robustness：
        - AND (a & b): min(ρ(a), ρ(b))
        - OR (a | b): max(ρ(a), ρ(b))
        - NOT (!a): -ρ(a)
        
        最终取所有有效边的 max
        
        Args:
            state: 环境状态
            
        Returns:
            robustness 值
        """
        if self.fsa_monitor.fsa is None:
            return 0.0
        
        current_fsa_state = self.fsa_monitor.current_state
        edges = self.fsa_monitor.fsa.get_transitions_from(current_fsa_state)
        
        if not edges:
            return 0.0  # 没有出边（不应该发生）
        
        # 只考虑有效边：
        # 1. 非自环（能推进 FSA 状态）
        # 2. 目标不是陷阱状态（不会导致任务失败）
        trap_states = self.fsa_monitor.fsa.trap_states
        valid_edges = [
            e for e in edges 
            if e.target != e.source and e.target not in trap_states
        ]
        
        if not valid_edges:
            # 没有有效边（已经在接受状态或只能进入陷阱）
            # 返回 0，不提供额外引导
            return 0.0
        
        edge_robustness = []
        for edge in valid_edges:
            # 计算这条边条件的 robustness
            cond_rho = self.fsa_monitor.compute_condition_robustness(edge.label, state)
            edge_robustness.append(cond_rho)
        
        # 取所有有效边的 max
        return max(edge_robustness)
    
    def _compute_state_values(self) -> Dict[int, int]:
        """
        BFS 计算状态价值（到接受状态的距离）
        
        Returns:
            state_values: {state_id: distance}
        """
        if self.fsa_monitor.fsa is None:
            return {}
        
        fsa = self.fsa_monitor.fsa
        accepting_states = fsa.accepting_states
        
        # 从接受状态反向 BFS
        values = {}
        queue = [(s, 0) for s in accepting_states]
        visited = set(accepting_states)
        
        while queue:
            state, dist = queue.pop(0)
            values[state] = dist  # 距离越近，价值越大（使用正数）
            
            # 找到所有能到达当前状态的前驱状态
            for trans in fsa.transitions:
                if trans.target == state and trans.source not in visited:
                    visited.add(trans.source)
                    queue.append((trans.source, dist + 1))
        
        return values
    
    def reset(self):
        """重置状态"""
        self.fsa_monitor.reset()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'fsa_state': self.fsa_monitor.current_state,
            'is_accepting': self.fsa_monitor.is_accepting(),
            'is_trap': not self.fsa_monitor.is_safe(),
            'prev_rho': self.prev_rho,
            'state_values': self.state_values
        }
