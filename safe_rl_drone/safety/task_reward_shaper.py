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
    'robustness': 0.1,   # Robustness 增量（稠密引导）
    'progress': 1.0,     # FSA 进度（稀疏里程碑）
    'acceptance': 10.0,  # 任务完成（最终目标）
    'trap': 1.0,         # 陷阱惩罚（任务失败，-10.0）
    'filter': 1.0,       # 过滤器惩罚（学习安全，-0.5）
    'time': 1.0          # 时间惩罚（鼓励效率，-0.01）
}


class TaskRewardShaper:
    """
    任务奖励塑形器：基于 Task FSA + Robustness 计算多层次奖励
    
    奖励组件（6 个部分）：
    1. Robustness 增量奖励（稠密）：r_rho = w_rho * (ρ_t - ρ_{t-1})
    2. FSA 进度奖励（稀疏）：r_progress = w_progress * (value_t - value_{t+1})
    3. 任务完成奖励（稀疏）：r_accept = w_accept * is_accepting
    4. 陷阱状态惩罚（稀疏）：r_trap = w_trap * (-10.0) * entered_trap
    5. 过滤器惩罚（稀疏）：r_filter = w_filter * (-0.5) * filtered
    6. 时间惩罚（稠密）：r_time = w_time * (-0.01)
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
        
        # 历史状态（用于计算增量）
        self.prev_state = None
        self.prev_rho = None
        
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
        # 1. Robustness 增量奖励
        rho_t = self.rho_calculator.compute(state, trajectory=[state])
        if self.prev_rho is not None:
            r_rho = self.weights['robustness'] * (rho_t - self.prev_rho)
        else:
            r_rho = 0.0
        
        # 2. FSA 进度奖励
        prev_fsa_state = self.fsa_monitor.current_state
        new_fsa_state, is_accepting, is_trap = self.fsa_monitor.step(state)
        
        # 2.1 FSA 进度奖励（只在非陷阱状态计算）
        if not is_trap:
            prev_value = self.state_values.get(prev_fsa_state, 0)
            new_value = self.state_values.get(new_fsa_state, 0)
            r_progress = self.weights['progress'] * (prev_value - new_value)
        else:
            # 陷阱状态没有进度奖励（已经有陷阱惩罚了）
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
        
        # 6. 时间惩罚
        r_time = self.weights['time'] * (-0.01)
        
        # 总奖励
        total_reward = r_rho + r_progress + r_accept + r_trap + r_filter + r_time
        
        # 更新历史
        self.prev_state = state.copy()
        self.prev_rho = rho_t
        
        # 返回结果（如果进入陷阱，标记 terminated）
        return total_reward, {
            'r_rho': r_rho,
            'r_progress': r_progress,
            'r_accept': r_accept,
            'r_trap': r_trap,
            'r_filter': r_filter,
            'r_time': r_time,
            'rho': rho_t,
            'fsa_state': new_fsa_state,
            'is_accepting': is_accepting,
            'is_trap': is_trap,
            'terminated': is_trap,  # 进入陷阱时终止 episode
            'reason': 'task_trap_state' if is_trap else None
        }
    
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
        self.prev_state = None
        self.prev_rho = None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'fsa_state': self.fsa_monitor.current_state,
            'is_accepting': self.fsa_monitor.is_accepting(),
            'is_trap': not self.fsa_monitor.is_safe(),
            'prev_rho': self.prev_rho,
            'state_values': self.state_values
        }
