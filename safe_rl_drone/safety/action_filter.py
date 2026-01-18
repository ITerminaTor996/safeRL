"""
安全过滤器模块

基于 FSA 的动作过滤器，支持任意 LTL 安全规范
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from ..ltl.fsa import FSAMonitor
from ..ltl.propositions import AtomicPropositionManager


class SafetyFilter:
    """
    安全过滤器（基于 FSA）
    
    职责：
    1. 预测动作的后果
    2. 检查是否违反安全规范（进入 FSA 陷阱状态）
    3. 如果不安全，选择最小干预的替代动作
    
    设计原则：
    - 不学习（无参数）
    - FSA 状态基于实际执行的动作更新
    - 命题评估基于真实状态
    """
    
    def __init__(self, 
                 safety_formula: str,
                 prop_manager: AtomicPropositionManager,
                 action_map: Dict,
                 num_actions: int = 5):
        """
        Args:
            safety_formula: 安全规范（如 "G(!wall) & G(!boundary)"）
            prop_manager: 原子命题管理器
            action_map: 动作到位移的映射 {0: (-1,0), 1: (1,0), ...}
            num_actions: 动作数量
        """
        self.safety_formula = safety_formula
        self.prop_manager = prop_manager
        self.action_map = action_map
        self.num_actions = num_actions
        
        # 创建 FSA 监控器
        self.fsa_monitor = FSAMonitor(safety_formula)
        self.fsa_monitor.set_prop_manager(prop_manager)
        
        # 统计
        self.intervention_count = 0
        self.total_actions = 0
        
        print(f"[SafetyFilter] 初始化完成")
        print(f"[SafetyFilter] 安全公式: {safety_formula}")
        if self.fsa_monitor.fsa:
            print(f"[SafetyFilter] FSA 状态数: {self.fsa_monitor.fsa.num_states}")
            print(f"[SafetyFilter] 陷阱状态: {self.fsa_monitor.fsa.trap_states}")
    
    def filter_action(self, 
                      current_state: np.ndarray, 
                      action: int) -> Tuple[int, bool, Dict]:
        """
        过滤动作
        
        流程：
        1. 预测下一状态
        2. 模拟 FSA 转移（不更新实际状态）
        3. 判断是否进入陷阱状态
        4. 如果不安全，找替代动作
        5. 根据实际执行的动作更新 FSA 状态
        
        Args:
            current_state: 当前状态 np.array([row, col])
            action: RL 输出的动作（int）
            
        Returns:
            safe_action: 安全的动作（int）
            filtered: 是否被过滤（bool）
            info: 调试信息（dict）
        """
        self.total_actions += 1
        
        # 保存当前 FSA 状态（用于模拟）
        saved_fsa_state = self.fsa_monitor.current_state
        
        # ========================================
        # Step 1: 预测下一个物理状态
        # ========================================
        next_state = self._predict_next_state(current_state, action)
        
        # ========================================
        # Step 2: 模拟 FSA 转移（不更新实际状态）
        # ========================================
        simulated_fsa_state, is_accepting, is_trap = self.fsa_monitor.step(next_state)
        
        # 恢复 FSA 状态（模拟不影响实际状态）
        self.fsa_monitor.current_state = saved_fsa_state
        
        # ========================================
        # Step 3: 判断是否安全
        # ========================================
        if is_trap:
            # 不安全，需要干预
            self.intervention_count += 1
            
            # 找到安全替代动作
            safe_action = self._find_safe_action(current_state)
            
            # 根据实际执行的动作更新 FSA 状态
            safe_next_state = self._predict_next_state(current_state, safe_action)
            actual_fsa_state, _, _ = self.fsa_monitor.step(safe_next_state)
            
            return safe_action, True, {
                'original_action': action,
                'filtered_action': safe_action,
                'reason': 'safety_violation',
                'fsa_state': actual_fsa_state,
                'intervention_count': self.intervention_count
            }
        else:
            # 安全，根据原动作更新 FSA 状态
            self.fsa_monitor.step(next_state)
            
            return action, False, {
                'fsa_state': simulated_fsa_state,
                'is_accepting': is_accepting
            }
    
    def _predict_next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        预测下一状态（离散网格版本）
        
        Args:
            state: 当前状态 [row, col]
            action: 动作 ID
            
        Returns:
            next_state: 预测的下一状态 [row, col]
        """
        delta = np.array(self.action_map[action])
        return state + delta
    
    def _find_safe_action(self, state: np.ndarray) -> int:
        """
        寻找安全替代动作（离散版本：最小干预）
        
        策略：
        1. 优先选择"不动"（action = 4）
        2. 如果"不动"也不安全，枚举其他动作
        3. 如果都不安全，仍返回"不动"（最保守）
        
        Args:
            state: 当前状态
            
        Returns:
            safe_action: 安全动作 ID
        """
        # 策略 1：优先尝试"不动"
        stay_action = 4
        if self._is_action_safe(state, stay_action):
            return stay_action
        
        # 策略 2：枚举其他动作
        for action in range(self.num_actions):
            if action == stay_action:
                continue
            if self._is_action_safe(state, action):
                return action
        
        # 策略 3：都不安全，返回"不动"（最保守）
        # 注：这种情况理论上不应该发生（当前状态安全，不动应该也安全）
        print(f"[SafetyFilter] 警告：所有动作都不安全，返回 stay")
        return stay_action
    
    def _is_action_safe(self, state: np.ndarray, action: int) -> bool:
        """
        检查动作是否安全（不会进入陷阱状态）
        
        Args:
            state: 当前状态
            action: 动作 ID
            
        Returns:
            is_safe: 是否安全
        """
        # 保存当前 FSA 状态
        saved_fsa_state = self.fsa_monitor.current_state
        
        # 预测下一状态
        next_state = self._predict_next_state(state, action)
        
        # 模拟 FSA 转移
        _, _, is_trap = self.fsa_monitor.step(next_state)
        
        # 恢复 FSA 状态
        self.fsa_monitor.current_state = saved_fsa_state
        
        return not is_trap
    
    def reset(self):
        """重置 FSA 状态和统计"""
        self.fsa_monitor.reset()
        self.intervention_count = 0
        self.total_actions = 0
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        intervention_rate = (
            self.intervention_count / self.total_actions 
            if self.total_actions > 0 else 0.0
        )
        return {
            'intervention_count': self.intervention_count,
            'total_actions': self.total_actions,
            'intervention_rate': intervention_rate,
            'fsa_state_info': self.fsa_monitor.get_state_info()
        }
    
    def get_safe_actions(self, current_state: np.ndarray) -> List[int]:
        """获取所有安全动作"""
        safe_actions = []
        for action in range(self.num_actions):
            if self._is_action_safe(current_state, action):
                safe_actions.append(action)
        return safe_actions


# 向后兼容：保留旧名称
ActionFilter = SafetyFilter
