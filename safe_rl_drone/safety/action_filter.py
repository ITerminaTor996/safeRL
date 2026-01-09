"""
动作过滤模块

当动作不安全时，选择安全的替代动作
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .monitor import SafetyMonitor


class ActionFilter:
    """
    动作过滤器
    
    功能：
    1. 检查动作是否安全
    2. 如果不安全，选择最佳替代动作
    3. 记录干预统计
    """
    
    def __init__(self, 
                 safety_monitor: SafetyMonitor,
                 action_map: Dict,
                 num_actions: int = 5):
        """
        Args:
            safety_monitor: 安全监控器
            action_map: 动作到位移的映射
            num_actions: 动作数量
        """
        self.safety_monitor = safety_monitor
        self.action_map = action_map
        self.num_actions = num_actions
        
        # 统计
        self.intervention_count = 0
        self.total_actions = 0
    
    def reset(self):
        """重置统计"""
        self.intervention_count = 0
        self.total_actions = 0
    
    def filter_action(self, 
                      current_state: np.ndarray, 
                      action: int) -> Tuple[int, bool, Dict]:
        """
        过滤动作，如果不安全则替换
        
        Args:
            current_state: 当前状态
            action: 原始动作
            
        Returns:
            (filtered_action, was_modified, info)
        """
        self.total_actions += 1
        
        # 检查原始动作是否安全
        is_safe, rho, reason = self.safety_monitor.is_action_safe(
            current_state, action, self.action_map
        )
        
        if is_safe:
            return action, False, {
                'original_action': action,
                'robustness': rho
            }
        
        # 动作不安全，需要干预
        self.intervention_count += 1
        
        # 找到最佳替代动作
        best_action, best_rho = self._find_best_safe_action(current_state, action)
        
        return best_action, True, {
            'original_action': action,
            'filtered_action': best_action,
            'original_robustness': rho,
            'filtered_robustness': best_rho,
            'reason': reason,
            'intervention_count': self.intervention_count
        }
    
    def _find_best_safe_action(self, 
                               current_state: np.ndarray,
                               original_action: int) -> Tuple[int, float]:
        """
        找到最佳安全替代动作
        
        策略：
        1. 优先选择与原动作方向最接近的安全动作
        2. 如果都不安全，选择"原地不动"
        3. 如果原地不动也不安全，选择 robustness 最高的动作
        """
        safe_actions = []
        all_actions_rho = []
        
        for action in range(self.num_actions):
            is_safe, rho, _ = self.safety_monitor.is_action_safe(
                current_state, action, self.action_map
            )
            all_actions_rho.append((action, rho, is_safe))
            
            if is_safe:
                safe_actions.append((action, rho))
        
        if not safe_actions:
            # 没有安全动作，选择 robustness 最高的（伤害最小）
            best = max(all_actions_rho, key=lambda x: x[1])
            return best[0], best[1]
        
        # 有安全动作，选择策略
        # 策略 1：优先选择"原地不动"（动作 4）
        stay_action = 4
        for action, rho in safe_actions:
            if action == stay_action:
                return action, rho
        
        # 策略 2：选择 robustness 最高的安全动作
        best = max(safe_actions, key=lambda x: x[1])
        return best[0], best[1]
    
    def get_safe_actions(self, current_state: np.ndarray) -> List[int]:
        """获取所有安全动作"""
        safe_actions = []
        for action in range(self.num_actions):
            is_safe, _, _ = self.safety_monitor.is_action_safe(
                current_state, action, self.action_map
            )
            if is_safe:
                safe_actions.append(action)
        return safe_actions
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        intervention_rate = (
            self.intervention_count / self.total_actions 
            if self.total_actions > 0 else 0.0
        )
        return {
            'intervention_count': self.intervention_count,
            'total_actions': self.total_actions,
            'intervention_rate': intervention_rate
        }
