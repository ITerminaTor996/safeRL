"""
安全监控模块

使用 LTL 公式和 Robustness 进行安全监控
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..ltl import (
    AtomicPropositionManager,
    LTLParser,
    RobustnessCalculator,
    FSAMonitor
)
from ..ltl.propositions import PositionProposition, RegionProposition


class SafetyMonitor:
    """
    安全监控器
    
    功能：
    1. 解析 LTL 安全公式
    2. 评估动作是否安全
    3. 计算 robustness 值
    4. 追踪轨迹
    """
    
    def __init__(self, 
                 safety_formula: str,
                 env_info: Dict,
                 config: Dict = None):
        """
        Args:
            safety_formula: LTL 安全公式，如 "G(!wall)"
            env_info: 环境信息，包含 wall_positions, goal_position 等
            config: 配置字典，包含原子命题定义等
        """
        self.safety_formula = safety_formula
        self.env_info = env_info
        self.config = config or {}
        
        # 初始化原子命题管理器
        self.prop_manager = AtomicPropositionManager()
        self._setup_propositions()
        self.prop_manager.update_env_info(env_info)
        
        # 解析公式
        self.parser = LTLParser()
        self.parser.parse(safety_formula)
        
        # 创建 Robustness 计算器
        self.robustness_calc = RobustnessCalculator(self.parser, self.prop_manager)
        
        # 轨迹记录
        self.trajectory: List[np.ndarray] = []
        
        # 统计信息
        self.stats = {
            'violations': 0,
            'interventions': 0,
            'total_steps': 0
        }
        
        print(f"[SafetyMonitor] 初始化完成")
        print(f"[SafetyMonitor] 安全公式: {safety_formula}")
        print(f"[SafetyMonitor] 原子命题: {self.parser.get_atomic_propositions()}")
    
    def _setup_propositions(self):
        """设置原子命题"""
        # 注册环境自动提供的命题
        self.prop_manager.register_auto_propositions()
        
        # 从配置注册自定义命题
        ap_config = self.config.get('atomic_propositions', {})
        for name, definition in ap_config.items():
            if definition == 'auto':
                continue  # 已经注册过了
            
            if isinstance(definition, dict):
                prop_type = definition.get('type')
                
                if prop_type == 'position':
                    pos = tuple(definition['pos'])
                    self.prop_manager.register_proposition(
                        name, PositionProposition(name, pos)
                    )
                
                elif prop_type == 'region':
                    positions = definition['positions']
                    avoid = definition.get('avoid', False)
                    self.prop_manager.register_proposition(
                        name, RegionProposition(name, positions, avoid)
                    )
    
    def update_env_info(self, env_info: Dict):
        """更新环境信息"""
        self.env_info = env_info
        self.prop_manager.update_env_info(env_info)
    
    def reset(self):
        """重置监控器"""
        self.trajectory = []
        self.stats = {
            'violations': 0,
            'interventions': 0,
            'total_steps': 0
        }
    
    def is_state_safe(self, state: np.ndarray) -> Tuple[bool, float]:
        """
        检查状态是否安全
        
        Args:
            state: 当前状态
            
        Returns:
            (is_safe, robustness)
        """
        # 计算当前状态的 robustness
        rho = self.robustness_calc.compute_safety(state)
        is_safe = rho > 0
        
        return is_safe, rho
    
    def is_action_safe(self, 
                       current_state: np.ndarray, 
                       action: int,
                       action_map: Dict) -> Tuple[bool, float, str]:
        """
        检查动作是否安全
        
        Args:
            current_state: 当前状态
            action: 要执行的动作
            action_map: 动作到位移的映射
            
        Returns:
            (is_safe, robustness, reason)
        """
        # 计算下一个状态
        move = action_map.get(action, np.array([0, 0]))
        next_state = current_state + move
        
        # 检查下一状态的安全性（包括边界检查，通过 boundary 命题）
        is_safe, rho = self.is_state_safe(next_state)
        
        if not is_safe:
            # 检查具体原因
            rows, cols = self.env_info.get('grid_size', (10, 10))
            r, c = next_state
            
            # 先检查是否越界
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return False, rho, "越界"
            
            # 再检查是否撞墙
            wall_positions = self.env_info.get('wall_positions', set())
            if tuple(next_state) in wall_positions:
                return False, rho, "撞墙"
            
            return False, rho, "违反安全约束"
        
        return True, rho, ""
    
    def step(self, state: np.ndarray) -> Dict:
        """
        记录一步并返回监控信息
        
        Args:
            state: 当前状态
            
        Returns:
            监控信息字典
        """
        self.trajectory.append(state.copy())
        self.stats['total_steps'] += 1
        
        # 对于安全公式 G(!wall)，只需计算当前状态的 robustness
        # 这样避免了对整个轨迹的递归计算，大幅提升性能
        current_rho = self.robustness_calc.compute_safety(state)
        
        # 计算增量 robustness（用于奖励塑形）
        if len(self.trajectory) >= 2:
            prev_rho = self.robustness_calc.compute_safety(self.trajectory[-2])
            delta_rho = current_rho - prev_rho
        else:
            delta_rho = 0.0
        
        return {
            'robustness': current_rho,
            'delta_robustness': delta_rho,
            'trajectory_length': len(self.trajectory),
            'is_safe': current_rho > 0
        }
    
    def get_trajectory_robustness(self) -> float:
        """获取整个轨迹的 robustness"""
        if not self.trajectory:
            return 0.0
        return self.robustness_calc.compute(self.trajectory[-1], self.trajectory)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'formula': self.safety_formula,
            'trajectory_robustness': self.get_trajectory_robustness()
        }
