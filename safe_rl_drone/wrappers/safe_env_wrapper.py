"""
安全环境包装器 v2.0

集成 LTL 安全监控、Robustness 计算、动作过滤
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Optional, Tuple, Any

from ..safety.monitor import SafetyMonitor
from ..safety.action_filter import ActionFilter


class SafeEnvWrapper(gym.Wrapper):
    """
    基于 LTL 形式化方法的安全环境包装器
    
    功能：
    1. 使用 LTL 公式定义安全约束
    2. 运行时监控并过滤不安全动作
    3. 计算 robustness 用于奖励塑形
    4. 记录安全统计信息
    """
    
    def __init__(self, 
                 env,
                 safety_formula: str = "G(!wall)",
                 config: Dict = None,
                 unsafe_penalty: float = -1.0,
                 use_robustness_reward: bool = True,
                 robustness_weight: float = 0.1):
        """
        Args:
            env: 要包装的 Gymnasium 环境
            safety_formula: LTL 安全公式
            config: 配置字典（包含原子命题定义等）
            unsafe_penalty: 尝试不安全动作的惩罚
            use_robustness_reward: 是否使用 robustness 作为奖励塑形
            robustness_weight: robustness 奖励的权重
        """
        super().__init__(env)
        
        self.safety_formula = safety_formula
        self.config = config or {}
        self.unsafe_penalty = unsafe_penalty
        self.use_robustness_reward = use_robustness_reward
        self.robustness_weight = robustness_weight
        
        # 获取环境信息
        self.env_info = self.unwrapped.get_env_info()
        self.action_map = self.unwrapped.get_action_map()
        
        # 创建安全监控器
        self.safety_monitor = SafetyMonitor(
            safety_formula=safety_formula,
            env_info=self.env_info,
            config=config
        )
        
        # 创建动作过滤器
        self.action_filter = ActionFilter(
            safety_monitor=self.safety_monitor,
            action_map=self.action_map,
            num_actions=self.action_space.n
        )
        
        # 统计信息
        self.episode_stats = {
            'interventions': 0,
            'collisions': 0,
            'total_robustness': 0.0,
            'steps': 0
        }
        
        print(f"[SafeEnvWrapper] 初始化完成")
        print(f"[SafeEnvWrapper] 安全公式: {safety_formula}")
        print(f"[SafeEnvWrapper] Robustness 奖励: {use_robustness_reward}")
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        
        # 重置监控器和过滤器
        self.safety_monitor.reset()
        self.action_filter.reset()
        
        # 重置统计
        self.episode_stats = {
            'interventions': 0,
            'collisions': 0,
            'total_robustness': 0.0,
            'steps': 0
        }
        
        # 记录初始状态（使用 agent_pos，不是 obs）
        agent_pos = self.unwrapped.agent_pos.copy()
        self.safety_monitor.step(agent_pos)
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步，包含安全检查和奖励塑形
        """
        current_state = self.unwrapped.agent_pos.copy()
        
        # 1. 过滤动作
        filtered_action, was_modified, filter_info = self.action_filter.filter_action(
            current_state, action
        )
        
        # 2. 执行动作
        obs, reward, terminated, truncated, info = self.env.step(filtered_action)
        
        # 3. 更新安全监控（使用 agent_pos，不是 obs）
        agent_pos = self.unwrapped.agent_pos.copy()
        monitor_info = self.safety_monitor.step(agent_pos)
        
        # 4. 计算奖励
        total_reward = reward
        
        # 4.1 如果动作被修改，施加惩罚
        if was_modified:
            total_reward += self.unsafe_penalty
            self.episode_stats['interventions'] += 1
            info['safety_intervention'] = True
            info['intervention_reason'] = filter_info.get('reason', '')
        
        # 4.2 Robustness 奖励塑形
        if self.use_robustness_reward:
            delta_rho = monitor_info.get('delta_robustness', 0.0)
            robustness_reward = self.robustness_weight * delta_rho
            total_reward += robustness_reward
            info['robustness_reward'] = robustness_reward
        
        # 5. 更新统计
        self.episode_stats['steps'] += 1
        self.episode_stats['total_robustness'] += monitor_info.get('robustness', 0.0)
        
        if info.get('collision', False):
            self.episode_stats['collisions'] += 1
        
        # 6. 添加额外信息
        info['robustness'] = monitor_info.get('robustness', 0.0)
        info['original_action'] = action
        info['executed_action'] = filtered_action
        info['episode_interventions'] = self.episode_stats['interventions']
        
        return obs, total_reward, terminated, truncated, info
    
    def get_episode_stats(self) -> Dict:
        """获取当前 episode 的统计信息"""
        stats = self.episode_stats.copy()
        stats['intervention_rate'] = (
            stats['interventions'] / stats['steps'] 
            if stats['steps'] > 0 else 0.0
        )
        stats['avg_robustness'] = (
            stats['total_robustness'] / stats['steps']
            if stats['steps'] > 0 else 0.0
        )
        stats['trajectory_robustness'] = self.safety_monitor.get_trajectory_robustness()
        return stats
    
    def get_safe_actions(self) -> list:
        """获取当前状态下的所有安全动作"""
        current_state = self.unwrapped.agent_pos
        return self.action_filter.get_safe_actions(current_state)
