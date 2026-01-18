"""
安全环境包装器 v3.0

集成双规范架构：SafetyFilter + TaskRewardShaper
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Optional, Tuple

from ..ltl.propositions import AtomicPropositionManager
from ..safety.action_filter import SafetyFilter
from ..safety.task_reward_shaper import TaskRewardShaper, DEFAULT_REWARD_WEIGHTS


class SafeEnvWrapper(gym.Wrapper):
    """
    基于双规范架构的安全环境包装器
    
    架构：
    1. Safety Specification → SafetyFilter（硬约束，零违规）
    2. Task Specification → TaskRewardShaper（软引导，奖励塑形）
    
    功能：
    - 安全过滤：不安全动作 → 安全替代动作
    - 任务奖励：6 个组件（robustness、progress、acceptance、trap、filter、time）
    - 陷阱终止：进入任务陷阱状态时终止 episode
    """
    
    def __init__(self, 
                 env,
                 safety_formula: str = "G(!wall) & G(!boundary)",
                 task_formula: str = "F(goal)",
                 config: Dict = None):
        """
        Args:
            env: 要包装的 Gymnasium 环境
            safety_formula: 安全规范（LTL 公式）
            task_formula: 任务规范（LTL 公式）
            config: 配置字典
        """
        super().__init__(env)
        
        self.safety_formula = safety_formula
        self.task_formula = task_formula
        self.config = config or {}
        
        # 获取环境信息
        self.env_info = self.unwrapped.get_env_info()
        self.action_map = self.unwrapped.get_action_map()
        
        # 创建原子命题管理器
        self.prop_manager = AtomicPropositionManager()
        self._setup_propositions()
        self.prop_manager.update_env_info(self.env_info)
        
        # 创建 SafetyFilter
        self.safety_filter = SafetyFilter(
            safety_formula=safety_formula,
            prop_manager=self.prop_manager,
            action_map=self.action_map,
            num_actions=self.action_space.n
        )
        
        # 创建 TaskRewardShaper
        reward_weights = self.config.get('reward_weights', DEFAULT_REWARD_WEIGHTS)
        self.task_reward_shaper = TaskRewardShaper(
            task_formula=task_formula,
            prop_manager=self.prop_manager,
            reward_weights=reward_weights
        )
        
        # 统计信息
        self.episode_stats = {
            'safety_interventions': 0,
            'task_completions': 0,
            'task_traps': 0,
            'steps': 0
        }
        
        print(f"[SafeEnvWrapper v3.0] 初始化完成")
        print(f"[SafeEnvWrapper] 安全规范: {safety_formula}")
        print(f"[SafeEnvWrapper] 任务规范: {task_formula}")
    
    def _setup_propositions(self):
        """设置原子命题"""
        # 注册环境自动提供的命题
        self.prop_manager.register_auto_propositions()
        
        # 从配置注册自定义命题
        from ..ltl.propositions import PositionProposition, RegionProposition
        
        ap_config = self.config.get('propositions', {})
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
    
    def _get_current_state(self) -> np.ndarray:
        """获取当前状态（用于命题评估）"""
        return self.unwrapped.agent_pos.copy()
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        
        # 更新环境信息（目标可能已改变）
        self.env_info = self.unwrapped.get_env_info()
        self.prop_manager.update_env_info(self.env_info)
        
        # 重置过滤器和奖励塑形器
        self.safety_filter.reset()
        self.task_reward_shaper.reset()
        
        # 重置统计
        self.episode_stats = {
            'safety_interventions': 0,
            'task_completions': 0,
            'task_traps': 0,
            'steps': 0
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步
        
        流程：
        1. SafetyFilter 过滤动作
        2. 执行安全动作
        3. TaskRewardShaper 计算任务奖励
        4. 处理任务陷阱状态的终止
        5. 返回结果
        """
        current_state = self._get_current_state()
        
        # 1. 安全过滤
        safe_action, filtered, filter_info = self.safety_filter.filter_action(
            current_state, action
        )
        
        if filtered:
            self.episode_stats['safety_interventions'] += 1
        
        # 2. 执行动作
        obs, base_reward, terminated, truncated, info = self.env.step(safe_action)
        
        # 3. 计算任务奖励（传入 filtered 标志）
        task_reward, task_info = self.task_reward_shaper.compute_reward(
            self._get_current_state(),
            filtered=filtered
        )
        
        # 4. 使用任务奖励（不使用环境的 base_reward）
        total_reward = task_reward
        
        # 5. 处理任务陷阱状态的终止
        if task_info.get('terminated', False):
            terminated = True
            self.episode_stats['task_traps'] += 1
        
        # 6. 处理任务完成
        if task_info.get('is_accepting', False):
            self.episode_stats['task_completions'] += 1
        
        # 7. 更新统计
        self.episode_stats['steps'] += 1
        
        # 8. 更新 info
        info.update({
            'safety_filtered': filtered,
            'task_rho': task_info['rho'],
            'task_fsa_state': task_info['fsa_state'],
            'task_is_accepting': task_info['is_accepting'],
            'task_is_trap': task_info['is_trap'],
            'r_rho': task_info['r_rho'],
            'r_progress': task_info['r_progress'],
            'r_accept': task_info['r_accept'],
            'r_trap': task_info.get('r_trap', 0.0),
            'r_filter': task_info['r_filter'],
            'r_time': task_info['r_time'],
            'original_action': action,
            'executed_action': safe_action
        })
        
        return obs, total_reward, terminated, truncated, info
    
    def get_episode_stats(self) -> Dict:
        """获取当前 episode 的统计信息"""
        stats = self.episode_stats.copy()
        stats['intervention_rate'] = (
            stats['safety_interventions'] / stats['steps'] 
            if stats['steps'] > 0 else 0.0
        )
        return stats
    
    def get_safe_actions(self) -> list:
        """获取当前状态下的所有安全动作"""
        current_state = self._get_current_state()
        return self.safety_filter.get_safe_actions(current_state)
