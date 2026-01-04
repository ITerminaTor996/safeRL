"""
原子命题管理模块

负责：
1. 管理所有原子命题的定义
2. 根据环境状态评估命题真值
3. 计算命题的 robustness 值（基于距离）
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class AtomicProposition:
    """原子命题基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, state: np.ndarray, env_info: Dict) -> bool:
        """评估命题在当前状态下的真值"""
        raise NotImplementedError
    
    def robustness(self, state: np.ndarray, env_info: Dict) -> float:
        """计算命题的 robustness 值"""
        raise NotImplementedError


class AutoProposition(AtomicProposition):
    """环境自动提供的命题（wall, goal, boundary）"""
    
    def __init__(self, name: str, prop_type: str):
        super().__init__(name)
        self.prop_type = prop_type  # 'wall', 'goal', 'boundary', 'empty'
    
    def evaluate(self, state: np.ndarray, env_info: Dict) -> bool:
        """评估命题真值"""
        agent_pos = tuple(state)
        
        if self.prop_type == 'wall':
            # 当前位置是否是墙（或下一步会撞墙）
            return agent_pos in env_info.get('wall_positions', set())
        
        elif self.prop_type == 'goal':
            # 当前位置是否是目标
            goal_pos = env_info.get('goal_position')
            return agent_pos == goal_pos
        
        elif self.prop_type == 'boundary':
            # 是否越界
            rows, cols = env_info.get('grid_size', (0, 0))
            r, c = agent_pos
            return r < 0 or r >= rows or c < 0 or c >= cols
        
        elif self.prop_type == 'empty':
            # 当前位置是否是空地
            return (agent_pos not in env_info.get('wall_positions', set()) and
                    agent_pos != env_info.get('goal_position'))
        
        return False
    
    def robustness(self, state: np.ndarray, env_info: Dict) -> float:
        """
        计算 robustness 值
        
        语义：ρ > 0 表示命题为真，ρ < 0 表示命题为假
        
        对于 wall：
            ρ > 0 表示在墙上（wall = true）
            ρ < 0 表示不在墙上（wall = false）
            
        对于 goal：
            ρ > 0 表示在目标上（goal = true）
            ρ < 0 表示不在目标上（goal = false）
        """
        agent_pos = tuple(state)  # 转为 tuple 便于比较
        
        if self.prop_type == 'wall':
            wall_positions = env_info.get('wall_positions', set())
            if not wall_positions:
                return -10.0  # 没有墙，wall = false
            
            # 直接检查是否在墙上
            if agent_pos in wall_positions:
                return 1.0  # 在墙上
            
            # 计算到最近墙的曼哈顿距离
            min_dist = min(
                abs(state[0] - w[0]) + abs(state[1] - w[1])
                for w in wall_positions
            )
            return -min_dist
        
        elif self.prop_type == 'goal':
            goal_pos = env_info.get('goal_position')
            if goal_pos is None:
                return -10.0
            
            # 直接检查是否在目标上
            if agent_pos == goal_pos:
                return 1.0
            
            # 计算到目标的曼哈顿距离
            distance = abs(state[0] - goal_pos[0]) + abs(state[1] - goal_pos[1])
            return -distance
        
        elif self.prop_type == 'boundary':
            rows, cols = env_info.get('grid_size', (10, 10))
            r, c = state
            
            # 检查是否越界
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return 1.0  # 越界了，boundary = true
            
            # 到边界的最小距离
            dist_to_boundary = min(r, c, rows - 1 - r, cols - 1 - c)
            return -dist_to_boundary - 0.5
        
        elif self.prop_type == 'empty':
            wall_positions = env_info.get('wall_positions', set())
            if not wall_positions:
                return 10.0  # 没有墙，empty = true
            
            if agent_pos in wall_positions:
                return -1.0  # 在墙上，empty = false
            
            min_dist = min(
                abs(state[0] - w[0]) + abs(state[1] - w[1])
                for w in wall_positions
            )
            return min_dist - 1.0
        
        return 0.0


class PositionProposition(AtomicProposition):
    """位置命题：agent 是否在指定位置"""
    
    def __init__(self, name: str, position: Tuple[int, int]):
        super().__init__(name)
        self.position = tuple(position)
    
    def evaluate(self, state: np.ndarray, env_info: Dict) -> bool:
        return tuple(state) == self.position
    
    def robustness(self, state: np.ndarray, env_info: Dict) -> float:
        agent_pos = np.array(state)
        target_pos = np.array(self.position)
        distance = np.abs(agent_pos - target_pos).sum()
        # 离散环境：只有距离=0才算到达
        if distance == 0:
            return 1.0
        else:
            return -distance


class RegionProposition(AtomicProposition):
    """区域命题：agent 是否在指定区域内"""
    
    def __init__(self, name: str, positions: List[Tuple[int, int]], 
                 avoid: bool = False):
        """
        Args:
            name: 命题名称
            positions: 区域包含的位置列表
            avoid: True 表示要避开这个区域（如 danger_zone）
        """
        super().__init__(name)
        self.positions = set(tuple(p) for p in positions)
        self.avoid = avoid
    
    def evaluate(self, state: np.ndarray, env_info: Dict) -> bool:
        return tuple(state) in self.positions
    
    def robustness(self, state: np.ndarray, env_info: Dict) -> float:
        """
        计算 robustness
        
        语义：ρ > 0 表示命题为真（agent 在区域内）
              ρ < 0 表示命题为假（agent 不在区域内）
        
        注意：avoid 参数不影响 robustness 计算，只是一个语义标记
              用户如果想避开区域，应该在 LTL 公式中写 G(!danger_zone)
        """
        agent_pos = np.array(state)
        
        if not self.positions:
            return -10.0  # 空区域，永远不在里面
        
        # 计算到区域的最小距离
        min_dist = min(
            np.abs(agent_pos - np.array(p)).sum() 
            for p in self.positions
        )
        
        # 离散环境：只有距离=0才算在区域内
        if min_dist == 0:
            return 1.0  # 在区域内
        else:
            return -min_dist  # 不在区域内


class AtomicPropositionManager:
    """原子命题管理器"""
    
    def __init__(self):
        self.propositions: Dict[str, AtomicProposition] = {}
        self._env_info: Dict = {}
    
    def register_auto_propositions(self):
        """注册环境自动提供的命题"""
        auto_props = ['wall', 'goal', 'boundary', 'empty']
        for prop_type in auto_props:
            self.propositions[prop_type] = AutoProposition(prop_type, prop_type)
    
    def register_proposition(self, name: str, prop: AtomicProposition):
        """注册自定义命题"""
        self.propositions[name] = prop
    
    def register_from_config(self, config: Dict):
        """从配置文件注册命题"""
        ap_config = config.get('atomic_propositions', {})
        
        for name, definition in ap_config.items():
            if definition == 'auto':
                # 自动命题
                self.propositions[name] = AutoProposition(name, name)
            
            elif isinstance(definition, dict):
                prop_type = definition.get('type')
                
                if prop_type == 'position':
                    pos = tuple(definition['pos'])
                    self.propositions[name] = PositionProposition(name, pos)
                
                elif prop_type == 'region':
                    positions = definition['positions']
                    avoid = definition.get('avoid', False)
                    self.propositions[name] = RegionProposition(name, positions, avoid)
    
    def update_env_info(self, env_info: Dict):
        """更新环境信息"""
        self._env_info = env_info
    
    def evaluate(self, name: str, state: np.ndarray) -> bool:
        """评估命题真值"""
        if name not in self.propositions:
            raise ValueError(f"Unknown proposition: {name}")
        return self.propositions[name].evaluate(state, self._env_info)
    
    def robustness(self, name: str, state: np.ndarray) -> float:
        """计算命题 robustness"""
        if name not in self.propositions:
            raise ValueError(f"Unknown proposition: {name}")
        return self.propositions[name].robustness(state, self._env_info)
    
    def get_all_names(self) -> List[str]:
        """获取所有命题名称"""
        return list(self.propositions.keys())
    
    def evaluate_all(self, state: np.ndarray) -> Dict[str, bool]:
        """评估所有命题"""
        return {name: self.evaluate(name, state) for name in self.propositions}


# ============================================================
# 测试代码
# ============================================================

def test_propositions():
    """测试原子命题模块"""
    print("=" * 60)
    print("测试原子命题模块")
    print("=" * 60)
    
    # 创建管理器
    manager = AtomicPropositionManager()
    manager.register_auto_propositions()
    
    # 添加自定义命题
    manager.register_proposition(
        'checkpoint1', 
        PositionProposition('checkpoint1', (2, 3))
    )
    manager.register_proposition(
        'danger_zone',
        RegionProposition('danger_zone', [(1, 1), (1, 2), (2, 1)], avoid=True)
    )
    
    # 设置环境信息
    env_info = {
        'wall_positions': {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
        'goal_position': (4, 4),
        'grid_size': (6, 6)
    }
    manager.update_env_info(env_info)
    
    # 测试不同位置
    test_positions = [
        np.array([3, 3]),  # 空地
        np.array([4, 4]),  # 目标
        np.array([1, 1]),  # 危险区
        np.array([2, 3]),  # checkpoint1
    ]
    
    print("\n命题列表:", manager.get_all_names())
    
    for pos in test_positions:
        print(f"\n位置 {pos}:")
        print(f"  wall: {manager.evaluate('wall', pos)}, ρ = {manager.robustness('wall', pos):.2f}")
        print(f"  goal: {manager.evaluate('goal', pos)}, ρ = {manager.robustness('goal', pos):.2f}")
        print(f"  boundary: {manager.evaluate('boundary', pos)}, ρ = {manager.robustness('boundary', pos):.2f}")
        print(f"  checkpoint1: {manager.evaluate('checkpoint1', pos)}, ρ = {manager.robustness('checkpoint1', pos):.2f}")
        print(f"  danger_zone: {manager.evaluate('danger_zone', pos)}, ρ = {manager.robustness('danger_zone', pos):.2f}")
    
    print("\n" + "=" * 60)
    print("原子命题模块测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    test_propositions()
