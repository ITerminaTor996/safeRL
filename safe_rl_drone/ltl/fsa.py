"""
FSA（有限状态自动机）模块

将 LTL 公式转换为监控自动机，用于：
1. 运行时监控公式满足状态
2. 检测不可恢复的违规（trap state）
3. 追踪任务进度
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# 尝试导入 spot
try:
    import spot
    SPOT_AVAILABLE = True
except ImportError:
    SPOT_AVAILABLE = False


class FSAState(Enum):
    """FSA 状态类型"""
    INITIAL = 'initial'
    ACCEPTING = 'accepting'
    TRAP = 'trap'
    NORMAL = 'normal'


@dataclass
class FSATransition:
    """FSA 转移"""
    source: int
    target: int
    label: str  # 转移条件（布尔表达式）
    
    def __repr__(self):
        return f"{self.source} --[{self.label}]--> {self.target}"


@dataclass
class FSA:
    """有限状态自动机"""
    num_states: int
    initial_state: int
    accepting_states: Set[int]
    trap_states: Set[int]
    transitions: List[FSATransition]
    state_types: Dict[int, FSAState] = field(default_factory=dict)
    
    def get_transitions_from(self, state: int) -> List[FSATransition]:
        """获取从某状态出发的所有转移"""
        return [t for t in self.transitions if t.source == state]
    
    def is_accepting(self, state: int) -> bool:
        return state in self.accepting_states
    
    def is_trap(self, state: int) -> bool:
        return state in self.trap_states


class FSAMonitor:
    """FSA 监控器"""
    
    def __init__(self, formula: str = None):
        """
        Args:
            formula: LTL 公式字符串
        """
        self.formula = formula
        self.fsa: Optional[FSA] = None
        self.current_state: int = 0
        self._spot_automaton = None
        self._prop_manager = None
        
        if formula:
            self.build_from_formula(formula)
    
    def build_from_formula(self, formula: str):
        """从 LTL 公式构建 FSA"""
        self.formula = formula
        
        if SPOT_AVAILABLE:
            self._build_with_spot(formula)
        else:
            self._build_fallback(formula)
    
    def _build_with_spot(self, formula: str):
        """使用 Spot 构建自动机"""
        try:
            # 解析公式
            f = spot.formula(formula)
            
            # 转换为自动机
            # 使用 monitor 模式，适合运行时监控
            aut = spot.translate(f, 'monitor', 'det')
            self._spot_automaton = aut
            
            # 提取自动机信息
            num_states = aut.num_states()
            initial_state = aut.get_init_state_number()
            
            # 获取接受状态
            accepting_states = set()
            for s in range(num_states):
                if aut.state_is_accepting(s):
                    accepting_states.add(s)
            
            # 提取转移
            transitions = []
            for s in range(num_states):
                for t in aut.out(s):
                    target = t.dst
                    # 获取转移条件
                    label = spot.bdd_format_formula(aut.get_dict(), t.cond)
                    transitions.append(FSATransition(s, target, label))
            
            # 识别 trap 状态（只有自环且不是接受状态）
            trap_states = set()
            for s in range(num_states):
                if s not in accepting_states:
                    out_trans = [t for t in transitions if t.source == s]
                    if len(out_trans) == 1 and out_trans[0].target == s:
                        trap_states.add(s)
            
            self.fsa = FSA(
                num_states=num_states,
                initial_state=initial_state,
                accepting_states=accepting_states,
                trap_states=trap_states,
                transitions=transitions
            )
            
            # 设置状态类型
            for s in range(num_states):
                if s == initial_state:
                    self.fsa.state_types[s] = FSAState.INITIAL
                elif s in accepting_states:
                    self.fsa.state_types[s] = FSAState.ACCEPTING
                elif s in trap_states:
                    self.fsa.state_types[s] = FSAState.TRAP
                else:
                    self.fsa.state_types[s] = FSAState.NORMAL
            
            self.current_state = initial_state
            
            print(f"[FSA] 从公式 '{formula}' 构建自动机")
            print(f"[FSA] 状态数: {num_states}, 接受状态: {accepting_states}")
            
        except Exception as e:
            print(f"[FSA] Spot 构建失败: {e}")
            self._build_fallback(formula)
    
    def _build_fallback(self, formula: str):
        """简单的回退实现"""
        # 对于简单的安全公式 G(!wall)，构建一个两状态自动机
        # 状态 0: 安全（接受）
        # 状态 1: 违规（trap）
        
        self.fsa = FSA(
            num_states=2,
            initial_state=0,
            accepting_states={0},
            trap_states={1},
            transitions=[
                FSATransition(0, 0, "!violation"),  # 保持安全
                FSATransition(0, 1, "violation"),   # 违规
                FSATransition(1, 1, "1"),           # trap 自环
            ]
        )
        
        self.fsa.state_types = {
            0: FSAState.ACCEPTING,
            1: FSAState.TRAP
        }
        
        self.current_state = 0
        print(f"[FSA] 使用简化自动机（公式: {formula}）")
    
    def set_prop_manager(self, prop_manager):
        """设置原子命题管理器"""
        self._prop_manager = prop_manager
    
    def reset(self):
        """重置到初始状态"""
        if self.fsa:
            self.current_state = self.fsa.initial_state
    
    def step(self, state: np.ndarray) -> Tuple[int, bool, bool]:
        """
        根据当前环境状态更新自动机状态
        
        Args:
            state: 环境状态
            
        Returns:
            (new_fsa_state, is_accepting, is_trap)
        """
        if self.fsa is None:
            return 0, True, False
        
        # 获取当前状态的所有转移
        transitions = self.fsa.get_transitions_from(self.current_state)
        
        # 评估每个转移条件
        for trans in transitions:
            if self._evaluate_transition(trans.label, state):
                self.current_state = trans.target
                break
        
        is_accepting = self.fsa.is_accepting(self.current_state)
        is_trap = self.fsa.is_trap(self.current_state)
        
        return self.current_state, is_accepting, is_trap
    
    def _evaluate_transition(self, label: str, state: np.ndarray) -> bool:
        """评估转移条件"""
        if label == '1' or label == 'true':
            return True
        if label == '0' or label == 'false':
            return False
        
        if self._prop_manager is None:
            # 没有命题管理器，使用简单规则
            if 'violation' in label:
                return False  # 默认不违规
            return True
        
        # 解析并评估布尔表达式
        return self._eval_bool_expr(label, state)
    
    def _eval_bool_expr(self, expr: str, state: np.ndarray) -> bool:
        """评估布尔表达式"""
        expr = expr.strip()
        
        # 处理常量
        if expr == '1' or expr.lower() == 'true':
            return True
        if expr == '0' or expr.lower() == 'false':
            return False
        
        # 处理否定
        if expr.startswith('!'):
            inner = expr[1:].strip()
            return not self._eval_bool_expr(inner, state)
        
        # 处理与
        if '&' in expr:
            parts = expr.split('&')
            return all(self._eval_bool_expr(p.strip(), state) for p in parts)
        
        # 处理或
        if '|' in expr:
            parts = expr.split('|')
            return any(self._eval_bool_expr(p.strip(), state) for p in parts)
        
        # 原子命题
        try:
            return self._prop_manager.evaluate(expr, state)
        except:
            return True  # 未知命题默认为真
    
    def is_safe(self) -> bool:
        """检查当前是否安全（不在 trap 状态）"""
        if self.fsa is None:
            return True
        return not self.fsa.is_trap(self.current_state)
    
    def is_accepting(self) -> bool:
        """检查当前是否在接受状态"""
        if self.fsa is None:
            return True
        return self.fsa.is_accepting(self.current_state)
    
    def get_state_info(self) -> Dict:
        """获取当前状态信息"""
        if self.fsa is None:
            return {'state': 0, 'type': 'unknown'}
        
        return {
            'state': self.current_state,
            'type': self.fsa.state_types.get(self.current_state, FSAState.NORMAL).value,
            'is_accepting': self.is_accepting(),
            'is_trap': not self.is_safe()
        }
    
    def check_action_safety(self, 
                            current_state: np.ndarray,
                            next_state: np.ndarray) -> bool:
        """
        检查动作是否安全（不会导致进入 trap 状态）
        
        Args:
            current_state: 当前环境状态
            next_state: 执行动作后的环境状态
            
        Returns:
            True 如果动作安全
        """
        # 保存当前自动机状态
        saved_fsa_state = self.current_state
        
        # 模拟执行
        self.step(next_state)
        is_safe = self.is_safe()
        
        # 恢复状态
        self.current_state = saved_fsa_state
        
        return is_safe


# ============================================================
# 测试代码
# ============================================================

def test_fsa():
    """测试 FSA 模块"""
    print("=" * 60)
    print("测试 FSA 模块")
    print("=" * 60)
    print(f"Spot 库可用: {SPOT_AVAILABLE}")
    
    # 测试公式
    test_formulas = [
        "G(!wall)",
        "F(goal)",
        "G(!wall) & F(goal)",
    ]
    
    for formula in test_formulas:
        print(f"\n公式: {formula}")
        monitor = FSAMonitor(formula)
        
        if monitor.fsa:
            print(f"  状态数: {monitor.fsa.num_states}")
            print(f"  初始状态: {monitor.fsa.initial_state}")
            print(f"  接受状态: {monitor.fsa.accepting_states}")
            print(f"  Trap 状态: {monitor.fsa.trap_states}")
            print(f"  转移:")
            for t in monitor.fsa.transitions[:5]:  # 只显示前5个
                print(f"    {t}")
            if len(monitor.fsa.transitions) > 5:
                print(f"    ... (共 {len(monitor.fsa.transitions)} 个转移)")
    
    # 测试运行时监控
    print("\n" + "-" * 40)
    print("测试运行时监控")
    
    from .propositions import AtomicPropositionManager
    
    prop_manager = AtomicPropositionManager()
    prop_manager.register_auto_propositions()
    
    env_info = {
        'wall_positions': {(1, 1)},
        'goal_position': (3, 3),
        'grid_size': (5, 5)
    }
    prop_manager.update_env_info(env_info)
    
    monitor = FSAMonitor("G(!wall)")
    monitor.set_prop_manager(prop_manager)
    
    # 模拟轨迹
    trajectory = [
        np.array([0, 0]),  # 安全
        np.array([0, 1]),  # 安全
        np.array([1, 1]),  # 墙！
        np.array([2, 2]),  # 安全
    ]
    
    print(f"\n监控公式: G(!wall)")
    for i, state in enumerate(trajectory):
        fsa_state, is_acc, is_trap = monitor.step(state)
        print(f"  t={i}, pos={tuple(state)}, FSA状态={fsa_state}, "
              f"接受={is_acc}, trap={is_trap}")
    
    print("\n" + "=" * 60)
    print("FSA 模块测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    test_fsa()
