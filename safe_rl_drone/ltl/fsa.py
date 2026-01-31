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
    
    def __init__(self, formula: str = None, mode: str = 'monitor'):
        """
        Args:
            formula: LTL 公式字符串
            mode: 自动机模式
                - 'monitor': 用于安全监控（检测违规）
                - 'ba': 用于任务跟踪（检测完成）
        """
        self.formula = formula
        self.mode = mode
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
            # - monitor: 适合安全监控（检测违规）
            # - ba: 适合任务跟踪（检测完成）
            # - det: 确定性自动机
            # - complete: 所有输入都有转移（包括违规转移）
            aut = spot.translate(f, self.mode, 'det', 'complete')
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
            
            # 识别 trap 状态
            # 定义：无法到达接受状态的非接受状态 = 陷阱状态
            # 这包括：
            # 1. 严格陷阱（所有出边指向自己）
            # 2. 通向陷阱的不可恢复状态
            trap_states = set()
            for s in range(num_states):
                if s not in accepting_states:
                    # 使用 BFS 检查是否能到达接受状态
                    if not self._can_reach_accepting(s, accepting_states, transitions, num_states):
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
            
            print(f"[FSA] 从公式 '{formula}' 构建自动机 (模式: {self.mode})")
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
    
    def _can_reach_accepting(self, 
                            start_state: int,
                            accepting_states: Set[int],
                            transitions: List[FSATransition],
                            num_states: int) -> bool:
        """
        检查从 start_state 是否能到达任何接受状态
        
        使用 BFS 遍历所有可达状态
        
        Args:
            start_state: 起始状态
            accepting_states: 接受状态集合
            transitions: 所有转移
            num_states: 状态总数
            
        Returns:
            True 如果能到达接受状态
        """
        if start_state in accepting_states:
            return True
        
        visited = set([start_state])
        queue = [start_state]
        
        while queue:
            current = queue.pop(0)
            
            # 获取所有出边
            for trans in transitions:
                if trans.source == current:
                    target = trans.target
                    
                    if target in accepting_states:
                        return True
                    
                    if target not in visited:
                        visited.add(target)
                        queue.append(target)
        
        return False
    
    def set_prop_manager(self, prop_manager):
        """设置原子命题管理器"""
        self._prop_manager = prop_manager
    
    def reset(self):
        """重置到初始状态"""
        if self.fsa:
            self.current_state = self.fsa.initial_state
    
    def step(self, state: np.ndarray) -> Tuple[int, bool, bool]:
        """
        根据当前环境状态和自动机状态更新自动机状态
        
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
        """
        评估布尔表达式
        
        支持的运算符（按优先级从低到高）：
        1. | (或)
        2. & (与)
        3. ! (非)
        4. () (括号)
        
        正确处理运算符优先级和括号
        """
        expr = expr.strip()
        
        # 处理常量
        if expr == '1' or expr.lower() == 'true':
            return True
        if expr == '0' or expr.lower() == 'false':
            return False
        
        # 去除最外层括号（如果有）
        if expr.startswith('(') and expr.endswith(')'):
            # 检查是否是匹配的最外层括号
            if self._is_outer_parens(expr):
                return self._eval_bool_expr(expr[1:-1], state)
        
        # 处理或（最低优先级）
        # 找到不在括号内的 |
        or_pos = self._find_operator_outside_parens(expr, '|')
        if or_pos != -1:
            left = expr[:or_pos].strip()
            right = expr[or_pos+1:].strip()
            return self._eval_bool_expr(left, state) or self._eval_bool_expr(right, state)
        
        # 处理与（中等优先级）
        # 找到不在括号内的 &
        and_pos = self._find_operator_outside_parens(expr, '&')
        if and_pos != -1:
            left = expr[:and_pos].strip()
            right = expr[and_pos+1:].strip()
            return self._eval_bool_expr(left, state) and self._eval_bool_expr(right, state)
        
        # 处理否定（最高优先级）
        if expr.startswith('!'):
            inner = expr[1:].strip()
            return not self._eval_bool_expr(inner, state)
        
        # 原子命题
        try:
            return self._prop_manager.evaluate(expr, state)
        except:
            return True  # 未知命题默认为真
    
    def compute_condition_robustness(self, expr: str, state: np.ndarray) -> float:
        """
        计算布尔表达式的 robustness
        
        参考：Xiao Li et al., Science Robotics 2019
        
        语义：
        - AND (a & b): min(ρ(a), ρ(b))
        - OR (a | b): max(ρ(a), ρ(b))
        - NOT (!a): -ρ(a)
        - 原子命题: prop_manager.robustness(name, state)
        
        Args:
            expr: 布尔表达式字符串
            state: 环境状态
            
        Returns:
            robustness 值
        """
        expr = expr.strip()
        
        # 处理常量
        if expr == '1' or expr.lower() == 'true':
            return 100.0  # 大正数
        if expr == '0' or expr.lower() == 'false':
            return -100.0  # 大负数
        
        # 去除最外层括号（如果有）
        if expr.startswith('(') and expr.endswith(')'):
            if self._is_outer_parens(expr):
                return self.compute_condition_robustness(expr[1:-1], state)
        
        # 处理或（最低优先级）：取 max
        or_pos = self._find_operator_outside_parens(expr, '|')
        if or_pos != -1:
            left = expr[:or_pos].strip()
            right = expr[or_pos+1:].strip()
            return max(
                self.compute_condition_robustness(left, state),
                self.compute_condition_robustness(right, state)
            )
        
        # 处理与（中等优先级）：取 min
        and_pos = self._find_operator_outside_parens(expr, '&')
        if and_pos != -1:
            left = expr[:and_pos].strip()
            right = expr[and_pos+1:].strip()
            return min(
                self.compute_condition_robustness(left, state),
                self.compute_condition_robustness(right, state)
            )
        
        # 处理否定（最高优先级）：取负
        if expr.startswith('!'):
            inner = expr[1:].strip()
            return -self.compute_condition_robustness(inner, state)
        
        # 原子命题
        try:
            return self._prop_manager.robustness(expr, state)
        except:
            return 0.0  # 未知命题默认为 0
    
    def _is_outer_parens(self, expr: str) -> bool:
        """
        检查表达式最外层的括号是否匹配
        
        例如：
        - "(a & b)" -> True
        - "(a) & (b)" -> False（括号不是最外层）
        """
        if not (expr.startswith('(') and expr.endswith(')')):
            return False
        
        depth = 0
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            
            # 如果在中间某处深度降到 0，说明不是最外层括号
            if depth == 0 and i < len(expr) - 1:
                return False
        
        return True
    
    def _find_operator_outside_parens(self, expr: str, op: str) -> int:
        """
        找到不在括号内的运算符位置
        
        从右往左找，这样可以正确处理左结合
        
        Args:
            expr: 表达式
            op: 运算符（'&' 或 '|'）
            
        Returns:
            运算符位置，如果没找到返回 -1
        """
        depth = 0
        # 从右往左找（处理左结合）
        for i in range(len(expr) - 1, -1, -1):
            ch = expr[i]
            if ch == ')':
                depth += 1
            elif ch == '(':
                depth -= 1
            elif ch == op and depth == 0:
                return i
        
        return -1
    
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
