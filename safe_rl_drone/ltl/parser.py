"""
LTL 公式解析模块

使用 Spot 库解析 LTL 公式，提取原子命题，生成语法树
"""

import re
from typing import List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# 尝试导入 spot
try:
    import spot
    SPOT_AVAILABLE = True
except ImportError:
    SPOT_AVAILABLE = False
    print("[Warning] Spot library not available. Using fallback parser.")


class LTLOperator(Enum):
    """LTL 算子枚举"""
    # 时序算子
    GLOBALLY = 'G'      # □ Always
    EVENTUALLY = 'F'    # ◇ Eventually  
    NEXT = 'X'          # ○ Next
    UNTIL = 'U'         # Until
    
    # 布尔算子
    AND = '&'
    OR = '|'
    NOT = '!'
    IMPLIES = '->'
    
    # 原子命题
    ATOM = 'ATOM'


@dataclass
class LTLNode:
    """LTL 语法树节点"""
    operator: LTLOperator
    value: Optional[str] = None  # 原子命题名称（仅 ATOM 类型）
    children: List['LTLNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def __repr__(self):
        if self.operator == LTLOperator.ATOM:
            return f"ATOM({self.value})"
        elif len(self.children) == 1:
            return f"{self.operator.value}({self.children[0]})"
        elif len(self.children) == 2:
            return f"({self.children[0]} {self.operator.value} {self.children[1]})"
        return f"{self.operator.value}"


class LTLParser:
    """LTL 公式解析器"""
    
    def __init__(self):
        self.formula_str: str = ""
        self.syntax_tree: Optional[LTLNode] = None
        self.atomic_propositions: Set[str] = set()
        self._spot_formula = None
    
    def parse(self, formula: str) -> 'LTLParser':
        """
        解析 LTL 公式
        
        支持的语法：
        - G(φ), F(φ), X(φ): 时序算子
        - φ & ψ, φ | ψ: 布尔与/或
        - !φ: 否定
        - φ -> ψ: 蕴含
        - φ U ψ: Until
        - 原子命题: wall, goal, checkpoint1 等
        
        Args:
            formula: LTL 公式字符串
            
        Returns:
            self（支持链式调用）
        """
        self.formula_str = formula
        self.atomic_propositions = self._extract_atoms(formula)
        
        if SPOT_AVAILABLE:
            self._parse_with_spot(formula)
        else:
            self._parse_fallback(formula)
        
        return self
    
    def _extract_atoms(self, formula: str) -> Set[str]:
        """提取公式中的原子命题"""
        # 移除算子和括号，提取标识符
        # 算子: G, F, X, U, &, |, !, ->
        cleaned = formula
        
        # 替换算子为空格
        operators = ['G', 'F', 'X', 'U', '->', '&', '|', '!', '(', ')']
        for op in operators:
            cleaned = cleaned.replace(op, ' ')
        
        # 提取标识符（字母开头，可包含字母数字下划线）
        atoms = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cleaned))
        
        # 过滤掉可能的关键字
        keywords = {'true', 'false', 'True', 'False'}
        atoms = atoms - keywords
        
        return atoms
    
    def _parse_with_spot(self, formula: str):
        """使用 Spot 库解析"""
        try:
            # Spot 使用的语法可能略有不同，做一些转换
            spot_formula = formula
            # F 在 spot 中也可以用 F 或 <>
            # G 在 spot 中也可以用 G 或 []
            
            self._spot_formula = spot.formula(spot_formula)
            self.syntax_tree = self._spot_to_tree(self._spot_formula)
            
        except Exception as e:
            print(f"[Warning] Spot parsing failed: {e}")
            print("[Warning] Falling back to simple parser")
            self._parse_fallback(formula)
    
    def _spot_to_tree(self, f) -> LTLNode:
        """将 Spot 公式转换为我们的语法树"""
        kind = f.kind()
        
        # 原子命题
        if f.is_literal():
            if f.is_tt():
                return LTLNode(LTLOperator.ATOM, value='true')
            elif f.is_ff():
                return LTLNode(LTLOperator.ATOM, value='false')
            
            # 处理否定的原子命题
            if kind == spot.op_Not:
                child = self._spot_to_tree(f[0])
                return LTLNode(LTLOperator.NOT, children=[child])
            
            # 普通原子命题
            ap_name = str(f)
            # 移除可能的引号
            ap_name = ap_name.strip('"').strip("'")
            return LTLNode(LTLOperator.ATOM, value=ap_name)
        
        # 一元算子
        if kind == spot.op_Not:
            child = self._spot_to_tree(f[0])
            return LTLNode(LTLOperator.NOT, children=[child])
        
        elif kind == spot.op_X:
            child = self._spot_to_tree(f[0])
            return LTLNode(LTLOperator.NEXT, children=[child])
        
        elif kind == spot.op_F:
            child = self._spot_to_tree(f[0])
            return LTLNode(LTLOperator.EVENTUALLY, children=[child])
        
        elif kind == spot.op_G:
            child = self._spot_to_tree(f[0])
            return LTLNode(LTLOperator.GLOBALLY, children=[child])
        
        # 二元算子
        elif kind == spot.op_And:
            children = [self._spot_to_tree(f[i]) for i in range(f.size())]
            # 多个子节点时，构建二叉树
            result = children[0]
            for child in children[1:]:
                result = LTLNode(LTLOperator.AND, children=[result, child])
            return result
        
        elif kind == spot.op_Or:
            children = [self._spot_to_tree(f[i]) for i in range(f.size())]
            result = children[0]
            for child in children[1:]:
                result = LTLNode(LTLOperator.OR, children=[result, child])
            return result
        
        elif kind == spot.op_U:
            left = self._spot_to_tree(f[0])
            right = self._spot_to_tree(f[1])
            return LTLNode(LTLOperator.UNTIL, children=[left, right])
        
        elif kind == spot.op_Implies:
            left = self._spot_to_tree(f[0])
            right = self._spot_to_tree(f[1])
            return LTLNode(LTLOperator.IMPLIES, children=[left, right])
        
        # 其他情况，尝试作为原子命题处理
        return LTLNode(LTLOperator.ATOM, value=str(f))
    
    def _parse_fallback(self, formula: str):
        """简单的回退解析器（不依赖 Spot）"""
        # 这是一个简化的解析器，只处理基本情况
        self.syntax_tree = self._parse_expr(formula.strip())
    
    def _parse_expr(self, expr: str) -> LTLNode:
        """递归解析表达式"""
        expr = expr.strip()
        
        # 移除最外层括号
        while expr.startswith('(') and expr.endswith(')'):
            # 检查括号是否匹配
            depth = 0
            matched = True
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    matched = False
                    break
            if matched:
                expr = expr[1:-1].strip()
            else:
                break
        
        # 检查蕴含 ->
        idx = self._find_operator(expr, '->')
        if idx != -1:
            left = self._parse_expr(expr[:idx])
            right = self._parse_expr(expr[idx+2:])
            return LTLNode(LTLOperator.IMPLIES, children=[left, right])
        
        # 检查 |
        idx = self._find_operator(expr, '|')
        if idx != -1:
            left = self._parse_expr(expr[:idx])
            right = self._parse_expr(expr[idx+1:])
            return LTLNode(LTLOperator.OR, children=[left, right])
        
        # 检查 &
        idx = self._find_operator(expr, '&')
        if idx != -1:
            left = self._parse_expr(expr[:idx])
            right = self._parse_expr(expr[idx+1:])
            return LTLNode(LTLOperator.AND, children=[left, right])
        
        # 检查 U
        idx = self._find_operator(expr, 'U')
        if idx != -1:
            left = self._parse_expr(expr[:idx])
            right = self._parse_expr(expr[idx+1:])
            return LTLNode(LTLOperator.UNTIL, children=[left, right])
        
        # 检查一元算子
        if expr.startswith('G(') or expr.startswith('G '):
            inner = self._extract_unary_arg(expr, 'G')
            child = self._parse_expr(inner)
            return LTLNode(LTLOperator.GLOBALLY, children=[child])
        
        if expr.startswith('F(') or expr.startswith('F '):
            inner = self._extract_unary_arg(expr, 'F')
            child = self._parse_expr(inner)
            return LTLNode(LTLOperator.EVENTUALLY, children=[child])
        
        if expr.startswith('X(') or expr.startswith('X '):
            inner = self._extract_unary_arg(expr, 'X')
            child = self._parse_expr(inner)
            return LTLNode(LTLOperator.NEXT, children=[child])
        
        if expr.startswith('!'):
            inner = expr[1:].strip()
            if inner.startswith('('):
                inner = self._extract_unary_arg('!' + inner, '!')
            child = self._parse_expr(inner)
            return LTLNode(LTLOperator.NOT, children=[child])
        
        # 原子命题
        return LTLNode(LTLOperator.ATOM, value=expr.strip())
    
    def _find_operator(self, expr: str, op: str) -> int:
        """在表达式中找到算子位置（考虑括号嵌套）"""
        depth = 0
        i = 0
        while i < len(expr):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
            elif depth == 0:
                if expr[i:i+len(op)] == op:
                    # 确保不是标识符的一部分
                    before_ok = i == 0 or not expr[i-1].isalnum()
                    after_ok = i + len(op) >= len(expr) or not expr[i+len(op)].isalnum()
                    if before_ok and after_ok:
                        return i
            i += 1
        return -1
    
    def _extract_unary_arg(self, expr: str, op: str) -> str:
        """提取一元算子的参数"""
        expr = expr[len(op):].strip()
        if expr.startswith('('):
            # 找到匹配的右括号
            depth = 0
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        return expr[1:i]
            return expr[1:-1]
        return expr
    
    def get_spot_formula(self):
        """获取 Spot 公式对象"""
        return self._spot_formula
    
    def get_atomic_propositions(self) -> Set[str]:
        """获取所有原子命题"""
        return self.atomic_propositions
    
    def to_string(self) -> str:
        """返回规范化的公式字符串"""
        if self._spot_formula is not None:
            return str(self._spot_formula)
        return self.formula_str


# ============================================================
# 测试代码
# ============================================================

def test_parser():
    """测试 LTL 解析器"""
    print("=" * 60)
    print("测试 LTL 解析器")
    print("=" * 60)
    print(f"Spot 库可用: {SPOT_AVAILABLE}")
    
    test_formulas = [
        "G(!wall)",
        "F(goal)",
        "G(!wall) & F(goal)",
        "F(checkpoint1) & F(goal)",
        "G(!wall) & G(!boundary) & F(goal)",
        "(!goal) U checkpoint",
        "F(a & F(b))",
        "G(danger -> X(!danger))",
    ]
    
    parser = LTLParser()
    
    for formula in test_formulas:
        print(f"\n公式: {formula}")
        try:
            parser.parse(formula)
            print(f"  原子命题: {parser.get_atomic_propositions()}")
            print(f"  语法树: {parser.syntax_tree}")
            if parser._spot_formula:
                print(f"  Spot 解析: {parser._spot_formula}")
        except Exception as e:
            print(f"  解析错误: {e}")
    
    print("\n" + "=" * 60)
    print("LTL 解析器测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    test_parser()
