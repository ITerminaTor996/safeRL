# Safe RL Drone

基于形式化方法的安全强化学习项目。采用**双规范架构**，将安全约束与任务目标分离：Safety 用形式化保证（硬约束），Task 交给 RL 学习（软目标）。

## 核心架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    双规范架构 (Dual-Specification)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Safety Layer（形式化保证，硬约束）                           │   │
│  │  ├── Safety Specification: G(!wall) & G(!boundary)          │   │
│  │  ├── SafetyFilter: 基于 FSA 的动作过滤                       │   │
│  │  │   └── 预测下一状态 → 检查是否进入陷阱 → 阻止不安全动作     │   │
│  │  └── 保证：训练和测试全程零违规                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓ 安全动作                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Task Layer（RL 学习，软目标）                                │   │
│  │  ├── Task Specification: F(goal), F(wp1 & F(wp2)), etc.     │   │
│  │  ├── TaskRewardShaper: 基于 FSA + Robustness 的奖励塑形      │   │
│  │  │   ├── 边条件 Robustness（稠密引导）                       │   │
│  │  │   ├── FSA 进度奖励（稀疏里程碑）                          │   │
│  │  │   ├── 任务完成奖励                                        │   │
│  │  │   ├── 任务陷阱惩罚（任务失败，终止 episode）              │   │
│  │  │   └── 过滤器惩罚（学习避免触发安全过滤）                  │   │
│  │  └── 任务陷阱：给惩罚并终止，不是硬约束                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **Safety ≠ Reward** | 安全不进 RL 的 reward，用 SafetyFilter 硬保证 |
| **G 算子 → 硬约束** | `G(!wall)` 这类"永远不"的约束用 SafetyFilter 强制执行 |
| **F/U 算子 → 软引导** | `F(goal)` 这类"最终要"的目标用 Robustness 引导 RL 学习 |
| **两种陷阱** | Safety 陷阱被阻止（永不进入），Task 陷阱给惩罚（任务失败） |

## 边条件 Robustness

参考 [Xiao Li et al., Science Robotics 2019](https://www.science.org/doi/10.1126/scirobotics.aay6276)，使用 FSA 转移边的条件计算 robustness，提供稠密的引导信号。

### 计算方法

对于当前 FSA 状态的每条**有效出边**（非自环、不通向陷阱），计算其条件的 robustness：

```
ρ(a & b) = min(ρ(a), ρ(b))    # AND 取 min
ρ(a | b) = max(ρ(a), ρ(b))    # OR 取 max
ρ(!a)    = -ρ(a)              # NOT 取负
ρ(goal)  = 1 - distance(agent, goal)  # 原子命题：基于距离
```

最终取所有有效边的 max 作为当前状态的 robustness。

### 有效边的定义

1. **非自环**：`source != target`（能推进 FSA 状态）
2. **不通向陷阱**：`target ∉ trap_states`（不会导致任务失败）

这避免了两个问题：
- 自环边（如 `!goal`）会在远离目标时给出正的 robustness
- 通向陷阱的边会错误引导 agent 进入失败状态

## 项目结构

```
safe_rl_drone/
├── env.py                     # GridWorld 环境
├── ltl/                       # LTL 形式化方法模块
│   ├── fsa.py                 # FSA 监控自动机（Spot 库）
│   │   ├── FSAMonitor         # 运行时监控
│   │   ├── _can_reach_accepting()  # BFS 陷阱检测
│   │   └── compute_condition_robustness()  # 边条件 robustness
│   ├── parser.py              # LTL 公式解析
│   ├── propositions.py        # 原子命题管理
│   └── robustness.py          # Robustness 计算器
├── safety/                    # 安全层模块
│   ├── action_filter.py       # SafetyFilter（基于 Safety FSA）
│   └── task_reward_shaper.py  # TaskRewardShaper（基于 Task FSA）
└── wrappers/
    └── safe_env_wrapper.py    # 集成包装器

tests/                         # 测试文件
├── test_safety_filter.py      # SafetyFilter 测试
├── test_task_reward_shaper.py # TaskRewardShaper 测试
└── ...

maps/                          # 地图文件
config.yaml                    # 默认配置
config_sequential.yaml         # 顺序任务配置
config_multi_goal.yaml         # 多目标配置
config_until.yaml              # Until 公式配置
```

## 安装

### Python 依赖

```bash
pip install -r requirements.txt
```

### Spot 库（LTL → FSA 转换）

```bash
# Ubuntu/WSL
sudo apt-get install python3-dev
wget https://www.lrde.epita.fr/dload/spot/spot-2.11.6.tar.gz
tar -xzf spot-2.11.6.tar.gz
cd spot-2.11.6
./configure --prefix=/usr/local
make
sudo make install
```

## 快速开始

```bash
# 设置环境变量（WSL）
export PYTHONPATH=/usr/local/lib/python3.12/site-packages:$PYTHONPATH

# 训练简单任务 F(goal)
python3 main.py

# 训练顺序任务 F(wp1 & F(wp2 & F(goal)))
python3 main.py --config config_sequential.yaml

# 跳过训练，测试已有模型
python3 main.py --skip-train --load models/ppo_safe
```

## 配置说明

```yaml
# 双规范配置
specifications:
  # 安全规范（硬约束）
  safety:
    enabled: true
    formula: "G(!wall) & G(!boundary)"
  
  # 任务规范（软引导）
  task:
    enabled: true
    formula: "F(goal)"  # 或 "F(wp1 & F(wp2 & F(goal)))"

# 奖励权重
reward:
  task_shaping_weights:
    robustness: 0.5     # 边条件 Robustness（稠密）
    progress: 2.0       # FSA 进度（稀疏）
    acceptance: 15.0    # 任务完成
    trap: 0.5           # 任务陷阱惩罚（-5.0）
    filter: 0.5         # 过滤器惩罚（-0.25）
```

## LTL 公式语法

| 算子 | 含义 | 用途 |
|------|------|------|
| `G(φ)` | 全局 (Always) | Safety 规范：`G(!wall)` |
| `F(φ)` | 最终 (Eventually) | Task 规范：`F(goal)` |
| `φ U ψ` | 直到 (Until) | Task 规范：`(!danger) U safe` |
| `!`, `&`, `\|` | 非、与、或 | 组合条件 |

### 任务示例

| 任务 | 公式 |
|------|------|
| 到达目标 | `F(goal)` |
| 多目标选择 | `F(goal1 \| goal2 \| goal3)` |
| 顺序访问 | `F(wp1 & F(wp2 & F(goal)))` |
| 避开危险直到安全 | `(!danger) U safe` |

## 运行测试

```bash
python3 -m pytest tests/ -v
```

## 已知问题

见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

## 参考文献

- Xiao Li et al. "A formal methods approach to interpretable reinforcement learning for robotic planning" Science Robotics, 2019

## License

MIT
