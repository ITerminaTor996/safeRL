# Safe RL Drone

基于形式化方法的安全强化学习项目。使用 LTL (Linear Temporal Logic) 提供安全保证，结合 Goal-Conditioned RL 实现目标泛化。

## 核心设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Architecture                              │
├─────────────────────────────────────────────────────────────┤
│  Safety Layer (形式化保证，硬约束)                            │
│  ├── LTL 公式: G(!wall) & G(!boundary)                       │
│  ├── 动作过滤: 不安全动作 → 原地不动                          │
│  └── 零违规保证: 训练和测试全程安全                           │
├─────────────────────────────────────────────────────────────┤
│  Goal-Conditioned Policy (RL 学习，软目标)                   │
│  ├── 观测: 局部视野 (5×5) + 目标方向 (2)                     │
│  ├── 能力: Reach(goal) - 安全到达任意目标                    │
│  └── 泛化: 换目标不用重新训练                                │
└─────────────────────────────────────────────────────────────┘
```

核心原则：
- **Safety ≠ Reward** - 安全不进 RL 的 reward，用形式化方法硬保证
- **Goal = Parameter** - 目标是策略的参数，不是固定的
- **Policy reused across goals** - 一个策略，任意目标

## 项目结构

```
safe_rl_drone/
├── env.py                 # GridWorld 环境 (支持 random_goal/random_start)
├── ltl/                   # LTL 形式化方法模块
│   ├── propositions.py    # 原子命题管理
│   ├── parser.py          # LTL 公式解析（使用 Spot）
│   ├── robustness.py      # Robustness 递归计算
│   └── fsa.py             # FSA 监控自动机
├── safety/                # 安全层模块
│   ├── monitor.py         # 安全监控器
│   └── action_filter.py   # 动作过滤器
└── wrappers/              # 环境包装器
    └── safe_env_wrapper.py

tests/                     # 测试文件
├── test_ltl_modules.py    # LTL 模块测试
├── test_safe_wrapper.py   # 安全包装器测试
└── test_goal_conditioned.py # Goal-Conditioned 测试

maps/                      # 地图文件
├── map1.txt               # 6×6 简单地图
├── map2_medium.txt        # 10×10 中等地图
└── map3_maze.txt          # 迷宫地图

models/                    # 训练好的模型
└── ppo_safe.zip           # Goal-Conditioned 安全策略
```

## 安装

### Python 依赖

```bash
pip install -r requirements.txt
```

### Spot 库（LTL 解析）

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

# 训练 + 测试
python3 main.py

# 跳过训练，直接测试已有模型
python3 main.py --skip-train --load models/ppo_safe
```

## 配置说明

编辑 `config.yaml`：

```yaml
environment:
  map_path: "maps/map2_medium.txt"
  view_size: 5           # 局部视野大小
  random_goal: true      # 训练时随机目标
  random_start: true     # 训练时随机起点

safety:
  enabled: true
  formula: "G(!wall) & G(!boundary)"  # LTL 安全公式

training:
  total_timesteps: 100000
```

## 实验结果

在 10×10 地图上训练 100000 步：

| 指标 | 结果 |
|------|------|
| 训练撞墙次数 | 0 |
| 训练越界次数 | 0 |
| 安全干预率 | 4.62% |
| 测试成功率 | 100% (5/5) |
| 测试安全违规 | 0 |

验证了两个核心能力：
1. **目标泛化** - 随机起点/目标都能到达
2. **安全保证** - 全程零违规

## LTL 公式语法

| 算子 | 含义 | 示例 |
|------|------|------|
| `G(φ)` | 全局 (Always) | `G(!wall)` 永不撞墙 |
| `F(φ)` | 最终 (Eventually) | `F(goal)` 最终到达目标 |
| `X(φ)` | 下一步 (Next) | `X(safe)` 下一步安全 |
| `φ U ψ` | 直到 (Until) | `safe U goal` |
| `!`, `&`, `\|` | 非、与、或 | `!wall & !boundary` |

## 命令行参数

```bash
python3 main.py --safe true/false    # 开启/关闭安全层
python3 main.py --ltl "G(!wall)"     # 指定 LTL 公式
python3 main.py --timesteps 50000    # 训练步数
python3 main.py --skip-train         # 跳过训练
python3 main.py --load models/xxx    # 加载模型
```

## 运行测试

```bash
python3 -m pytest tests/ -v
```

## 后续计划

- [ ] Task FSA 集成 - 支持顺序任务 (F(A) & F(B) & ...)
- [ ] 更复杂地图验证
- [ ] 技能库扩展 (Avoid, Patrol, ...)

## 参考文献

- Xiao Li et al. "A formal methods approach to interpretable reinforcement learning for robotic planning" Science Robotics, 2019

## License

MIT
