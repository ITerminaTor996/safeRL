# Safe RL Drone

基于形式化方法的安全强化学习项目。使用 LTL (Linear Temporal Logic) 提供安全保证，结合 Goal-Conditioned RL 实现目标泛化。

## 核心设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Architecture v2.0                         │
├─────────────────────────────────────────────────────────────┤
│  Safety Layer (形式化保证，硬约束)                            │
│  ├── Safety Specification: G(!wall) & G(!boundary)          │
│  ├── SafetyFilter: 基于 FSA 的动作过滤                       │
│  └── 零违规保证: 训练和测试全程安全                           │
├─────────────────────────────────────────────────────────────┤
│  Task Layer (RL 学习，软目标)                                │
│  ├── Task Specification: F(goal)                            │
│  ├── TaskRewardShaper: 6 个奖励组件                         │
│  │   ├── Robustness 增量 (稠密引导)                         │
│  │   ├── FSA 进度 (稀疏里程碑)                              │
│  │   ├── 任务完成 (最终目标)                                │
│  │   ├── 陷阱惩罚 (任务失败)                                │
│  │   ├── 过滤器惩罚 (学习安全)                              │
│  │   └── 时间惩罚 (鼓励效率)                                │
│  └── Goal-Conditioned Policy: 泛化到任意目标                │
└─────────────────────────────────────────────────────────────┘
```

核心原则：
- **双规范架构** - Safety 用形式化保证（硬约束），Task 交给 RL 学习（软目标）
- **Safety ≠ Reward** - 安全不进 RL 的 reward，用 SafetyFilter 硬保证
- **Task = LTL + Reward Shaping** - 任务用 LTL 定义，用 Robustness + FSA 引导学习
- **Goal = Parameter** - 目标是策略的参数，不是固定的

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
│   ├── action_filter.py   # SafetyFilter (基于 Safety FSA)
│   ├── task_reward_shaper.py  # TaskRewardShaper (基于 Task FSA)
│   └── monitor.py         # [已弃用] 旧的安全监控器
└── wrappers/              # 环境包装器
    └── safe_env_wrapper.py

tests/                     # 测试文件
├── test_ltl_modules.py    # LTL 模块测试
├── test_safety_filter.py  # SafetyFilter 测试
├── test_task_reward_shaper.py  # TaskRewardShaper 测试
├── test_safe_wrapper.py   # 安全包装器测试
└── test_goal_conditioned.py # Goal-Conditioned 测试

maps/                      # 地图文件
├── map1.txt               # 6×6 简单地图
├── map2_medium.txt        # 10×10 中等地图
└── map4_large.txt         # 20×20 大地图

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
  map_path: "maps/map4_large.txt"
  view_size: 5           # 局部视野大小
  random_goal: true      # 训练时随机目标
  random_start: true     # 训练时随机起点

# 双规范配置
specifications:
  # 安全规范（硬约束）
  safety:
    enabled: true
    formula: "G(!wall) & G(!boundary)"
  
  # 任务规范（软引导）
  task:
    enabled: true
    formula: "F(goal)"

# 奖励权重
reward:
  task_shaping_weights:
    robustness: 0.1     # Robustness 增量
    progress: 1.0       # FSA 进度
    acceptance: 10.0    # 任务完成
    trap: 1.0           # 陷阱惩罚
    filter: 1.0         # 过滤器惩罚
    time: 1.0           # 时间惩罚

training:
  total_timesteps: 500000
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

**Phase 1: 双规范架构** (当前进度)
- [x] Step 1.1: 双规范配置系统
- [x] Step 1.2: SafetyFilter 类（基于 Safety FSA）
- [x] Step 1.3: TaskRewardShaper 类（基于 Task FSA + Robustness）
- [ ] Step 1.4: 集成到 SafeEnvWrapper
- [ ] Step 1.5: 测试双规范架构

**Phase 2: 连续环境迁移**
- [ ] PyBullet 2D 导航环境
- [ ] 连续动作空间适配
- [ ] 几何安全过滤器（CBF）

**Phase 3: 动态环境**
- [ ] 动态障碍物
- [ ] 在线规范更新

## 参考文献

- Xiao Li et al. "A formal methods approach to interpretable reinforcement learning for robotic planning" Science Robotics, 2019

## License

MIT
