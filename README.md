# Safe RL Drone

基于形式化方法的安全强化学习项目。使用 LTL (Linear Temporal Logic) 提供安全保证，确保智能体在训练和测试过程中不违反安全约束。

## 核心思想

- **Safety（安全）**: 用形式化方法保证，作为硬约束
- **Task（任务）**: 交给 RL 学习，作为软目标

这种设计既保证了安全性，又不伤害 RL 的泛化性能。

## 项目结构

```
safe_rl_drone/
├── env.py                 # GridWorld 环境
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
maps/                      # 地图文件
models/                    # 训练好的模型
```

## 安装

### 依赖

```bash
pip install -r requirements.txt
```

### Spot 库（LTL 解析）

Spot 需要从源码编译安装：

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

## 使用方法

### 基本运行

```bash
# 设置环境变量（WSL）
export PYTHONPATH=/usr/local/lib/python3.12/site-packages:$PYTHONPATH
export DISPLAY=:0

# 运行（使用默认配置）
python3 main.py
```

### 命令行参数

```bash
# 开启/关闭安全层
python3 main.py --safe true
python3 main.py --safe false

# 指定 LTL 公式
python3 main.py --ltl "G(!wall) & G(!boundary)"

# 跳过训练，直接测试
python3 main.py --skip-train

# 加载已有模型
python3 main.py --load models/ppo_safe
```

### 配置文件

编辑 `config.yaml` 进行详细配置：

```yaml
safety:
  enabled: true
  formula: "G(!wall)"      # LTL 安全公式
  unsafe_penalty: -1.0

task:
  use_robustness_reward: true
  robustness_weight: 0.1
```

## LTL 公式语法

支持的算子：
- `G(φ)` - 全局（Always）：φ 在所有时刻都成立
- `F(φ)` - 最终（Eventually）：φ 在某个时刻成立
- `X(φ)` - 下一步（Next）：φ 在下一时刻成立
- `φ U ψ` - 直到（Until）：φ 成立直到 ψ 成立
- `!φ` - 非（Not）
- `φ & ψ` - 与（And）
- `φ | ψ` - 或（Or）

示例公式：
- `G(!wall)` - 永远不撞墙
- `G(!wall) & G(!boundary)` - 永远不撞墙且不越界
- `G(!wall) & F(goal)` - 安全地到达目标

## 实验结果

在相同配置下训练 20000 步的对比：

| 指标 | 安全版本 | 不安全版本 |
|------|---------|-----------|
| 训练撞墙次数 | 0 | 6029 |
| 安全干预次数 | 6768 | 0 |
| 最终性能 | 到达目标 | 到达目标 |

安全层在训练过程中提供了零撞墙保证。

## 运行测试

```bash
# LTL 模块测试
python3 tests/test_ltl_modules.py

# 安全包装器测试
python3 tests/test_safe_wrapper.py
```

## 参考文献

- Hasanbeig et al. "A formal methods approach to interpretable reinforcement learning for robotic planning" Science Robotics, 2019

## License

MIT
