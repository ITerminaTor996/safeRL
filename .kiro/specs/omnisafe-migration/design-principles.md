# 设计原则与系统不变式

## 核心理念

本系统实现了一个**双规范架构**的安全强化学习系统，其中：

- **安全规范（Safety Specification）**：定义不可接受的行为（硬约束，零违规）
- **任务规范（Task Specification）**：定义优化目标（软目标，奖励塑形）

**核心洞察**：安全和任务成功是本质上不同的关注点，应该由不同的机制处理。

---

## 基本不变式

### 不变式 1：安全永远不依赖学习

**原则**：安全保证必须独立于 RL 策略的质量。

**含义**：
- `SafetyFilter` 没有可学习参数
- `SafetyFilter` 不接收奖励信号
- `SafetyFilter` 不基于训练进度更新
- 安全通过**运行时验证**强制执行，而非学习行为

**代码体现**：
```python
class SafetyFilter:
    def __init__(self, safety_formula, ...):
        # 没有 nn.Module，没有 optimizer，没有 loss
        self.automaton = build_automaton(safety_formula)  # 静态
    
    def filter_action(self, state, action):
        # 纯函数：state × action → safe_action
        # 没有 self.update()，没有反向传播
```

**为什么重要**：如果安全依赖学习，我们永远无法保证训练或部署期间的零违规。

---

### 不变式 2：任务失败是允许的

**原则**：RL 策略允许在任务上失败，但永远不允许违反安全。

**含义**：
- 任务自动机陷阱 → 负奖励，但 episode 继续
- 安全自动机陷阱 → 动作被否决，episode 安全继续
- 任务成功被优化，安全被强制执行

**代码体现**：
```python
# SafetyFilter
if self.automaton.is_rejecting(next_state):
    return safe_action  # 否决，防止违规

# TaskRewardShaper
if self.state_values[state] == -inf:
    return -10.0  # 惩罚，但不强制终止
```

**为什么重要**：这种分离允许 RL 探索和学习，而不损害安全。

---

### 不变式 3：自动机状态跟踪实际执行的行为

**原则**：FSA 状态基于**实际执行的动作**更新，而非提议的动作。

**含义**：
- 如果 `SafetyFilter` 修改了动作，FSA 看到的是修改后的动作
- FSA 状态必须与物理状态保持一致
- 没有"推测性"的 FSA 转移

**代码体现**：
```python
# 错误做法
safe_action, filtered = safety_filter.filter_action(state, action)
env.step(safe_action)  # 执行 safe_action
# 但 FSA 基于原始 action 更新了！❌

# 正确做法
safe_action, filtered = safety_filter.filter_action(state, action)
obs, reward, done, info = env.step(safe_action)
# FSA 基于 safe_action 的结果更新 ✓
safety_filter.update_fsa(obs)  # 保持一致
```

**为什么重要**：不一致的 FSA 状态会导致错误的安全判断和虚假违规。

---

### 不变式 4：命题评估基于真实状态

**原则**：命题评估必须基于**环境真实状态**，而非 RL 观测。

**含义**：
- 安全/任务监控器访问 `env.get_env_info()`，而非 `obs`
- 规范语义中没有 POMDP 歧义
- 规范定义在真实状态上，而非信念状态

**代码体现**：
```python
class PropositionEvaluator:
    def evaluate(self, prop_name, state):
        # state = env.get_env_info()  ← 真实状态
        # 不是 state = rl_observation
        
        if prop_name == "wall":
            true_position = state['agent_position']  # 不是从 obs 来的
            return self._check_collision(true_position)
```

**为什么重要**：
- 避免"智能体认为安全但实际不安全"的场景
- 使规范无歧义
- 分离感知（RL 的问题）和验证（形式化方法的问题）

---

### 不变式 5：Robustness 是连续的，自动机是离散的

**原则**：Robustness 提供稠密引导，自动机提供离散里程碑。

**含义**：
- Robustness (ρ)：从命题计算，连续实数值
- 自动机状态：离散，跟踪规范进度
- 两者都使用，但目的不同：
  - ρ → 稠密奖励塑形（每一步）
  - FSA → 稀疏里程碑奖励（状态转移）

**代码体现**：
```python
class TaskRewardShaper:
    def compute_reward(self, state):
        # 稠密：Robustness
        rho = compute_robustness(self.formula, state)
        
        # 稀疏：自动机进度
        next_fsa_state = self.automaton.step(...)
        progress_reward = 1.0 if (new_value > old_value) else 0.0
        
        # 组合
        return 0.1 * rho + 1.0 * progress_reward
```

**为什么重要**：单独的 Robustness 是短视的（贪心），单独的自动机是稀疏的。两者结合提供局部引导和全局结构。

---

## 设计决策与假设

### 决策 1：下一状态预测的线性外推

**当前方法**（Phase 1-2）：
```python
next_state = current_state + action * dt
```

**假设**：
- 动作效果在一个时间步内近似线性
- 非线性动力学（碰撞、摩擦）在 `dt` 内可忽略
- 安全裕度补偿预测误差

**局限性**：
- 对复杂物理不准确（Phase 3+）
- 假设小 `dt` 和平滑动力学

**未来改进**：
- 可达集过近似
- 预测误差的保守界
- 显式安全裕度调优

**为什么现在接受这个**：
- 对 Phase 1（离散网格）足够
- 对 Phase 2（简单 2D 运动学）合理
- 明确记录为假设，而非声明

---

### 决策 2：通过 BFS 检测任务陷阱

**方法**：
```python
state_values = BFS_from_accepting_states()
if state not in state_values:
    state_values[state] = -inf  # 任务陷阱
```

**假设**：
- 任务自动机是有限且完全可探索的
- 陷阱是没有路径到接受状态的状态

**处理**：
- 陷阱检测：显式检查 `-inf` 值
- 陷阱惩罚：大负奖励
- 不强制终止：Episode 继续（安全仍被强制执行）

**为什么有效**：
- 使任务陷阱在代码中显式化
- 与"任务失败是允许的"原则一致
- 向 RL 提供清晰信号："你卡住了，试试别的"

---

### 决策 3：奖励权重作为超参数

**当前方法**：
```yaml
reward:
  task_shaping:
    robustness: 0.1
    progress: 1.0
    acceptance: 10.0
```

**理由**：
- 没有通用的"正确"权重
- 任务相关（简单 vs 复杂）
- 通过配置调整，无需代码修改

**指导原则**：
- Robustness：小（0.01-0.1），提供梯度
- Progress：中（1.0），里程碑激励
- Acceptance：大（10.0），终端目标

**为什么不学习**：
- 奖励塑形是问题规范的一部分，而非解决方案
- 用户应该显式控制任务优先级

---

## 三层架构

```
┌─────────────────────────────────────────┐
│  RL 策略（Goal-Conditioned）             │
│  - 学习泛化                              │
│  - 优化任务成功                          │
│  - 无安全责任                            │
└─────────────────┬───────────────────────┘
                  │ 提议动作
                  ↓
┌─────────────────────────────────────────┐
│  安全层（形式化方法）                     │
│  - SafetyFilter: 否决不安全动作          │
│  - TaskRewardShaper: 引导学习            │
│  - 不学习，纯验证                        │
└─────────────────┬───────────────────────┘
                  │ 安全动作 + 塑形奖励
                  ↓
┌─────────────────────────────────────────┐
│  环境（真实状态）                         │
│  - 执行动作                              │
│  - 提供真实状态                          │
│  - 无规范知识                            │
└─────────────────────────────────────────┘
```

**核心洞察**：每一层都有清晰、不重叠的职责。

---

## 本系统不是什么

为了澄清设计，这里明确说明我们**不做**什么：

❌ **不是"安全 RL 算法"**：我们不修改 PPO/SAC 等来"学习安全"

❌ **不是基于约束的 RL**：我们不使用拉格朗日乘子或成本函数

❌ **不是作为备份的屏蔽**：安全不是"最后手段"，而是始终活跃

❌ **不是规范合成**：用户提供规范，我们不推断规范

❌ **不感知 POMDP**：规范定义在真实状态上，而非信念

---

## 本系统是什么

✅ **运行时强制执行 + RL 优化**：安全被强制执行，任务被优化

✅ **双规范架构**：安全和任务是分离的、显式的规范

✅ **Robustness 驱动的奖励塑形**：稠密、可解释、理论基础

✅ **零违规保证**：在声明的假设下（预测准确性、规范正确性）

✅ **目标条件泛化**：RL 提供适应性，形式化方法提供安全

---

## 实现检查清单

在实现任何组件之前，验证：

- [ ] 这个组件学习吗？（如果是，它不是安全关键的）
- [ ] 这个组件访问真实状态吗？（命题必须访问）
- [ ] 这个组件更新 FSA 状态吗？（必须与执行的动作一致）
- [ ] 这个组件做了假设吗？（显式记录它们）
- [ ] 这个组件有清晰的失败模式吗？（失败时会发生什么？）

---

## 论文级问题表述（草稿）

**问题**：给定一个系统，状态空间 $\mathcal{S}$，动作空间 $\mathcal{A}$，动力学 $s_{t+1} = f(s_t, a_t)$，以及两个 STL 规范：

- **安全规范** $\varphi_{\text{safe}}$：定义不可接受的行为
- **任务规范** $\varphi_{\text{task}}$：定义优化目标

**目标**：学习一个策略 $\pi: \mathcal{S} \times \mathcal{G} \to \mathcal{A}$ 使得：

1. **安全**：$\forall s_0, g. \; \text{trace}(\pi, s_0, g) \models \varphi_{\text{safe}}$（硬约束）
2. **任务**：$\max_\pi \mathbb{E}[\rho(\varphi_{\text{task}}, \text{trace}(\pi, s_0, g))]$（软目标）
3. **泛化**：$\pi$ 泛化到未见过的目标 $g \in \mathcal{G}$

**方法**：
- 安全：通过基于自动机的动作过滤进行运行时强制执行
- 任务：Robustness 驱动的奖励塑形 + 自动机进度奖励
- 泛化：目标条件策略架构

**核心贡献**：安全（强制执行）和任务（优化）的分离使得可证明的安全与学习的泛化成为可能。

---

## 版本历史

- **v1.0** (2025-01-17): 初始设计原则文档
- 基于 GPT-4 对实施计划的分析反馈
- 融合了 Phase 0（离散网格世界）的经验教训

---

## 未来论文参考文献

- **运行时强制执行**: Falcone et al., "Runtime Enforcement of Temporal Properties"
- **STL Robustness**: Donzé & Maler, "Robust Satisfaction of Temporal Logic"
- **基于势能的塑形**: Ng et al., "Policy Invariance Under Reward Shaping"
- **目标条件 RL**: Schaul et al., "Universal Value Function Approximators"
