# 需求文档：形式化安全强化学习框架升级计划

## 项目核心理念

本项目的核心贡献是：**将形式化方法与强化学习结合，实现可证明的安全保证和可解释的奖励塑形**。

### 核心创新点

1. **双规范架构**：
   - **安全规范（Safety Specification）**：用户指定的"坏事"，通过 Shield/Filter 强制保证（硬约束，零违规）
   - **任务规范（Task Specification）**：用户指定的"好事"，通过 Robustness 引导 RL 学习（软目标，奖励塑形）

2. **形式化 + RL 的协同**：
   - 形式化方法：提供安全保证和可解释的奖励信号
   - RL 方法：提供泛化能力和自适应控制
   - 两者互补，而非竞争

3. **任意 LTL 规范支持**：
   - 不限于简单的 reach-avoid
   - 支持顺序任务、条件任务、时间约束等复杂规范
   - 用户可以用 LTL 表达任意安全和任务需求

4. **Robustness 驱动的可解释训练**：
   - Robustness 值提供稠密、连续的奖励信号
   - 自动机状态跳转提供稀疏的里程碑奖励
   - 训练过程可解释、可调试

### 设计原则

- **Safety 用形式化保证**（硬约束，零违规）
- **Task 交给 RL 学习**（软目标，Goal-Conditioned Policy）
- **环境是载体，不是重点**（从离散到连续，验证方法的通用性）

## 术语表

- **LTL (Linear Temporal Logic)**: 线性时序逻辑，布尔语义（满足/不满足）
- **TLTL (Time-Window Temporal Logic)**: 时间窗口时序逻辑，在有限时间窗口内评估的 LTL
- **STL (Signal Temporal Logic)**: 信号时序逻辑，支持 **Robustness 语义**（实数值，表示满足程度）
- **Robustness (ρ)**: 鲁棒度，量化命题满足程度的**连续实数值**
  - ρ > 0: 满足，值越大越"安全"
  - ρ < 0: 违反，值越小越"危险"
  - |ρ|: 满足/违反的程度
- **FSA (Finite State Automaton)**: 有限状态自动机，时序逻辑公式的可执行表示
- **Safety Specification**: 安全规范，定义"不希望发生的事"（如撞墙、越界）
- **Task Specification**: 任务规范，定义"希望达成的目标"（如到达目标、顺序访问）
- **Shield/Filter**: 安全过滤器，在执行前拦截不安全动作
- **Reward Shaping**: 奖励塑形，用 Robustness 引导 RL 学习
- **Goal-Conditioned Policy**: 目标条件策略，可泛化到任意目标的控制策略
- **PyBullet**: 开源物理仿真引擎，支持刚体动力学和 3D 渲染

**注**：本项目使用 **STL 的 Robustness 语义**
- 公式语法：类似 LTL（G, F, U, &, |, !）
- 语义：Robustness（实数值，不是布尔值）
- 时间：离散时间步（每个 `env.step()`）
- 状态：连续（位置、速度、距离）
- 核心创新：将 Robustness 用于安全过滤和奖励塑形
- **Safety Specification**: 安全规范，定义"不希望发生的事"（如撞墙、越界）
- **Task Specification**: 任务规范，定义"希望达成的目标"（如到达目标、顺序访问）
- **Shield/Filter**: 安全过滤器，在执行前拦截不安全动作
- **Reward Shaping**: 奖励塑形，用 Robustness 引导 RL 学习
- **Goal-Conditioned Policy**: 目标条件策略，可泛化到任意目标的控制策略
- **PyBullet**: 开源物理仿真引擎，支持刚体动力学和 3D 渲染

**注**：本项目采用渐进式方法：
- Phase 1-2: 使用 LTL（离散时间步）
- Phase 3: 引入 TLTL（时间界约束）
- Phase 4: 可选 STL（连续信号约束）

## 升级路线图

本次升级的核心目标是：**从简单的 reach-avoid 任务扩展到任意 LTL 规范，并迁移到连续物理环境验证方法的通用性**。

### 整体架构

```
用户输入 LTL 规范
    ↓
┌─────────────────────────────────────┐
│  规范解析与自动机构建                │
│  - Safety Formula → Safety FSA      │
│  - Task Formula → Task FSA          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  连续环境 (PyBullet 2D/3D)          │
│  - 场景配置系统                      │
│  - 物理仿真                          │
│  - 动态障碍物支持                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  命题评估层                          │
│  - 连续距离计算                      │
│  - Robustness 计算                   │
│  - 命题真值判断                      │
└─────────────────────────────────────┘
    ↓
┌──────────────────┬──────────────────┐
│  安全过滤器       │  任务奖励塑形     │
│  (Safety Filter) │  (Reward Shaper) │
│                  │                  │
│  - 预测下一状态   │  - 计算 ρ 值     │
│  - 检查 FSA 转移  │  - 检查 FSA 进度 │
│  - 过滤不安全动作 │  - 组合奖励信号   │
└──────────────────┴──────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Goal-Conditioned RL Policy         │
│  - PPO 训练                          │
│  - 目标泛化                          │
│  - 零安全违规                        │
└─────────────────────────────────────┘
```

### Phase 1: 双规范架构 + 任意 LTL 支持

**目标**: 实现安全规范和任务规范的分离，支持任意 LTL 公式

**核心模块**:

1. **规范解析与自动机构建**
   - 输入：两个 LTL 公式字符串
   - 输出：两个 FSA（Safety FSA 和 Task FSA）
   - 技术：使用 Spot 库，已有基础

2. **双自动机监控系统**
   - Safety FSA：识别违规路径，用于动作过滤
   - Task FSA：识别任务进度，用于奖励塑形
   - 状态跳转：提供里程碑奖励

3. **安全过滤器（基于 Safety FSA）**
   - 预测：`next_state = current_state + action * dt`
   - 评估：计算命题真值
   - 检查：是否会进入 FSA 的拒绝状态
   - 过滤：修正或拒绝不安全动作

4. **任务奖励塑形器（基于 Task FSA + Robustness）**
   - Robustness 奖励：稠密信号，引导方向
   - 状态跳转奖励：稀疏信号，里程碑激励
   - 接受状态奖励：任务完成大奖励

**测试场景**:
- 简单任务：`Safety="G(!wall)"`, `Task="F(goal)"`
- 顺序任务：`Safety="G(!wall)"`, `Task="F(wp1) & F(wp2) & F(goal)"`
- 条件任务：`Safety="G(danger → F[0,5](safe))"`, `Task="F(goal)"`

**成功标准**:
- ✅ 支持任意 LTL 公式输入
- ✅ 安全规范零违规
- ✅ 任务规范成功率 > 80%
- ✅ Robustness 奖励有效引导训练

---

### Phase 2: 连续环境迁移 (PyBullet 2D)

**目标**: 将 Phase 1 的方法迁移到连续物理环境，验证通用性

**核心模块**:

1. **场景配置系统**
   - YAML 配置文件定义场景
   - 物体：类型、形状、位置、大小、颜色
   - 支持动态障碍物配置
   - 参考 VernaCopter 的设计

2. **PyBullet 2D 环境**
   - 平面机器人（圆柱体，速度控制）
   - 物理仿真：碰撞、摩擦、惯性
   - 渲染：3D 可视化（俯视图）
   - Gymnasium 接口

3. **连续空间命题评估**
   - 距离计算：欧氏距离
   - Robustness 定义：
     - `ρ(near_obstacle) = safety_threshold - min_distance`
     - `ρ(at_goal) = goal_threshold - distance_to_goal`
   - 命题真值：`ρ > 0` 为真

4. **连续动作过滤**
   - 几何投影（Phase 2 简化版）
   - 线性外推预测
   - 切线方向修正
   - 零动作兜底

5. **Goal-Conditioned Policy 适配**
   - 观测空间：`[x, y, vx, vy, goal_dx, goal_dy]`
   - 动作空间：`[vx, vy]` 连续速度
   - 随机目标/起点采样
   - 保持泛化能力

**测试场景**:
- 同 Phase 1 的 LTL 公式
- 在连续空间中验证

**成功标准**:
- ✅ PyBullet 环境稳定运行
- ✅ 连续空间 Robustness 计算正确
- ✅ 动作过滤有效（零违规）
- ✅ Goal-Conditioned Policy 训练成功（成功率 > 90%）
- ✅ 策略泛化到随机目标

---

### Phase 3: 动态环境与复杂规范

**目标**: 支持动态障碍物和更复杂的 LTL 规范

**核心模块**:

1. **动态障碍物系统**
   - 场景配置中标记 `dynamic: true`
   - 运行时随机出现/消失
   - 实时反应能力验证

2. **时间约束支持（TLTL）**
   - `F[0,T](goal)`: T 步内到达
   - `G[0,T](!danger)`: T 步内避开危险
   - 有界自动机构建

3. **复杂任务规范**
   - 多目标选择：`F(goal1) | F(goal2)`
   - 条件响应：`G(trigger → F[0,5](response))`
   - 嵌套时序：`G(F(checkpoint))`

**测试场景**:
- 动态避障：障碍物随机消失/出现
- 时间限制：100 步内到达目标
- 顺序 + 条件：先到 A，遇到 B 则去 C，最后到 D

**成功标准**:
- ✅ 动态障碍物下零违规
- ✅ 实时反应能力
- ✅ 时间约束满足率 > 70%
- ✅ 复杂任务完成率 > 60%

---

### Phase 4 (可选): 3D 扩展

**目标**: 扩展到 3D 空间，验证方法的可扩展性

**核心模块**:
- 四旋翼模型（简化的速度控制）
- 3D 命题评估
- 高度约束

**成功标准**:
- ✅ 3D 导航成功率 > 80%
- ✅ 高度约束满足

---

## 详细需求

### 需求 1: 双规范架构

**用户故事**: 作为研究者，我希望分别指定安全规范和任务规范，以便清晰地表达"不希望发生的事"和"希望达成的目标"。

#### 验收标准

1. 配置文件应支持两个独立的 LTL 公式：
   ```yaml
   safety:
     formula: "G(!wall) & G(!boundary)"
   task:
     formula: "F(wp1) & F(wp2) & F(goal)"
   ```

2. 系统应为每个公式构建独立的 FSA

3. Safety FSA 应识别拒绝状态（违规路径）

4. Task FSA 应识别接受状态（任务完成）和状态价值（到接受状态的距离）

5. 两个 FSA 应独立运行，互不干扰

---

### 需求 2: 安全过滤器（基于 Safety FSA）

**用户故事**: 作为研究者，我希望系统能自动过滤所有会导致安全违规的动作，以便保证零违规。

#### 验收标准

1. 当 RL 输出动作时，过滤器应：
   - 预测下一状态
   - 评估命题真值
   - 检查 Safety FSA 是否会进入拒绝状态

2. 如果动作不安全，过滤器应：
   - 尝试修正动作（减小幅度、改变方向）
   - 如果无法修正，返回零动作（停止）
   - 记录干预次数

3. 过滤器应支持：
   - 离散动作空间（Phase 1 测试）
   - 连续动作空间（Phase 2）

4. 过滤策略应可配置：
   - `geometric`: 几何投影（Phase 2）
   - `cbf`: 控制屏障函数（Phase 4 可选）

#### 技术细节

```python
class SafetyFilter:
    def filter_action(self, state, action):
        # 1. 预测
        next_state = predict(state, action, dt)
        
        # 2. 评估命题
        props = evaluate_propositions(next_state)
        
        # 3. 检查 FSA
        next_fsa_state = safety_fsa.step(current_fsa_state, props)
        
        # 4. 判断
        if safety_fsa.is_rejecting(next_fsa_state):
            # 不安全，修正
            safe_action = find_safe_action(state, action)
            return safe_action, True  # filtered
        else:
            # 安全，通过
            return action, False
```

---

### 需求 3: 任务奖励塑形器（基于 Task FSA + Robustness）

**用户故事**: 作为研究者，我希望系统能根据任务规范自动生成稠密的奖励信号，以便高效训练 RL 策略。

#### 验收标准

1. 奖励塑形器应计算三种奖励：
   - **Robustness 奖励**（稠密）：基于命题的 ρ 值
   - **状态跳转奖励**（稀疏）：FSA 状态更接近接受状态
   - **接受状态奖励**（稀疏）：到达 FSA 接受状态

2. Robustness 计算应：
   - 从原子命题层面计算（基于距离）
   - 递归组合到公式层面（使用现有算法）
   - 提供连续的引导信号

3. FSA 状态价值应：
   - 通过 BFS 预计算（到接受状态的最短距离）
   - 状态跳转时比较新旧价值
   - 价值增加 → 给予正奖励

4. 总奖励应可配置权重：
   ```yaml
   task:
     reward_weights:
       robustness: 0.1      # 稠密引导
       progress: 1.0        # 里程碑
       acceptance: 10.0     # 完成
   ```

#### 技术细节

```python
class TaskRewardShaper:
    def compute_reward(self, state):
        # 1. Robustness（稠密）
        rho = compute_robustness(task_formula, state)
        
        # 2. FSA 状态跳转（稀疏）
        props = evaluate_propositions(state)
        next_fsa_state = task_fsa.step(current_fsa_state, props)
        
        progress_reward = 0.0
        if next_fsa_state != current_fsa_state:
            if state_value[next_fsa_state] > state_value[current_fsa_state]:
                progress_reward = 1.0
        
        # 3. 接受状态（稀疏）
        acceptance_reward = 0.0
        if task_fsa.is_accepting(next_fsa_state):
            acceptance_reward = 10.0
        
        # 4. 组合
        total_reward = (
            0.1 * rho +
            1.0 * progress_reward +
            10.0 * acceptance_reward
        )
        
        return total_reward
```

---

### 需求 4: 连续空间命题评估

**用户故事**: 作为研究者，我希望在连续空间中评估命题真值和 Robustness，以便支持连续环境。

#### 验收标准

1. 命题评估器应支持：
   - 障碍物：`ρ(near_obstacle) = safety_threshold - min_distance_to_obstacles`
   - 目标：`ρ(at_goal) = goal_threshold - distance_to_goal`
   - 边界：`ρ(near_boundary) = boundary_margin - distance_to_boundary`
   - Waypoint：`ρ(at_waypoint) = waypoint_threshold - distance_to_waypoint`

2. 距离计算应：
   - 使用欧氏距离
   - 考虑物体形状（圆柱、立方体）
   - 高效计算（避免遍历所有障碍物）

3. 阈值应可配置：
   ```yaml
   propositions:
     obstacle:
       safety_threshold: 0.8  # 机器人半径 + 障碍物半径 + 裕度
     goal:
       reach_threshold: 0.5
     boundary:
       margin: 0.5
   ```

4. 命题真值应：
   - 从 Robustness 导出：`prop = (ρ > 0)`
   - 用于 FSA 状态转移

---

### 需求 5: PyBullet 2D 环境

**用户故事**: 作为研究者，我希望在 PyBullet 中创建 2D 导航环境，以便验证方法在连续空间的有效性。

#### 验收标准

1. 环境应支持：
   - 平面机器人（圆柱体，半径 0.3m）
   - 连续动作空间：`[vx, vy]` 速度控制
   - 连续观测空间：`[x, y, vx, vy, goal_dx, goal_dy]`
   - 物理仿真：碰撞、摩擦、惯性

2. 场景应通过 YAML 配置：
   ```yaml
   objects:
     goal:
       shape: "cylinder"
       position: [4.0, 4.0, 0.0]
       radius: 0.5
     obstacle1:
       shape: "cylinder"
       position: [2.0, 2.0, 0.0]
       radius: 0.5
       dynamic: true  # 支持动态
   ```

3. 环境应提供：
   - `get_object_positions()`: 获取所有物体位置（用于命题评估）
   - `update_dynamic_obstacles()`: 更新动态障碍物
   - 3D 渲染（俯视图清晰）

4. 环境应支持：
   - `random_goal=True`: 随机目标采样
   - `random_start=True`: 随机起点采样
   - Goal-Conditioned RL

---

### 需求 6: Goal-Conditioned Policy 适配

**用户故事**: 作为研究者，我希望 Goal-Conditioned Policy 能在连续空间工作，以便保持泛化能力。

#### 验收标准

1. 策略应能：
   - 训练时：随机目标和起点
   - 测试时：泛化到任意目标
   - 保持零安全违规（通过 SafeEnvWrapper）

2. 观测空间应包含：
   - 当前状态：位置、速度
   - 目标方向：归一化向量

3. 训练应：
   - 使用 PPO 算法
   - 结合 Robustness 奖励
   - 结合 FSA 状态跳转奖励

4. 测试应验证：
   - 成功率 > 90%
   - 零安全违规
   - 泛化到训练时未见过的目标

---

### 需求 7: 动态障碍物支持

**用户故事**: 作为研究者，我希望支持动态障碍物，以便验证策略的实时反应能力。

#### 验收标准

1. 场景配置应支持：
   ```yaml
   obstacle1:
     dynamic: true
     change_prob: 0.01  # 每步变化概率
   ```

2. 动态行为应：
   - 在原始障碍物位置随机出现/消失
   - 保证基础路径始终存在
   - 不在 agent/目标位置出现

3. 策略应：
   - 每帧重新获取观测（包含动态变化）
   - 实时调整路径
   - 保持零违规

4. 测试应验证：
   - 动态环境下成功率 > 80%
   - 零安全违规
   - 实时反应能力

---

## 通用需求

### 需求 G1: 配置系统

**验收标准**:

1. 支持场景配置（YAML）：
   - 环境类型、边界
   - 物体定义
   - 初始状态

2. 支持规范配置（YAML）：
   - 安全公式
   - 任务公式
   - 命题阈值
   - 奖励权重

3. 支持训练配置（YAML）：
   - RL 算法参数
   - 训练步数
   - 测试参数

### 需求 G2: 测试与验证

**验收标准**:

1. 每个模块应有单元测试：
   - `test_safety_filter.py`
   - `test_task_reward_shaper.py`
   - `test_proposition_evaluator.py`
   - `test_pybullet_env.py`

2. 集成测试应验证：
   - 端到端训练流程
   - 零安全违规
   - 目标泛化
   - 动态环境适应

3. 测试应包含可视化验证

### 需求 G3: 文档与示例

**验收标准**:

1. 每个 Phase 应有：
   - 设计文档
   - 使用示例
   - 测试报告

2. README 应包含：
   - 项目理念
   - 快速开始
   - 配置说明
   - 扩展指南

---

## 实施顺序与里程碑

### Milestone 1: 双规范架构（2-3周）

**目标**: 实现安全规范和任务规范的分离，支持任意 LTL

**任务**:
1. 修改配置系统：支持两个独立的 LTL 公式
2. 实现 `SafetyFilter` 类（基于 Safety FSA）
3. 实现 `TaskRewardShaper` 类（基于 Task FSA + Robustness）
4. 修改 `SafeEnvWrapper` 集成两个组件
5. 测试：简单任务、顺序任务

**交付物**:
- 代码：`safety/safety_filter.py`, `safety/task_reward_shaper.py`
- 测试：`tests/test_dual_specification.py`
- 文档：Phase 1 设计文档

---

### Milestone 2: 场景配置系统（1周）

**目标**: 实现灵活的场景配置，支持 YAML 定义

**任务**:
1. 设计场景配置 YAML 格式
2. 实现 `ScenarioLoader` 类
3. 创建示例场景配置文件
4. 测试：加载场景、验证物体定义

**交付物**:
- 代码：`scenarios/scenario_loader.py`
- 配置：`scenarios/reach_avoid_2d.yaml`
- 测试：`tests/test_scenario_loader.py`

---

### Milestone 3: PyBullet 2D 环境（2周）

**目标**: 创建连续 2D 导航环境

**任务**:
1. 实现 `PyBullet2DEnv` 类（Gymnasium 接口）
2. 集成 `ScenarioLoader`
3. 实现物理仿真和渲染
4. 实现 Goal-Conditioned 支持
5. 测试：机器人移动、碰撞检测、目标到达

**交付物**:
- 代码：`envs/pybullet_2d_env.py`
- 测试：`tests/test_pybullet_2d_env.py`
- 示例：`examples/pybullet_2d_demo.py`

---

### Milestone 4: 连续空间命题评估（1周）

**目标**: 实现基于距离的 Robustness 计算

**任务**:
1. 修改 `PropositionEvaluator` 支持连续距离
2. 实现距离计算函数
3. 适配 Robustness 递归算法
4. 测试：距离计算、命题真值、Robustness 值

**交付物**:
- 代码：修改 `ltl/propositions.py`
- 测试：`tests/test_continuous_propositions.py`

---

### Milestone 5: 连续动作过滤（1周）

**目标**: 实现连续空间的安全过滤

**任务**:
1. 修改 `SafetyFilter` 支持连续动作
2. 实现几何投影算法
3. 实现动作预测和修正
4. 测试：不安全动作过滤、边界裁剪

**交付物**:
- 代码：修改 `safety/safety_filter.py`
- 测试：`tests/test_continuous_filter.py`

---

### Milestone 6: 端到端训练与验证（1-2周）

**目标**: 在 PyBullet 2D 环境中训练 Goal-Conditioned Policy

**任务**:
1. 适配 `main.py` 支持 PyBullet 环境
2. 训练简单任务：`Safety="G(!obstacle)"`, `Task="F(goal)"`
3. 训练顺序任务：`Task="F(wp1) & F(wp2) & F(goal)"`
4. 验证：零违规、目标泛化、成功率

**交付物**:
- 训练脚本：修改 `main.py`
- 测试报告：Phase 2 测试报告
- 模型：训练好的策略

---

### Milestone 7: 动态障碍物（1周）

**目标**: 支持动态障碍物和实时反应

**任务**:
1. 在场景配置中添加 `dynamic` 支持
2. 实现动态更新逻辑
3. 训练和测试动态环境
4. 验证：零违规、实时反应

**交付物**:
- 代码：修改 `envs/pybullet_2d_env.py`
- 配置：`scenarios/dynamic_obstacles.yaml`
- 测试：`tests/test_dynamic_obstacles.py`

---

### Milestone 8 (可选): 3D 扩展（2-3周）

**目标**: 扩展到 3D 空间

**任务**:
1. 实现 `PyBullet3DEnv` 类
2. 四旋翼模型（简化速度控制）
3. 3D 命题评估
4. 训练和测试

**交付物**:
- 代码：`envs/pybullet_3d_env.py`
- 测试：`tests/test_pybullet_3d_env.py`

---

## 总体时间估算

- **Phase 1 (双规范架构)**: 2-3 周
- **Phase 2 (连续环境迁移)**: 5-6 周
- **Phase 3 (动态环境)**: 1 周
- **Phase 4 (3D 扩展，可选)**: 2-3 周

**总计**: 8-10 周（不含 Phase 4）

---

## 成功标准总结

### Phase 1 成功标准
- ✅ 支持任意 LTL 公式输入（安全 + 任务）
- ✅ 安全规范零违规
- ✅ 任务规范成功率 > 80%
- ✅ Robustness 奖励有效引导训练
- ✅ FSA 状态跳转奖励有效

### Phase 2 成功标准
- ✅ PyBullet 2D 环境稳定运行
- ✅ 连续空间 Robustness 计算正确
- ✅ 动作过滤有效（零违规）
- ✅ Goal-Conditioned Policy 训练成功（成功率 > 90%）
- ✅ 策略泛化到随机目标
- ✅ 所有测试通过

### Phase 3 成功标准
- ✅ 动态障碍物下零违规
- ✅ 实时反应能力
- ✅ 动态环境成功率 > 80%

### Phase 4 成功标准（可选）
- ✅ 3D 导航成功率 > 80%
- ✅ 高度约束满足
