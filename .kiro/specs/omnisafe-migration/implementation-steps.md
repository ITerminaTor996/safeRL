# 实施步骤详细计划

## 工作流程

每一步的流程：
1. **讨论设计方案**（类接口、代码结构、关键逻辑）
2. **用户审查和批准**（提出修改意见）
3. **实现代码**
4. **用户审查代码**（确保理解每一行）
5. **用户运行测试**（验证功能）
6. **进入下一步**

**核心原则**：主要代码流程由用户把握，每步都需要用户批准后才进行。

---

## Phase 1: 双规范架构（预计 2-3 周）

### Step 1.1: 双规范配置系统

**目标**: 修改 config.yaml，支持分别指定安全规范和任务规范

**当前状态**:
```yaml
safety:
  formula: "G(!wall) & G(!boundary)"
task:
  type: "reach_goal"  # 隐式
```

**目标状态**:
```yaml
specifications:
  safety:
    formula: "G(!wall) & G(!boundary)"
  task:
    formula: "F(goal)"  # 显式 STL 公式
```

**要做什么**:
1. 重组 config.yaml 结构
2. 添加 `specifications` 顶层键
3. 明确 `task.formula` 字段
4. 添加奖励权重配置

**讨论点**:
- 配置文件格式是否合理？
- 是否需要其他配置项？
- 向后兼容性如何处理？

**交付物**:
- 修改后的 `config.yaml`
- 配置加载代码（如果需要修改）

**预计时间**: 30 分钟讨论 + 10 分钟实现

---

### Step 1.2: SafetyFilter 类

**目标**: 创建独立的安全过滤器类，基于 Safety FSA

**设计方案**:

```python
# safe_rl_drone/safety/safety_filter.py

class SafetyFilter:
    """
    安全过滤器：基于 Safety FSA 过滤不安全动作
    """
    def __init__(self, safety_formula, prop_evaluator, env_info):
        """
        Args:
            safety_formula: 安全规范字符串（如 "G(!wall) & G(!boundary)"）
            prop_evaluator: 命题评估器
            env_info: 环境信息（用于动作预测）
        """
        self.formula = parse_formula(safety_formula)
        self.automaton = build_automaton(safety_formula)
        self.prop_evaluator = prop_evaluator
        self.current_state = self.automaton.initial_state
        
    def filter_action(self, state, action, dt=1.0):
        """
        过滤动作
        
        Args:
            state: 当前状态（dict 或 array）
            action: RL 输出的动作
            dt: 时间步长
            
        Returns:
            safe_action: 安全的动作
            filtered: 是否被过滤（bool）
            info: 调试信息（dict）
        """
        # 1. 预测下一状态
        next_state = self._predict_next_state(state, action, dt)
        
        # 2. 评估命题真值
        props = self.prop_evaluator.evaluate_all(next_state)
        
        # 3. 检查 FSA 转移
        next_fsa_state = self.automaton.step(self.current_state, props)
        
        # 4. 判断是否安全
        if self.automaton.is_rejecting(next_fsa_state):
            # 不安全，尝试修正
            safe_action, success = self._find_safe_action(state, action, dt)
            return safe_action, True, {'filtered': True, 'success': success}
        else:
            # 安全，通过
            self.current_state = next_fsa_state
            return action, False, {'filtered': False}
    
    def _predict_next_state(self, state, action, dt):
        """预测下一状态（简化版：线性外推）"""
        # 离散环境：直接应用动作
        # 连续环境：state + action * dt
        pass
    
    def _find_safe_action(self, state, action, dt):
        """寻找安全替代动作"""
        # 策略1：减小幅度
        # 策略2：零动作
        # 策略3：切线方向（Phase 2）
        pass
    
    def reset(self):
        """重置 FSA 状态"""
        self.current_state = self.automaton.initial_state
```

**讨论点**:
1. 类接口是否合理？
2. `_predict_next_state` 如何实现（离散 vs 连续）？
3. `_find_safe_action` 的策略优先级？
4. 如何与现有 `ActionFilter` 集成或替换？

**交付物**:
- `safe_rl_drone/safety/safety_filter.py`
- 单元测试：`tests/test_safety_filter.py`

**预计时间**: 1 小时讨论 + 实现 + 审查

---

### Step 1.3: TaskRewardShaper 类

**目标**: 创建独立的任务奖励塑形器类，基于 Task FSA + Robustness 提供多层次奖励

**状态**: ✓ 已完成设计讨论

---

#### 设计讨论总结

**核心思想**：
- 任务奖励 = Robustness 增量 + FSA 进度 + 任务完成 + 过滤器惩罚 + 时间惩罚
- 使用 Büchi 自动机（ba 模式）构建任务 FSA
- 预计算 BFS 距离（每个状态到接受状态的最短路径）

**奖励组件**（6 个部分）：

1. **Robustness 增量奖励**（稠密，主要引导信号）
   - 计算方式：`r_rho = w_rho * (ρ_t - ρ_{t-1})`
   - 只用当前状态和上一状态，不用完整轨迹
   - 语义：ρ 增加 = 更接近满足任务 = 正奖励
   - 权重：`w_rho = 0.1`

2. **FSA 进度奖励**（稀疏，里程碑）
   - 计算方式：`r_progress = w_progress * (value_t - value_{t+1})`
   - `value_t` = BFS 距离（到接受状态的步数，正数）
   - 语义：距离减少 = 前进 = 正奖励；距离增加 = 后退 = 负奖励
   - 例子：从距离 2 前进到距离 1 → `r = 1.0 * (2 - 1) = 1.0`
   - 权重：`w_progress = 1.0`

3. **任务完成奖励**（稀疏，最终目标）
   - 计算方式：`r_accept = w_accept * is_accepting`
   - 到达接受状态时触发
   - 权重：`w_accept = 10.0`

4. **陷阱状态惩罚**（稀疏，任务失败）
   - 计算方式：`r_trap = w_trap * (-10.0) * entered_trap`
   - 进入任务 FSA 的陷阱状态时触发
   - 陷阱状态定义：无法到达接受状态的非接受状态（与安全 FSA 定义一致）
   - **关键**：进入陷阱后**立即终止 episode**
   - 理由：任务已经不可能完成，后续奖励引导没有意义
   - 权重：`w_trap = 1.0`，惩罚值 `-10.0`（与完成奖励对称）

5. **过滤器惩罚**（稀疏，学习安全）
   - 计算方式：`r_filter = w_filter * (-0.5) * filtered`
   - 触发 SafetyFilter 时给予惩罚
   - 理由：让 RL 学会主动避免不安全行为，减少对过滤器的依赖
   - 权重：`w_filter = 1.0`

6. **时间惩罚**（稠密，鼓励效率）
   - 计算方式：`r_time = w_time * (-0.01)`
   - 每步固定惩罚，鼓励最短路径
   - 权重：`w_time = 1.0`

**总奖励公式**：
```python
# 正常情况
reward = (
    w_rho * (rho_t - rho_{t-1})           # Robustness 增量
    + w_progress * (value_t - value_{t+1}) # FSA 进度
    + w_accept * is_accepting              # 任务完成
    + w_filter * (-0.5) * filtered         # 过滤器惩罚
    + w_time * (-0.01)                     # 时间惩罚
)

# 进入陷阱状态
if entered_trap:
    reward = w_trap * (-10.0)
    terminated = True  # 立即终止 episode
```

---

#### Robustness 计算方式

**关键决策**：只用当前状态，不用历史轨迹

**理由**：
- 目标是提供**引导信号**，不是判断任务是否完成
- 只需要知道"往哪个方向走"，不需要完整轨迹
- 对于 reach-avoid 任务，当前状态的距离就是最好的信号

**实现**：
```python
# 当前状态的 robustness
rho_t = calculator.compute(state_t, trajectory=[state_t])

# 上一状态的 robustness
rho_{t-1} = calculator.compute(state_{t-1}, trajectory=[state_{t-1}])

# 增量奖励
r_rho = rho_t - rho_{t-1}
```

**对于时序算子的简化**：
- `F(goal)`：只看当前时刻 → `ρ(goal, state_t)` = 到目标的负距离
- `G(!wall)`：只看当前时刻 → `ρ(!wall, state_t)` = 到墙的距离
- 时序语义已经编码在 FSA 结构中，robustness 只提供稠密引导

---

#### 类设计方案

```python
# safe_rl_drone/safety/task_reward_shaper.py

class TaskRewardShaper:
    """
    任务奖励塑形器：基于 Task FSA + Robustness 计算多层次奖励
    """
    def __init__(self, task_formula, prop_manager, reward_weights):
        """
        Args:
            task_formula: 任务规范字符串（如 "F(goal)"）
            prop_manager: 原子命题管理器
            reward_weights: 奖励权重配置（dict）
        """
        self.task_formula = task_formula
        self.prop_manager = prop_manager
        self.weights = reward_weights
        
        # 构建 Task FSA（使用 ba 模式）
        self.fsa_monitor = FSAMonitor(task_formula, mode='ba')
        self.fsa_monitor.set_prop_manager(prop_manager)
        
        # 构建 Robustness 计算器
        parser = LTLParser()
        parser.parse(task_formula)
        self.rho_calculator = RobustnessCalculator(parser, prop_manager)
        
        # 预计算 FSA 状态价值（BFS 距离）
        self.state_values = self._compute_state_values()
        
        # 历史状态（用于计算增量）
        self.prev_state = None
        self.prev_rho = None
    
    def compute_reward(self, state, filtered=False):
        """
        计算任务奖励
        
        Args:
            state: 当前状态（np.ndarray）
            filtered: 是否触发了安全过滤器
            
        Returns:
            reward: 总奖励（float）
            info: 调试信息（dict，包含 terminated 标志）
        """
        # 1. Robustness 增量奖励
        rho_t = self.rho_calculator.compute(state, trajectory=[state])
        if self.prev_rho is not None:
            r_rho = self.weights['robustness'] * (rho_t - self.prev_rho)
        else:
            r_rho = 0.0
        
        # 2. FSA 进度奖励
        prev_fsa_state = self.fsa_monitor.current_state
        new_fsa_state, is_accepting, is_trap = self.fsa_monitor.step(state)
        
        # 检查是否进入陷阱状态
        if is_trap:
            # 任务失败，大惩罚 + 终止 episode
            r_trap = self.weights['trap'] * (-10.0)
            
            return r_trap, {
                'terminated': True,
                'reason': 'task_trap_state',
                'fsa_state': new_fsa_state,
                'is_trap': True,
                'r_trap': r_trap
            }
        
        prev_value = self.state_values.get(prev_fsa_state, 0)
        new_value = self.state_values.get(new_fsa_state, 0)
        r_progress = self.weights['progress'] * (prev_value - new_value)
        
        # 3. 任务完成奖励
        r_accept = self.weights['acceptance'] * (1.0 if is_accepting else 0.0)
        
        # 4. 过滤器惩罚
        r_filter = self.weights['filter'] * (-0.5 if filtered else 0.0)
        
        # 5. 时间惩罚
        r_time = self.weights['time'] * (-0.01)
        
        # 总奖励
        total_reward = r_rho + r_progress + r_accept + r_filter + r_time
        
        # 更新历史
        self.prev_state = state.copy()
        self.prev_rho = rho_t
        
        return total_reward, {
            'r_rho': r_rho,
            'r_progress': r_progress,
            'r_accept': r_accept,
            'r_filter': r_filter,
            'r_time': r_time,
            'rho': rho_t,
            'fsa_state': new_fsa_state,
            'is_accepting': is_accepting,
            'terminated': False
        }
    
    def _compute_state_values(self):
        """
        BFS 计算状态价值（到接受状态的距离）
        
        Returns:
            state_values: {state_id: -distance}
        """
        if self.fsa_monitor.fsa is None:
            return {}
        
        fsa = self.fsa_monitor.fsa
        accepting_states = fsa.accepting_states
        
        # 从接受状态反向 BFS
        values = {}
        queue = [(s, 0) for s in accepting_states]
        visited = set(accepting_states)
        
        while queue:
            state, dist = queue.pop(0)
            values[state] = dist  # 距离越近，价值越大
            
            # 找到所有能到达当前状态的前驱状态
            for trans in fsa.transitions:
                if trans.target == state and trans.source not in visited:
                    visited.add(trans.source)
                    queue.append((trans.source, dist + 1))
        
        return values
    
    def reset(self):
        """重置状态"""
        self.fsa_monitor.reset()
        self.prev_state = None
        self.prev_rho = None
```

---

#### 默认权重配置

```python
DEFAULT_REWARD_WEIGHTS = {
    'robustness': 0.1,   # Robustness 增量（稠密引导）
    'progress': 1.0,     # FSA 进度（稀疏里程碑）
    'acceptance': 10.0,  # 任务完成（最终目标）
    'trap': 1.0,         # 陷阱惩罚（任务失败，-10.0）
    'filter': 1.0,       # 过滤器惩罚（学习安全，-0.5）
    'time': 1.0          # 时间惩罚（鼓励效率，-0.01）
}
```

---

#### 讨论点

1. ✓ **Robustness 计算**：只用当前状态，不用历史轨迹
2. ✓ **奖励组件**：5 个部分，权重合理
3. ✓ **过滤器惩罚**：必须给，让 RL 学会主动安全
4. ✓ **时间惩罚**：每步 -0.01，鼓励最短路径
5. ✓ **进度奖励**：前进奖励，后退惩罚

**待确认**：
- ✓ BFS 距离计算是否正确？
- ✓ 权重是否需要调整？
- ✓ 陷阱状态处理：进入后立即终止 episode
- ✓ 陷阱状态定义：复用 `fsa.py` 中的代码（无法到达接受状态的非接受状态）
- 是否需要其他奖励组件？

---

#### 交付物

- `safe_rl_drone/safety/task_reward_shaper.py`
- 单元测试：`tests/test_task_reward_shaper.py`
- 配置文件更新：添加 `reward_weights` 字段

**预计时间**: 1.5 小时讨论（已完成）+ 2 小时实现 + 1 小时测试

---

### Step 1.4: 集成到 SafeEnvWrapper

**目标**: 修改现有的 SafeEnvWrapper，使用新的两个类

**设计方案**:

```python
# safe_rl_drone/wrappers/safe_env_wrapper.py

class SafeEnvWrapper(gym.Wrapper):
    def __init__(self, env, safety_formula, task_formula, config):
        super().__init__(env)
        
        # 创建命题评估器
        self.prop_evaluator = PropositionEvaluator(env)
        
        # 创建安全过滤器
        self.safety_filter = SafetyFilter(
            safety_formula,
            self.prop_evaluator,
            env.get_env_info()
        )
        
        # 创建任务奖励塑形器
        self.task_reward_shaper = TaskRewardShaper(
            task_formula,
            self.prop_evaluator,
            config['reward_weights']
        )
        
        # 统计信息
        self.stats = {
            'safety_interventions': 0,
            'task_progress_rewards': 0,
            'task_completions': 0
        }
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.safety_filter.reset()
        self.task_reward_shaper.reset()
        return obs, info
    
    def step(self, action):
        # 1. 安全过滤
        safe_action, filtered, filter_info = self.safety_filter.filter_action(
            self._get_current_state(),
            action
        )
        
        if filtered:
            self.stats['safety_interventions'] += 1
        
        # 2. 执行动作
        obs, base_reward, terminated, truncated, info = self.env.step(safe_action)
        
        # 3. 计算任务奖励（传入 filtered 标志）
        task_reward, task_info = self.task_reward_shaper.compute_reward(
            self._get_current_state(),
            filtered=filtered  # 告诉 TaskRewardShaper 是否触发了过滤器
        )
        
        # 4. 使用任务奖励（不使用环境的 base_reward）
        # 理由：我们的 6 个奖励组件已经完全覆盖了任务需求
        total_reward = task_reward
        
        # 5. 处理任务陷阱状态的终止
        if task_info.get('terminated', False):
            terminated = True  # 任务失败，终止 episode
        
        # 6. 更新 info
        info.update({
            'safety_filtered': filtered,
            'task_rho': task_info['rho'],
            'task_fsa_state': task_info['fsa_state'],
            'task_is_accepting': task_info['is_accepting'],
            'task_is_trap': task_info['is_trap'],
            'r_rho': task_info['r_rho'],
            'r_progress': task_info['r_progress'],
            'r_accept': task_info['r_accept'],
            'r_trap': task_info.get('r_trap', 0.0),
            'r_filter': task_info['r_filter'],
            'r_time': task_info['r_time']
        })
        
        return obs, total_reward, terminated, truncated, info
    
    def _get_current_state(self):
        """获取当前状态（用于命题评估）"""
        # 从环境中提取状态信息
        pass
```

**讨论点**:
1. 如何平滑过渡（不破坏现有功能）？
2. `_get_current_state` 如何实现？
3. 是否需要保留旧的接口（向后兼容）？
4. 统计信息是否足够？

**交付物**:
- 修改后的 `safe_rl_drone/wrappers/safe_env_wrapper.py`
- 更新测试：`tests/test_safe_wrapper.py`

**预计时间**: 30 分钟讨论 + 实现 + 审查

---

### Step 1.5: 测试双规范架构

**目标**: 在当前网格世界验证新架构

**测试场景**:

1. **简单任务**:
   - Safety: `G(!wall)`
   - Task: `F(goal)`
   - 验证：零违规，成功率 > 90%

2. **顺序任务**:
   - Safety: `G(!wall)`
   - Task: `F(wp1) & F(wp2) & F(goal)`
   - 验证：按顺序到达，零违规

3. **条件任务**（可选）:
   - Safety: `G(!wall)`
   - Task: `F(goal1) | F(goal2)`
   - 验证：到达任一目标

**测试代码**:

```python
# tests/test_dual_specification.py

def test_simple_task():
    """测试简单任务：F(goal)"""
    # 创建环境
    # 训练
    # 验证零违规
    # 验证成功率
    pass

def test_sequential_task():
    """测试顺序任务：F(wp1) & F(wp2) & F(goal)"""
    # 创建环境（带 waypoints）
    # 训练
    # 验证顺序
    pass

def test_reward_components():
    """测试奖励组件"""
    # 验证 robustness 奖励
    # 验证 progress 奖励
    # 验证 acceptance 奖励
    pass
```

**讨论点**:
1. 测试场景是否足够？
2. 成功标准是否合理？
3. 如何可视化训练过程？

**交付物**:
- `tests/test_dual_specification.py`
- 测试报告（Markdown）

**预计时间**: 1 小时讨论 + 实现测试 + 运行验证

---

## Phase 2: 连续环境迁移（预计 5-6 周）

### Step 2.1: 场景配置系统

**目标**: 设计 YAML 场景配置格式

**设计方案**:

```yaml
# scenarios/reach_avoid_2d.yaml

name: "reach_avoid_2d"
description: "2D navigation with obstacles"

environment:
  type: "pybullet_2d"
  bounds: [-5.0, 5.0, -5.0, 5.0]  # [xmin, xmax, ymin, ymax]
  physics:
    gravity: [0, 0, -9.81]
    time_step: 0.01
    
initial_state:
  position: [0.0, 0.0]
  velocity: [0.0, 0.0]

objects:
  goal:
    type: "target"
    shape: "cylinder"
    position: [4.0, 4.0, 0.0]
    radius: 0.5
    height: 0.5
    color: [0, 1, 0, 0.5]
    
  obstacle1:
    type: "obstacle"
    shape: "cylinder"
    position: [2.0, 2.0, 0.0]
    radius: 0.5
    height: 2.0
    color: [1, 0, 0, 0.7]
    dynamic: false

propositions:
  goal:
    reach_threshold: 0.5
  obstacle:
    safety_threshold: 0.8
  boundary:
    margin: 0.5
```

**讨论点**:
1. 配置格式是否合理？
2. 是否需要其他字段？
3. 如何支持动态障碍物？

**交付物**:
- 场景配置示例
- `ScenarioLoader` 类设计

**预计时间**: 30 分钟讨论

---

### Step 2.2-2.7: 后续步骤

（详细设计在需要时展开）

---

## 总体时间估算

- **Phase 1**: 2-3 周（7 个小步骤）
- **Phase 2**: 5-6 周（环境 + 适配）
- **Phase 3**: 1 周（动态）
- **总计**: 8-10 周

---

## 当前状态

**准备开始**: Step 1.1 - 双规范配置系统

**下一步**: 讨论配置文件格式，用户批准后开始实现
