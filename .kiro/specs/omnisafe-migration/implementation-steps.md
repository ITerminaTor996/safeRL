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

**目标**: 创建独立的任务奖励塑形器类

**设计方案**:

```python
# safe_rl_drone/safety/task_reward_shaper.py

class TaskRewardShaper:
    """
    任务奖励塑形器：基于 Task FSA + Robustness 计算奖励
    """
    def __init__(self, task_formula, prop_evaluator, reward_weights):
        """
        Args:
            task_formula: 任务规范字符串（如 "F(goal)"）
            prop_evaluator: 命题评估器
            reward_weights: 奖励权重配置（dict）
        """
        self.formula = parse_formula(task_formula)
        self.automaton = build_automaton(task_formula)
        self.prop_evaluator = prop_evaluator
        self.current_state = self.automaton.initial_state
        self.weights = reward_weights
        
        # 预计算 FSA 状态价值（到接受状态的距离）
        self.state_values = self._compute_state_values()
    
    def compute_reward(self, state):
        """
        计算任务奖励
        
        Args:
            state: 当前状态
            
        Returns:
            reward: 总奖励（float）
            info: 调试信息（dict）
        """
        # 1. 计算 Robustness（稠密）
        rho = compute_robustness(self.formula, state, self.prop_evaluator)
        
        # 2. 评估命题真值
        props = self.prop_evaluator.evaluate_all(state)
        
        # 3. 检查 FSA 状态转移
        next_state = self.automaton.step(self.current_state, props)
        
        # 4. 计算状态跳转奖励（稀疏）
        progress_reward = 0.0
        if next_state != self.current_state:
            old_value = self.state_values.get(self.current_state, 0)
            new_value = self.state_values.get(next_state, 0)
            if new_value > old_value:
                progress_reward = 1.0
        
        # 5. 检查接受状态（稀疏）
        acceptance_reward = 0.0
        if self.automaton.is_accepting(next_state):
            acceptance_reward = 1.0
        
        # 6. 更新状态
        self.current_state = next_state
        
        # 7. 组合奖励
        total_reward = (
            self.weights['robustness'] * rho +
            self.weights['progress'] * progress_reward +
            self.weights['acceptance'] * acceptance_reward
        )
        
        return total_reward, {
            'rho': rho,
            'progress': progress_reward,
            'acceptance': acceptance_reward,
            'fsa_state': self.current_state
        }
    
    def _compute_state_values(self):
        """BFS 计算状态价值"""
        # 从接受状态反向 BFS
        values = {}
        queue = [(s, 0) for s in self.automaton.accepting_states]
        visited = set(self.automaton.accepting_states)
        
        while queue:
            state, dist = queue.pop(0)
            values[state] = -dist  # 距离越近，价值越大
            
            for prev_state in self.automaton.predecessors(state):
                if prev_state not in visited:
                    visited.add(prev_state)
                    queue.append((prev_state, dist + 1))
        
        return values
    
    def reset(self):
        """重置 FSA 状态"""
        self.current_state = self.automaton.initial_state
```

**讨论点**:
1. 类接口是否合理？
2. 奖励权重如何设置（默认值）？
3. `_compute_state_values` 的实现是否正确？
4. 如何处理没有接受状态的情况？

**交付物**:
- `safe_rl_drone/safety/task_reward_shaper.py`
- 单元测试：`tests/test_task_reward_shaper.py`

**预计时间**: 1 小时讨论 + 实现 + 审查

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
        
        # 3. 计算任务奖励
        task_reward, task_info = self.task_reward_shaper.compute_reward(
            self._get_current_state()
        )
        
        # 4. 组合奖励
        total_reward = base_reward + task_reward
        
        # 5. 更新 info
        info.update({
            'safety_filtered': filtered,
            'task_rho': task_info['rho'],
            'task_progress': task_info['progress'],
            'task_acceptance': task_info['acceptance']
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
