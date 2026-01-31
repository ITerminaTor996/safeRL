# 已知问题和待改进项

## 已修复

### 问题 1：边条件 Robustness 计算包含自环边
- **状态**：✅ 已修复
- **描述**：原来的 `compute_edge_based_robustness` 计算所有出边的 max，包括自环边。但自环边（如 `!goal`）不能推进 FSA 状态，不应该被考虑。
- **影响**：当 agent 远离目标时，`ρ(!goal) > ρ(goal)`，导致 robustness 为正，给出错误的引导信号。
- **修复**：排除自环边（`source == target`），只对能推进 FSA 状态的边计算 robustness。

---

## 待改进

### 问题 2：状态价值权重未充分利用
- **状态**：⏳ 待改进
- **描述**：论文公式 (11) 中，边的 robustness 可能需要考虑目标状态的价值作为权重。目前我们只在 FSA 状态转移时使用 `state_values`，没有在边条件 robustness 计算中使用。
- **影响**：可能影响多步任务（如顺序任务）的学习效率。
- **参考**：Xiao Li et al., Science Robotics 2019, Equation (11)

### 问题 3：通向陷阱的边被计算
- **状态**：✅ 已修复
- **描述**：`(!danger) U safe` 的 FSA 有边通向陷阱状态。原来取 max 时包含了这些边，导致在 danger 区域时 robustness 为正，给出错误引导。
- **影响**：agent 被鼓励进入 danger 区域。
- **修复**：有效边的定义改为：(1) 非自环 (2) 目标不是陷阱状态。

### 问题 4：Robustness 尺度不一致
- **状态**：⏳ 待改进
- **描述**：不同命题的 robustness 尺度可能不一样（取决于地图大小和目标位置）。
- **影响**：AND/OR 的 min/max 计算可能被尺度较大的命题主导。
- **建议**：考虑对 robustness 进行归一化。

---

## 设计决策

### Safety vs Task 分离
- **Safety 规范**（如 `G(!wall)`）→ SafetyFilter（硬约束，形式化保证）
- **Task 规范**（如 `F(goal)`）→ TaskRewardShaper（软奖励，RL 学习）
- **原因**：G 算子适合硬约束，F/U 算子适合软奖励引导

### 边条件 Robustness vs 轨迹 Robustness
- 使用边条件 robustness（只看当前状态）而不是轨迹 robustness
- **原因**：
  1. FSA 边条件是纯布尔表达式，时序语义已编码在状态转移中
  2. 避免"刷分"问题（来回走增加轨迹 robustness）
  3. 计算简单，不需要维护完整轨迹
