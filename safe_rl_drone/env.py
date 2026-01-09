import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端支持 GUI 显示
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义动作：我们有4个可能的动作
# 0: 向上, 1: 向下, 2: 向左, 3: 向右, 4: 原地不动
ACTION_MAP = {
    0: np.array([-1, 0]),  # 向上 (行减1)
    1: np.array([1, 0]),   # 向下 (行加1)
    2: np.array([0, -1]),  # 向左 (列减1)
    3: np.array([0, 1]),   # 向右 (列加1)
    4: np.array([0, 0]),   # 原地不动
}

# 定义地图元素
WALL = 1
AGENT = 2
TARGET = 3
EMPTY = 0

class GridWorldEnv(gym.Env):
    """
    一个自定义的2D网格世界环境, 遵循 Gymnasium API 接口。

    在这个环境里:
    - 智能体(Agent)的目标是从一个起始点 'S' 移动到一个目标点 'T'。
    - 地图中包含无法通过的墙 'W'。
    - 智能体每次移动都会收到小的负奖励, 到达目标会收到大的正奖励, 撞墙会收到中的负奖励。
    
    观测空间：局部视野 (view_size × view_size) + 目标方向 (2)
    - 局部视野编码: 0=空地, 1=墙, 2=目标, 3=边界外
    - 目标方向: 归一化的 [dy, dx] 向量
    
    Goal-Conditioned RL 支持:
    - random_goal=True: 每次 reset 随机采样目标位置
    - random_start=True: 每次 reset 随机采样起始位置
    - 可以通过 reset(options={'goal': (r,c)}) 指定目标
    
    概念上，这个策略等价于一个 Reach(goal) 技能。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_map, view_size: int = 5, 
                 random_goal: bool = False, random_start: bool = False,
                 dynamic_obstacles: bool = False, obstacle_change_prob: float = 0.02):
        super(GridWorldEnv, self).__init__()

        self.grid_map = np.array(grid_map)
        self.rows, self.cols = self.grid_map.shape
        self.view_size = view_size  # 局部视野大小 (奇数)
        self.view_radius = view_size // 2
        self.random_goal = random_goal
        self.random_start = random_start
        
        # 动态障碍物设置
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_change_prob = obstacle_change_prob

        # 在地图中找到特殊点的位置
        start_pos = np.where(self.grid_map == 'S')
        target_pos = np.where(self.grid_map == 'T')
        
        self.agent_start_pos = np.array([start_pos[0][0], start_pos[1][0]])
        self.default_target_pos = np.array([target_pos[0][0], target_pos[1][0]])
        self.target_pos = np.copy(self.default_target_pos)
        self.agent_pos = np.copy(self.agent_start_pos)

        # 将字符地图转换为数字地图, 方便处理
        # 0=空地, 1=墙, 2=目标
        self.numeric_map = np.zeros_like(self.grid_map, dtype=int)
        self.numeric_map[self.grid_map == 'W'] = WALL
        self.numeric_map[self.grid_map == 'T'] = TARGET
        
        # 保存原始墙位置（用于动态障碍物）
        self.original_wall_positions = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.numeric_map[r, c] == WALL:
                    self.original_wall_positions.add((r, c))
        
        # 当前活跃的墙位置（可能动态变化）
        self.wall_positions = self.original_wall_positions.copy()
        
        # 提取所有空格位置（用于随机采样目标/起点）
        self.empty_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.numeric_map[r, c] == EMPTY:
                    self.empty_cells.append((r, c))
        # 起点和终点也是可用位置
        self.empty_cells.append(tuple(self.agent_start_pos))
        self.empty_cells.append(tuple(self.default_target_pos))
        
        # 定义动作空间 (Action Space): 离散的5个动作
        self.action_space = gym.spaces.Discrete(5)

        # 定义观测空间 (Observation Space): 
        # 局部视野 (view_size × view_size) 展平 + 目标方向 (2)
        obs_dim = view_size * view_size + 2
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=3.0,  # 局部视野最大值为3(边界外)，方向在[-1,1]
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 用于渲染
        self.fig = None
        self.ax = None

    def _get_local_view(self) -> np.ndarray:
        """
        获取以智能体为中心的局部视野
        
        Returns:
            view_size × view_size 的数组
            编码: 0=空地, 1=墙, 2=目标, 3=边界外
        """
        view = np.zeros((self.view_size, self.view_size), dtype=np.float32)
        
        for dr in range(-self.view_radius, self.view_radius + 1):
            for dc in range(-self.view_radius, self.view_radius + 1):
                r = self.agent_pos[0] + dr
                c = self.agent_pos[1] + dc
                
                view_r = dr + self.view_radius
                view_c = dc + self.view_radius
                
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    view[view_r, view_c] = 3  # 边界外
                else:
                    view[view_r, view_c] = self.numeric_map[r, c]
        
        return view

    def _get_goal_direction(self) -> np.ndarray:
        """
        获取目标方向（归一化向量）
        
        Returns:
            [dy, dx] 归一化方向向量，如果在目标位置则返回 [0, 0]
        """
        diff = self.target_pos - self.agent_pos
        dist = np.linalg.norm(diff)
        
        if dist < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        return (diff / dist).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        获取观测：局部视野 + 目标方向
        
        Returns:
            展平的观测向量 [local_view.flatten(), goal_direction]
        """
        local_view = self._get_local_view().flatten()
        goal_dir = self._get_goal_direction()
        return np.concatenate([local_view, goal_dir])

    def _sample_random_position(self, exclude=None):
        """
        随机采样一个空格位置
        
        Args:
            exclude: 要排除的位置 (tuple 或 list of tuples)
        """
        available = self.empty_cells.copy()
        if exclude is not None:
            if isinstance(exclude, tuple):
                exclude = [exclude]
            for pos in exclude:
                if pos in available:
                    available.remove(pos)
        
        if not available:
            return self.empty_cells[0]  # fallback
        
        idx = self.np_random.integers(0, len(available))
        return available[idx]

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        
        支持 Goal-Conditioned RL:
        - options={'goal': (r, c)}: 指定目标位置
        - options={'start': (r, c)}: 指定起始位置
        - random_goal=True: 随机采样目标
        - random_start=True: 随机采样起点
        """
        super().reset(seed=seed)
        options = options or {}
        
        # 0. 重置墙状态到原始状态
        self.wall_positions = self.original_wall_positions.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.original_wall_positions:
                    self.numeric_map[r, c] = WALL
                elif self.grid_map[r, c] != 'T':
                    self.numeric_map[r, c] = EMPTY
        
        # 1. 设置目标位置
        if 'goal' in options:
            self.target_pos = np.array(options['goal'])
        elif self.random_goal:
            goal = self._sample_random_position()
            self.target_pos = np.array(goal)
        else:
            self.target_pos = np.copy(self.default_target_pos)
        
        # 2. 设置起始位置
        if 'start' in options:
            self.agent_pos = np.array(options['start'])
        elif self.random_start:
            start = self._sample_random_position(exclude=tuple(self.target_pos))
            self.agent_pos = np.array(start)
        else:
            self.agent_pos = np.copy(self.agent_start_pos)
        
        observation = self._get_obs()
        info = {
            'goal': tuple(self.target_pos),
            'start': tuple(self.agent_pos),
            'dynamic_obstacles': self.dynamic_obstacles
        }
        return observation, info

    def step(self, action):
        """
        环境根据Agent的动作前进一步。
        1. 动态更新障碍物（如果启用）
        2. 计算Agent的下一个位置。
        3. 检查是否撞墙或越界。
        4. 计算奖励 (Reward)。
        5. 检查是否到达终点 (Terminated)。
        6. 返回 (观测, 奖励, 是否终止, 是否截断, 额外信息)。
        """
        # 0. 动态更新障碍物
        if self.dynamic_obstacles:
            self._update_dynamic_obstacles()
        
        # 1. 计算潜在的下一个位置
        move = ACTION_MAP[action]
        next_pos = self.agent_pos + move

        # 2. 检查碰撞和越界（使用当前墙位置）
        hit_wall = False
        hit_boundary = False
        if not (0 <= next_pos[0] < self.rows and 0 <= next_pos[1] < self.cols):
            # 越界了, 位置不变
            next_pos = self.agent_pos
            hit_boundary = True
        elif tuple(next_pos) in self.wall_positions:
            # 撞墙了, 位置不变（使用动态墙位置检查）
            next_pos = self.agent_pos
            hit_wall = True
        
        self.agent_pos = next_pos
        
        # 3. 检查是否到达目标
        terminated = np.array_equal(self.agent_pos, self.target_pos)

        # 4. 计算奖励
        if terminated:
            reward = 10.0  # 到达终点, 获得大奖励
        elif hit_wall or hit_boundary:
            reward = -1.0  # 撞墙或越界, 获得惩罚
        else:
            reward = -0.1 # 每走一步, 给予小的负奖励, 鼓励尽快到达
            
        observation = self._get_obs()
        truncated = False
        info = {
            'collision': hit_wall,
            'boundary': hit_boundary,
            'dynamic_obstacles': self.dynamic_obstacles,
            'active_walls': len(self.wall_positions)
        }

        return observation, reward, terminated, truncated, info
    
    def _update_dynamic_obstacles(self):
        """
        动态更新障碍物状态
        
        策略：在原始墙位置中随机切换墙的存在/消失状态
        这样保证基础路径始终存在（因为只在原有墙位置变化）
        """
        if not self.original_wall_positions:
            return
        
        # 遍历所有原始墙位置，随机切换状态
        for wall_pos in self.original_wall_positions:
            if self.np_random.random() < self.obstacle_change_prob:
                if wall_pos in self.wall_positions:
                    # 墙消失
                    self.wall_positions.discard(wall_pos)
                    self.numeric_map[wall_pos[0], wall_pos[1]] = EMPTY
                else:
                    # 墙出现（只有不是 agent 当前位置和目标位置才能出现）
                    if wall_pos != tuple(self.agent_pos) and wall_pos != tuple(self.target_pos):
                        self.wall_positions.add(wall_pos)
                        self.numeric_map[wall_pos[0], wall_pos[1]] = WALL

    def render(self):
        """
        可视化当前环境的状态。
        """
        if self.fig is None:
            # 首次渲染时, 初始化绘图窗口
            plt.ion() # 开启交互模式
            self.fig, self.ax = plt.subplots()
        
        self.ax.clear() # 清除上一帧

        # 绘制网格
        for r in range(self.rows + 1):
            self.ax.axhline(r, color='black', lw=0.5)
        for c in range(self.cols + 1):
            self.ax.axvline(c, color='black', lw=0.5)

        # 填充颜色
        for r in range(self.rows):
            for c in range(self.cols):
                if self.numeric_map[r, c] == WALL:
                    self.ax.add_patch(patches.Rectangle((c, self.rows - 1 - r), 1, 1, color='gray'))
        
        # 绘制当前目标（动态目标支持）
        goal_r, goal_c = self.target_pos
        self.ax.add_patch(patches.Rectangle((goal_c, self.rows - 1 - goal_r), 1, 1, color='green'))

        # 绘制Agent
        agent_patch = patches.Circle((self.agent_pos[1] + 0.5, self.rows - 1 - self.agent_pos[0] + 0.5), 0.3, color='blue')
        self.ax.add_patch(agent_patch)

        # 设置坐标轴
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis() # 将y轴原点放在左上角, 与numpy数组索引匹配

        plt.title("Grid World")
        plt.draw()
        plt.pause(0.1) # 暂停一小段时间, 以便我们能看到图像

    def close(self):
        """
        关闭并清理渲染窗口。
        """
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def get_env_info(self) -> dict:
        """
        获取环境信息（用于 LTL 安全层）
        
        Returns:
            包含墙位置、目标位置、网格大小等信息的字典
        """
        return {
            'wall_positions': self.wall_positions,
            'goal_position': tuple(self.target_pos),
            'grid_size': (self.rows, self.cols),
            'agent_start': tuple(self.agent_start_pos)
        }
    
    def get_action_map(self) -> dict:
        """获取动作映射"""
        return ACTION_MAP
