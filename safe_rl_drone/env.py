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
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_map, view_size: int = 5):
        super(GridWorldEnv, self).__init__()

        self.grid_map = np.array(grid_map)
        self.rows, self.cols = self.grid_map.shape
        self.view_size = view_size  # 局部视野大小 (奇数)
        self.view_radius = view_size // 2

        # 在地图中找到特殊点的位置
        start_pos = np.where(self.grid_map == 'S')
        target_pos = np.where(self.grid_map == 'T')
        
        self.agent_start_pos = np.array([start_pos[0][0], start_pos[1][0]])
        self.target_pos = np.array([target_pos[0][0], target_pos[1][0]])
        self.agent_pos = np.copy(self.agent_start_pos)

        # 将字符地图转换为数字地图, 方便处理
        # 0=空地, 1=墙, 2=目标
        self.numeric_map = np.zeros_like(self.grid_map, dtype=int)
        self.numeric_map[self.grid_map == 'W'] = WALL
        self.numeric_map[self.grid_map == 'T'] = TARGET
        
        # 提取墙的位置集合（用于 LTL 安全层）
        self.wall_positions = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.numeric_map[r, c] == WALL:
                    self.wall_positions.add((r, c))
        
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

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        - Agent回到起始位置。
        - 返回初始观测。
        """
        super().reset(seed=seed)
        self.agent_pos = np.copy(self.agent_start_pos)
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """
        环境根据Agent的动作前进一步。
        1. 计算Agent的下一个位置。
        2. 检查是否撞墙或越界。
        3. 计算奖励 (Reward)。
        4. 检查是否到达终点 (Terminated)。
        5. 返回 (观测, 奖励, 是否终止, 是否截断, 额外信息)。
        """
        # 1. 计算潜在的下一个位置
        move = ACTION_MAP[action]
        next_pos = self.agent_pos + move

        # 2. 检查碰撞和越界
        hit_wall = False
        hit_boundary = False
        if not (0 <= next_pos[0] < self.rows and 0 <= next_pos[1] < self.cols):
            # 越界了, 位置不变
            next_pos = self.agent_pos
            hit_boundary = True
        elif self.numeric_map[next_pos[0], next_pos[1]] == WALL:
            # 撞墙了, 位置不变
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
            'boundary': hit_boundary
        }

        return observation, reward, terminated, truncated, info

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
                elif self.numeric_map[r, c] == TARGET:
                     self.ax.add_patch(patches.Rectangle((c, self.rows - 1 - r), 1, 1, color='green'))

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
