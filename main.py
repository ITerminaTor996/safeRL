"""
Safe RL Drone - 形式化方法保证安全的强化学习

使用方法:
    python main.py                    # 使用默认配置
    python main.py --config my.yaml   # 使用自定义配置
    python main.py --safe false       # 命令行覆盖配置
"""

import os
import time
import yaml
import argparse
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from safe_rl_drone.env import GridWorldEnv
from safe_rl_drone.wrappers import SafeEnvWrapper
from utils import load_map_from_file


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_args_to_config(config: dict, args) -> dict:
    """将命令行参数合并到配置中（命令行优先）"""
    if args.safe is not None:
        config['safety']['enabled'] = args.safe.lower() == 'true'
    if args.map:
        config['environment']['map_path'] = args.map
    if args.timesteps:
        config['training']['total_timesteps'] = args.timesteps
    if args.ltl:
        config['safety']['formula'] = args.ltl
    if args.load:
        config['model']['load_path'] = args.load
    if args.skip_train:
        config['training']['enabled'] = False
    return config


class SafetyStatsCallback(BaseCallback):
    """统计训练过程中的安全数据"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_interventions = 0
        self.total_collisions = 0
        self.total_boundary = 0
        self.total_steps = 0
        
    def _on_step(self) -> bool:
        self.total_steps += 1
        infos = self.locals.get('infos', [])
        for info in infos:
            if info.get('safety_intervention', False):
                self.total_interventions += 1
            if info.get('collision', False):
                self.total_collisions += 1
            if info.get('boundary', False):
                self.total_boundary += 1
        return True
    
    def get_stats(self) -> dict:
        intervention_rate = self.total_interventions / max(self.total_steps, 1)
        return {
            'interventions': self.total_interventions,
            'collisions': self.total_collisions,
            'boundary': self.total_boundary,
            'total_steps': self.total_steps,
            'intervention_rate': intervention_rate
        }


def print_banner(config: dict):
    """打印启动信息"""
    print(f"\n{'='*60}")
    print("Safe RL Drone - 形式化方法 + 强化学习")
    print("Goal-Conditioned RL: Reach Skill with Safety Guarantee")
    print(f"{'='*60}")
    print(f"地图: {config['environment']['map_path']}")
    print(f"安全保护: {'✓ 开启' if config['safety']['enabled'] else '✗ 关闭'}")
    if config['safety']['enabled']:
        print(f"LTL 安全公式: {config['safety']['formula']}")
    print(f"随机目标: {'✓' if config['environment'].get('random_goal', False) else '✗'}")
    print(f"随机起点: {'✓' if config['environment'].get('random_start', False) else '✗'}")
    dynamic = config['environment'].get('dynamic_obstacles', False)
    print(f"动态障碍物: {'✓ 开启' if dynamic else '✗ 关闭'}")
    if dynamic:
        prob = config['environment'].get('obstacle_change_prob', 0.02)
        print(f"  变化概率: {prob}")
    print(f"Robustness 奖励: {'✓' if config['task'].get('use_robustness_reward', False) else '✗'}")
    print(f"训练: {'✓ 启用' if config['training']['enabled'] else '✗ 跳过'}")
    print(f"{'='*60}\n")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Safe RL Drone")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--safe', type=str, default=None,
                        help='是否启用安全保护 (true/false)')
    parser.add_argument('--map', type=str, default=None,
                        help='地图文件路径')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='训练步数')
    parser.add_argument('--ltl', type=str, default=None,
                        help='LTL 安全公式')
    parser.add_argument('--load', type=str, default=None,
                        help='加载已有模型路径')
    parser.add_argument('--skip-train', action='store_true', dest='skip_train',
                        help='跳过训练，直接测试')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config = merge_args_to_config(config, args)
    
    print_banner(config)
    
    # ==============================================================
    # 1. 加载地图
    # ==============================================================
    print("--> Step 1: Loading map...")
    grid_map = load_map_from_file(config['environment']['map_path'])
    
    # ==============================================================
    # 2. 创建环境
    # ==============================================================
    print("--> Step 2: Creating environment...")
    view_size = config['environment'].get('view_size', 5)
    random_goal = config['environment'].get('random_goal', False)
    random_start = config['environment'].get('random_start', False)
    dynamic_obstacles = config['environment'].get('dynamic_obstacles', False)
    obstacle_change_prob = config['environment'].get('obstacle_change_prob', 0.02)
    
    base_env = GridWorldEnv(
        grid_map=grid_map, 
        view_size=view_size,
        random_goal=random_goal,
        random_start=random_start,
        dynamic_obstacles=dynamic_obstacles,
        obstacle_change_prob=obstacle_change_prob
    )
    
    print(f"  可用目标位置数: {len(base_env.empty_cells)}")
    if dynamic_obstacles:
        print(f"  动态障碍物: ✓ 开启 (变化概率: {obstacle_change_prob})")
    
    if config['safety']['enabled']:
        # 使用新的安全包装器
        env = SafeEnvWrapper(
            base_env,
            safety_formula=config['safety']['formula'],
            unsafe_penalty=config['safety'].get('unsafe_penalty', -1.0),
            use_robustness_reward=config['task'].get('use_robustness_reward', False),
            robustness_weight=config['task'].get('robustness_weight', 0.1),
            config=config
        )
    else:
        env = base_env
    
    # 添加训练时的超时限制
    max_episode_steps = config['environment'].get('max_episode_steps', 500)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    print(f"  每回合最大步数: {max_episode_steps}")
    
    # ==============================================================
    # 3. 加载或创建智能体
    # ==============================================================
    load_path = config['model'].get('load_path', '')
    
    if load_path and os.path.exists(f"{load_path}.zip"):
        print(f"--> Step 3: Loading model from {load_path}...")
        model = PPO.load(load_path, env=env)
        print(f"模型已加载: {load_path}")
        callback = None
    else:
        print("--> Step 3: Creating PPO agent...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=config['training']['verbose'],
            device=config['training']['device']
        )
        callback = SafetyStatsCallback()
        if load_path:
            print(f"  (模型文件 {load_path}.zip 不存在，将创建新模型)")
    
    # ==============================================================
    # 4. 训练
    # ==============================================================
    if config['training']['enabled']:
        print(f"--> Step 4: Training for {config['training']['total_timesteps']} timesteps...")
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback
        )
        
        # 打印训练统计
        if callback:
            stats = callback.get_stats()
            print(f"\n{'='*60}")
            print("训练安全统计")
            print(f"{'='*60}")
            print(f"总步数: {stats['total_steps']}")
            print(f"撞墙次数: {stats['collisions']}")
            print(f"越界次数: {stats['boundary']}")
            if config['safety']['enabled']:
                print(f"安全干预次数: {stats['interventions']}")
                print(f"干预率: {stats['intervention_rate']:.2%}")
            print(f"{'='*60}")
        
        # 保存模型
        if config['model']['auto_save']:
            os.makedirs(config['model']['save_path'], exist_ok=True)
            model_name = "ppo_safe" if config['safety']['enabled'] else "ppo_unsafe"
            model_path = os.path.join(config['model']['save_path'], model_name)
            model.save(model_path)
            print(f"模型已保存: {model_path}.zip")
    else:
        print("--> Step 4: Skipping training...")
    
    # ==============================================================
    # 5. 测试（验证目标泛化性）
    # ==============================================================
    print(f"\n--> Step 5: Testing for {config['testing']['episodes']} episodes...")
    print("  验证目标泛化性：测试时使用随机目标")
    
    test_interventions = 0
    test_collisions = 0
    test_boundary = 0
    total_robustness = 0.0
    max_steps = config['environment'].get('max_episode_steps', 500)
    successes = 0
    
    # 获取测试目标列表
    test_goals = config['testing'].get('test_goals', [])
    
    for episode in range(config['testing']['episodes']):
        # 设置测试目标
        if test_goals and episode < len(test_goals):
            goal = tuple(test_goals[episode])
            obs, info = env.reset(options={'goal': goal})
        else:
            # 随机目标
            obs, info = env.reset()
            goal = info.get('goal', None)
        
        start = info.get('start', None)
        print(f"\n  Episode {episode + 1}: 起点={start}, 目标={goal}")
        
        terminated = False
        truncated = False
        steps = 0
        episode_robustness = []
        
        while not terminated and not truncated and steps < max_steps:
            action, _ = model.predict(obs, deterministic=config['testing']['deterministic'])
            obs, reward, terminated, truncated, info = env.step(action.item())
            steps += 1
            
            if info.get('safety_intervention', False):
                test_interventions += 1
            if info.get('collision', False):
                test_collisions += 1
            if info.get('boundary', False):
                test_boundary += 1
            
            episode_robustness.append(info.get('robustness', 0.0))
            
            if config['environment']['render']:
                env.render()
                time.sleep(0.1)
        
        avg_rho = np.mean(episode_robustness) if episode_robustness else 0.0
        total_robustness += avg_rho
        
        if terminated:
            successes += 1
            status = "✓ 到达目标"
        else:
            status = "✗ 超时"
        
        print(f"    结果: {steps} 步, avg ρ={avg_rho:.2f} ({status})")
    
    success_rate = successes / config['testing']['episodes'] * 100
    
    print(f"\n{'='*60}")
    print("测试结果（目标泛化性验证）")
    print(f"{'='*60}")
    print(f"成功率: {successes}/{config['testing']['episodes']} ({success_rate:.1f}%)")
    if config['safety']['enabled']:
        print(f"安全干预次数: {test_interventions}")
    print(f"撞墙次数: {test_collisions}")
    print(f"越界次数: {test_boundary}")
    print(f"平均 Robustness: {total_robustness / config['testing']['episodes']:.2f}")
    print(f"{'='*60}")
    
    env.close()


if __name__ == '__main__':
    main()
