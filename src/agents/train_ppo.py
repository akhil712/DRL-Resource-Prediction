# src/agents/train_ppo.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from agents.dppo_env import VNFPlacementEnv

def make_env(sim, sfc_list):
    def _init():
        return VNFPlacementEnv(sim, sfc_list)
    return _init

def train_ppo(sim, sfc_list, save_path="models/ppo"):
    os.makedirs(save_path, exist_ok=True)
    num_envs = 4
    env_fns = [make_env(sim, sfc_list) for _ in range(num_envs)]
    vec = DummyVecEnv(env_fns)
    model = PPO("MlpPolicy", vec, verbose=1, batch_size=32, n_steps=128, learning_rate=1e-3)
    model.learn(total_timesteps=2000)
    model.save(os.path.join(save_path, "ppo_placement"))
    return model
