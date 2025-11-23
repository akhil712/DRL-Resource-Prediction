# src/run_sim.py
import os
import random
import json
import numpy as np
import torch

from leaf_model.network_sim import SimpleNetworkSim
from ml.fedbi_gru import BiGRUModel
from ml.trainer_fed import train_local_one_step, state_dict_to_cpu, federated_average
from agents.train_ppo import train_ppo
from agents.dppo_env import VNFPlacementEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def make_synthetic_vnfs(sim, num_sfc=2, vnfs_per_sfc=2):
    sfc_list = []
    for i in range(num_sfc):
        for j in range(vnfs_per_sfc):
            node = random.choice(list(sim.nodes.keys()))
            cpu = random.uniform(1.0, 4.0)
            mem = random.uniform(1.0, 8.0)
            bw = random.uniform(1.0, 10.0)
            sim.add_vnf(i, j, node, cpu, mem, bw)
            sfc_list.append((i, j))
    return sfc_list

def simple_federated_training(sim, sfc_list, seq_len=5):
    device = torch.device("cpu")
    models = {}
    optimizers = {}
    loss_fn = torch.nn.MSELoss()
    for key in sfc_list:
        m = BiGRUModel(input_dim=3, hidden_dim=20, num_layers=2, out_dim=3).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=0.005)
        models[key] = m
        optimizers[key] = opt
    histories = {}
    for key in sfc_list:
        arr = np.random.rand(seq_len, 3).astype(np.float32) * 2.0
        histories[key] = arr
    for round in range(10):
        state_dicts = []
        for key, m in models.items():
            x = torch.tensor(histories[key]).unsqueeze(0)
            y = x[:, -1, :] * 1.02
            loss = train_local_one_step(m, optimizers[key], loss_fn, x, y)
            state_dicts.append(state_dict_to_cpu(m.state_dict()))
        avg = federated_average(state_dicts)
        for key, m in models.items():
            m.load_state_dict(avg)
    return models

def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    sim = SimpleNetworkSim(num_nodes=6)
    sfc_list = make_synthetic_vnfs(sim, num_sfc=3, vnfs_per_sfc=2)
    print("VNFs:", sfc_list)

    # 1) federated predictor (toy)
    models = simple_federated_training(sim, sfc_list)
    print("Federated training done.")

    # 2) train PPO agent to produce placement policies
    model = train_ppo(sim, sfc_list)
    print("PPO trained.")

    # 3) Evaluation: use DummyVecEnv with a proper factory that returns a fresh env
    eval_env = DummyVecEnv([lambda: VNFPlacementEnv(sim, sfc_list)])
    obs = eval_env.reset()
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = eval_env.step(action)
        total_reward += float(np.sum(reward))
        # dones may be array-like from vectorized env
        if getattr(dones, "any", None) and dones.any():
            break
    print("Eval total reward:", total_reward)
    eval_env.close()

    # 4) snapshot & save logs
    logs = []

    # Evaluation loop
    obs = eval_env.reset()
    total_reward = 0.0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = eval_env.step(action)
        total_reward += float(np.sum(reward))

        # Snapshot at each step
        snap = sim.snapshot()
        snap["step"] = step
        logs.append(snap)

        if getattr(dones, "any", None) and dones.any():
            break
        step += 1

    # Save all step-by-step logs
    os.makedirs("results", exist_ok=True)
    with open("results/log.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("✅ Detailed logs saved to results/log.json")


if __name__ == "__main__":
    main()
