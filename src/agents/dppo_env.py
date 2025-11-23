# src/agents/dppo_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class VNFPlacementEnv(gym.Env):
    """
    Simplified environment:
    - Observations: flattened node utilizations + one-hot current VNF node
    - Action: integer = target node index
    """
    metadata = {"render.modes": []}

    def __init__(self, sim, sfc_list, max_nodes=6):
        super().__init__()
        self.sim = sim
        self.sfc_list = sfc_list
        self.num_nodes = len(sim.nodes)
        obs_dim = self.num_nodes * 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes)
        self.ptr = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = 0
        obs = self._get_obs(self.ptr)
        info = {}
        return obs, info

    def step(self, action):
        sfc, vnf = self.sfc_list[self.ptr]
        self.sim.move_vnf(sfc, vnf, int(action))
        P, L = self.sim.compute_energy_and_load()
        reward = -0.5 * P - 0.5 * (L * 100.0)
        self.ptr += 1
        terminated = self.ptr >= len(self.sfc_list)
        truncated = False
        obs = self._get_obs(self.ptr if not terminated else len(self.sfc_list) - 1)
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self, ptr):
        cpu_fracs = [self.sim.nodes[n].cpu_util_fraction() for n in sorted(self.sim.nodes)]
        sfc, vnf = self.sfc_list[min(ptr, len(self.sfc_list) - 1)]
        node_idx = self.sim.vnf_map[(sfc, vnf)]['node']
        onehot = [1.0 if i == node_idx else 0.0 for i in range(self.num_nodes)]
        return np.array(cpu_fracs + onehot, dtype=np.float32)
