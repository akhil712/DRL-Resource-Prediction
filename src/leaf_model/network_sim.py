# src/leaf_model/network_sim.py
import networkx as nx
import numpy as np
import simpy
from collections import defaultdict

class PhysicalNode:
    def __init__(self, nid, cpu_cap=16.0, mem_cap=64.0, Pb=50.0, Pcpu=100.0, Pswitch=20.0):
        self.id = nid
        self.cpu_cap = cpu_cap
        self.mem_cap = mem_cap
        self.Pb = Pb
        self.Pcpu = Pcpu
        self.Pswitch = Pswitch
        # store VNFs as a set of hashable tuples
        self.vnfs = set()
        self.on = False
        self.prev_on = False

    def cpu_util_fraction(self):
        # convert stored hashables back to dicts, then sum cpu
        if not self.vnfs:
            return 0.0
        vnfs_as_dicts = [dict(h) if isinstance(h, dict) else dict(h) for h in [self._hashable_to_rec(h) for h in self.vnfs]]
        # vnfs_as_dicts now list of dicts with 'cpu' keys
        used = sum(v['cpu'] for v in vnfs_as_dicts) if vnfs_as_dicts else 0.0
        return min(1.0, used / self.cpu_cap) if self.cpu_cap > 0 else 0.0

    # the following methods are defined below in class SimpleNetworkSim but we
    # add placeholders here to keep instance methods accessible; actual implementations
    # will refer to the global functions in SimpleNetworkSim via duck-typing in usage.

    def vnfs_state(self):
        # return list of dicts describing each vnf on this node
        return [self._hashable_to_rec(h) for h in self.vnfs]

    def _rec_to_hashable(self, rec_dict):
        return tuple(sorted(rec_dict.items()))

    def _hashable_to_rec(self, h):
        return dict(h)

class SimpleNetworkSim:
    def __init__(self, num_nodes=6):
        # fully connected directed graph (paper used fully connected for example)
        self.G = nx.complete_graph(num_nodes)
        self.nodes = {n: PhysicalNode(n) for n in self.G.nodes()}
        # mapping (sfc_id, vnf_id) -> metadata dict {node, cpu, mem, bw}
        self.vnf_map = {}
        self.time = 0.0
        self.logs = []

    # helper converters (to keep logic consistent)
    def _rec_to_hashable(self, rec_dict):
        return tuple(sorted(rec_dict.items()))

    def _hashable_to_rec(self, h):
        return dict(h)

    def add_vnf(self, sfc, vnf, node_id, cpu, mem, bw):
        key = (sfc, vnf)
        meta = {'node': node_id, 'cpu': cpu, 'mem': mem, 'bw': bw}
        self.vnf_map[key] = meta
        rec = {'sfc': sfc, 'vnf': vnf, 'cpu': cpu, 'mem': mem, 'bw': bw}
        self.nodes[node_id].vnfs.add(self._rec_to_hashable(rec))

    def move_vnf(self, sfc, vnf, target_node):
        key = (sfc, vnf)
        if key not in self.vnf_map:
            return False
        old = self.vnf_map[key]['node']
        rec = {'sfc': sfc, 'vnf': vnf,
               'cpu': self.vnf_map[key]['cpu'],
               'mem': self.vnf_map[key]['mem'],
               'bw': self.vnf_map[key]['bw']}
        old_hash = self._rec_to_hashable(rec)
        # remove from old node if present
        if old_hash in self.nodes[old].vnfs:
            self.nodes[old].vnfs.remove(old_hash)
        # add to target
        self.nodes[target_node].vnfs.add(self._rec_to_hashable(rec))
        # update mapping
        self.vnf_map[key]['node'] = target_node
        return True

    def compute_energy_and_load(self):
        Ptotal = 0.0
        Lcpu_vals = []
        Lmem_vals = []
        for n, node in sorted(self.nodes.items()):
            # record previous on state
            node.prev_on = node.on
            node.on = len(node.vnfs) > 0
            # convert stored hashables to dicts for reading
            vnfs_as_dicts = [self._hashable_to_rec(h) for h in node.vnfs]
            used_cpu = sum(v['cpu'] for v in vnfs_as_dicts) if vnfs_as_dicts else 0.0
            ucpu = min(1.0, used_cpu / node.cpu_cap) if node.cpu_cap > 0 else 0.0
            Pactive = (node.Pb if node.on else 0.0) + ucpu * node.Pcpu
            switch_cost = node.Pswitch if (node.on != node.prev_on) else 0.0
            Ptotal += Pactive + switch_cost
            # load metrics
            Lcpu_vals.append(ucpu)
            mem_used = sum(v['mem'] for v in vnfs_as_dicts) if vnfs_as_dicts else 0.0
            Lmem_vals.append(mem_used / node.mem_cap if node.mem_cap > 0 else 0.0)
        # compute variance metrics
        Lcpu_mean = float(np.mean(Lcpu_vals)) if Lcpu_vals else 0.0
        Lmem_mean = float(np.mean(Lmem_vals)) if Lmem_vals else 0.0
        Lcpu_var = float(np.mean([(x - Lcpu_mean)**2 for x in Lcpu_vals])) if Lcpu_vals else 0.0
        Lmem_var = float(np.mean([(x - Lmem_mean)**2 for x in Lmem_vals])) if Lmem_vals else 0.0
        Ltotal = 0.5 * Lcpu_var + 0.5 * Lmem_var
        return Ptotal, Ltotal

    def snapshot(self):
        P, L = self.compute_energy_and_load()
        self.logs.append({'time': self.time, 'Ptotal': P, 'Ltotal': L})
        return {'time': self.time, 'Ptotal': P, 'Ltotal': L}
