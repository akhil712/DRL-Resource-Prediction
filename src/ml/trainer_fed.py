# src/ml/trainer_fed.py
import torch
import torch.optim as optim
import torch.nn as nn
from ml.fedbi_gru import BiGRUModel

def train_local_one_step(model, optimizer, loss_fn, x_tensor, y_tensor):
    model.train()
    optimizer.zero_grad()
    pred = model(x_tensor)
    loss = loss_fn(pred, y_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

def state_dict_to_cpu(sd):
    return {k: v.cpu().detach().clone() for k, v in sd.items()}

def federated_average(state_dicts):
    # state_dicts: list of state_dicts (all on CPU)
    avg = {}
    keys = state_dicts[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        avg[k] = torch.mean(stacked, dim=0)
    return avg
