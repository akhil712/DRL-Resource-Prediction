# Deep Reinforcement Learning for Resource Demand Prediction and VNF Migration in a Digital Twin Network

This repository contains a practical implementation of a **Digital Twin–based VNF migration framework** inspired by:

- Liu *et al.*, “Deep Reinforcement Learning for Resource Demand Prediction and Virtual Function Network Migration in Digital Twin Network” (IEEE IoT Journal, 2023)
- Wiesner & Thamsen, “LEAF: Simulating Large Energy-Aware Fog Computing Environments” (ICFEC 2021)

The implementation integrates:
- A **LEAFSim-inspired Digital Twin simulator**
- **Federated Bi-GRU** for decentralized demand prediction
- **PPO reinforcement learning** for VNF migration control

---

## 🔧 Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<username>/leaf-vnf-dt.git
cd leaf-vnf-dt
```

### 2️⃣ Create and Activate Virtual Environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
If Gym compatibility warning appears:
```bash
pip install "shimmy>=2.0"
```

### 4️⃣ Smoke Test
```bash
python smoke_test.py
```

### 5️⃣ Run Simulation (Training + Evaluation)
```bash
python src/run_sim.py
```

### 6️⃣ Generate Plots
```bash
python plot_results.py
```

---

## 📁 Project Structure
```
src/
 ├ leaf_model/        # Digital Twin simulation
 ├ agents/            # Bi-GRU + PPO logic
 ├ run_sim.py         # Full pipeline execution
results/              # Logs and plots
smoke_test.py         # Dependency check
plot_results.py       # Visualization script
```

---

## 📊 Output
After execution, the `results/` folder will contain:
- `energy_plot.png`
- `load_plot.png`
- `combined_plot.png`
- `log.json` (raw simulation logs)

---

## 🚀 Implementation Overview

| Component | Purpose |
|----------|---------|
| LEAFSim Digital Twin | Simulates nodes, VNFs & energy usage |
| Federated Bi-GRU | Predicts next-step VNF resource demand |
| PPO Agent | Learns optimal migration decisions |

The reward function is designed to minimize:
```
Energy + Load Variance
```

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: shimmy` | `pip install "shimmy>=2.0"` |
| Gymnasium warning | Safe to ignore |
| CUDA unavailable | Training runs on CPU automatically |

---

## 🔗 References
If this implementation is used for coursework or research, please cite the original papers.

---

## 📄 License
MIT License
