import sys
print("Python:", sys.version.splitlines()[0])

import numpy as np, pandas as pd, networkx as nx, simpy
print("numpy", np.__version__, "pandas", pd.__version__, "networkx", nx.__version__, "simpy", simpy.__version__)

try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)

leaf_ok = False
for name in ("leaf", "leafsim"):
    try:
        m = __import__(name)
        print(f"{name} imported, version:", getattr(m, "__version__", "unknown"))
        leaf_ok = True
        break
    except Exception as e:
        print(f"{name} import failed: {e}")

print("LEAF import OK?" , leaf_ok)
print("Smoke test finished")
