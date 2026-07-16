"""Module 0 — data pipeline for the EG-IPG project.

Each step is a module exposing ``run()``; ``run_module0.run()`` orchestrates them
in dependency order. See AGENTS.md / README.md for project context.
"""
import warnings

# torch_geometric 2.8 calls the now-deprecated ``torch.jit.script`` at import time on torch 2.13
# (torch_geometric/nn/pool/select/base.py) — third-party, not fixable here. Silence just that one
# message, installed before any submodule pulls PyG in, so it doesn't clutter test/smoke output.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=DeprecationWarning)
