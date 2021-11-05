from stable_baselines.ppo2 import PPO2

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

__version__ = "2.10.1a0"
