"""Project paths and cache redirection (everything must stay inside the project folder)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"
TMP = ROOT / ".tmp"


def setup_env():
    """Redirect library caches into the project. Call before importing matplotlib / torch / huggingface_hub."""
    env = {
        "HF_HOME": MODELS / "hf_cache",
        "TORCH_HOME": MODELS / "torch_cache",
        "MPLCONFIGDIR": TMP / "mpl",
        "XDG_CACHE_HOME": TMP / "xdg",
        "TMPDIR": TMP,
        "CUDA_CACHE_PATH": TMP / "cuda_cache",
        "PYTHONPYCACHEPREFIX": TMP / "pycache",
    }
    for k, v in env.items():
        v.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(v)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # plain HTTPS (the xet backend failed on this network)
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "67108864")   # GDAL /vsicurl/ in-memory cache (64 MB)


setup_env()
