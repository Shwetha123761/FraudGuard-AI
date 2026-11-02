# core/config.py
import os
from pathlib import Path

# Base directory of the major project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories & file paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "full_scaled_data.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

# Model weights paths (migrated from mini-project or new)
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
AUTOENCODER_WEIGHTS = WEIGHTS_DIR / "autoencoder.keras"
GNN_WEIGHTS = WEIGHTS_DIR / "gnn_model.pt"

# Database configuration (SQLite for local major project, easily upgradable to PostgreSQL)
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'fraudguard.db'}"

# Redis Feature Store Settings (for local Docker or local redis-server)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Decision Engine Thresholds (Master Risk Score 0 - 100)
RISK_THRESHOLD_APPROVE = 30.0     # Score < 30: Approved
RISK_THRESHOLD_OTP = 75.0         # 30 <= Score <= 75: Step-up OTP Verification
# Score > 75: Automatically Blocked
