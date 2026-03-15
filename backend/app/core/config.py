"""KavachX Configuration Settings"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./kavachx.db"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173,http://127.0.0.1:3000"
    SECRET_KEY: str = "kavachx-dev-secret-change-in-production"
    ENVIRONMENT: str = "development"

    def get_cors_origins(self) -> List[str]:
        return [s.strip() for s in self.CORS_ORIGINS.split(",")]
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours

    # Governance thresholds
    RISK_SCORE_HIGH_THRESHOLD: float = 0.60
    RISK_SCORE_MEDIUM_THRESHOLD: float = 0.20
    FAIRNESS_DISPARITY_THRESHOLD: float = 0.20
    CONFIDENCE_LOW_THRESHOLD: float = 0.55

    class Config:
        env_file = ".env"


settings = Settings()

import json
import os

def load_thresholds():
    try:
        if os.path.exists("kavachx_thresholds.json"):
            with open("kavachx_thresholds.json", "r") as f:
                t = json.load(f)
                settings.RISK_SCORE_HIGH_THRESHOLD = t.get("risk_high", settings.RISK_SCORE_HIGH_THRESHOLD)
                settings.RISK_SCORE_MEDIUM_THRESHOLD = t.get("risk_medium", settings.RISK_SCORE_MEDIUM_THRESHOLD)
                settings.FAIRNESS_DISPARITY_THRESHOLD = t.get("fairness_disparity", settings.FAIRNESS_DISPARITY_THRESHOLD)
                settings.CONFIDENCE_LOW_THRESHOLD = t.get("confidence_low", settings.CONFIDENCE_LOW_THRESHOLD)
    except Exception as e:
        print(f"Failed to load user thresholds: {e}")

def save_thresholds(t: dict):
    with open("kavachx_thresholds.json", "w") as f:
        json.dump(t, f)
    load_thresholds()

load_thresholds()
