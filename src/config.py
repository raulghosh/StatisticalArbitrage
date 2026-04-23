"""
config.py — Central configuration for the StatArb project.
 
All parameters live here. Never hardcode values in strategy files.
Loaded once at startup; accessed everywhere via get_config().
"""

import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# ===========================================
# Load environment variables
# ===========================================
load_dotenv()

# ===========================================
# Schwab Configuration
# ===========================================
class SchwabConfig(BaseModel):
    api_key: str = Field(default=os.getenv("SCHWAB_APP_KEY"), env="SCHWAB_APP_KEY")
    api_secret: str = Field(default=os.getenv("SCHWAB_APP_SECRET"), env="SCHWAB_APP_SECRET")
    callback_url: str = Field(default=os.getenv("SCHWAB_CALLBACK_URL"), env="SCHWAB_CALLBACK_URL")

    @field_validator("api_key", "api_secret", "callback_url", mode="before")
    @classmethod
    def check_required(cls, v: str) -> str:
        if not v:
            raise ValueError("Required environment variable is missing")
        return v