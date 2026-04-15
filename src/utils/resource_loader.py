from pathlib import Path
from dotenv import load_dotenv
import os

def get_env_path() -> Path:
    base = Path.home() / "Library" / "Application Support" / "Vignity"
    return base / ".env"

def load_app_env(env_path=None):
    env_path = env_path or get_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(env_path, override=True)
    return env_path

def get_env_var(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()
