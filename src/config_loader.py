"""Configuration loader for the instrument detection project."""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Singleton configuration loader."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load()

    def load(self, config_path: str = "config.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'data.n_mfcc')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def config(self) -> Dict[str, Any]:
        return self._config


# Global config instance
config = Config()