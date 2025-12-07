"""
Configuration Management for Multi-Agent Decision Making Application

This module handles all application configuration using pydantic-settings.
Configuration precedence (highest to lowest):
1. Environment variables
2. .env file
3. config.ini file
4. Default values
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import configparser

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.ini"


class Settings(BaseSettings):
    """
    Application Settings
    
    All settings can be overridden via environment variables.
    Example: export API_KEY=your-key-here
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields
        validate_assignment=True
    )
    
    # ===== API Configuration =====
    api_key: Optional[str] = Field(
        None,
        description="API key for AI model provider (e.g., OpenAI, Anthropic)"
    )
    
    model_name: str = Field(
        default="gpt-4.1-mini",
        description="AI model name to use for agents"
    )
    
    # Alias for backward compatibility
    @property
    def decision_model_name(self) -> str:
        """Alias for model_name"""
        return self.model_name
    
    evaluation_model: str = Field(
        default="gpt-5-nano",
        description="AI model name to use for evaluators"
    )
    
    # Alias for backward compatibility
    @property
    def evaluator_model_name(self) -> str:
        """Alias for evaluation_model"""
        return self.evaluation_model
    
    # ===== Server Configuration =====
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    
    port: int = Field(
        default=8000,
        description="Server port number"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    reload: bool = Field(
        default=True,
        description="Enable auto-reload in development"
    )
    
    # ===== CORS Configuration =====
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    
    cors_allow_methods: list[str] = Field(
        default=["*"],
        description="Allowed HTTP methods"
    )
    
    cors_allow_headers: list[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )
    
    # ===== Application Configuration =====
    app_name: str = Field(
        default="Multi-Agent Decision Making API",
        description="Application name"
    )
    
    app_version: str = Field(
        default="0.2.0",
        description="Application version"
    )
    
    timeout_seconds: int = Field(
        default=300,
        description="Request timeout in seconds"
    )
    
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent decision-making processes"
    )
    
    # ===== Database Configuration (optional, for future use) =====
    database_url: Optional[str] = Field(
        None,
        description="Database connection URL"
    )
    
    # ===== Redis Configuration =====
    enable_redis_persistence: bool = Field(
        default=False,
        description="Enable Redis for process persistence"
    )
    
    redis_host: str = Field(
        default="localhost",
        description="Redis server host"
    )
    
    redis_port: int = Field(
        default=6379,
        description="Redis server port"
    )
    
    redis_db: int = Field(
        default=0,
        description="Redis database number"
    )
    
    redis_password: Optional[str] = Field(
        None,
        description="Redis password (if authentication enabled)"
    )
    
    redis_url: Optional[str] = Field(
        None,
        description="Redis connection URL (alternative to individual fields)"
    )
    
    # ===== Logging Configuration =====
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    # ===== File Storage =====
    persistence_dir: Path = Field(
        default=BASE_DIR / "data" / "persistence",
        description="Directory for persistent storage"
    )
    
    prompts_dir: Path = Field(
        default=BASE_DIR / "app" / "core" / "prompts" / "templates",
        description="Directory containing system prompts"
    )
    
    @field_validator("persistence_dir", "prompts_dir")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        """Ensure directories exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """
        Customize settings sources to include config.ini file
        
        Priority order:
        1. init_settings (passed to Settings())
        2. env_settings (environment variables)
        3. dotenv_settings (.env file)
        4. ini_settings (config.ini file)
        5. file_secret_settings (Docker secrets)
        """
        
        def ini_settings():
            """Load settings from config.ini"""
            if not CONFIG_FILE.exists():
                return {}
            
            config = configparser.ConfigParser()
            config.read(CONFIG_FILE)
            
            settings_dict = {}
            
            # Read from DEFAULT section
            for key in config.defaults():
                settings_dict[key.lower()] = config.get("DEFAULT", key)
            
            # Read from app section if it exists
            if config.has_section("app"):
                for key, value in config.items("app"):
                    if key not in config.defaults():  # Don't override defaults
                        settings_dict[key.lower()] = value
            
            return settings_dict
        
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            ini_settings,
            file_secret_settings,
        )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get application settings (singleton pattern)
    
    Args:
        reload: If True, reload settings from sources
        
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None or reload:
        _settings = Settings()
    
    return _settings


# For convenience
settings = get_settings()


if __name__ == "__main__":
    # Test configuration loading
    import json
    s = get_settings()
    print("Loaded configuration:")
    print(json.dumps(s.model_dump(), indent=2, default=str))
