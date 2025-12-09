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

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.ini"


class AgentConfig(BaseModel):
    """Configuration specific to agent lifecycle and per-agent settings."""

    max_concurrent_agents: int = Field(
        default=10,
        ge=1,
        description="Maximum number of agents allowed to run concurrently",
    )
    agent_timeout: int = Field(
        default=300,
        ge=1,
        description="Timeout in seconds for an active agent before termination",
    )
    enable_agent_to_agent_communication: bool = Field(
        default=True,
        description="Allow agents to send messages to each other",
    )
    agent_model_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "identify_trigger_agent": "openai:gpt-4o-mini",
            "root_cause_analyzer_agent": "openai:gpt-4o",
            "scope_definition_agent": "openai:gpt-4o",
            "drafting_agent": "openai:gpt-4o",
            "establish_goals_agent": "openai:gpt-4o-mini",
            "identify_information_needed_agent": "openai:gpt-4o-mini",
            "retrieve_information_needed_agent": "openai:gpt-4o-mini",
            "draft_update_agent": "openai:gpt-4o",
            "generation_of_alternatives_agent": "openai:gpt-4o",
            "result_agent": "openai:gpt-4o",
        },
        description="Per-agent model overrides keyed by agent name",
    )
    agent_temperature_mapping: dict[str, float] = Field(
        default_factory=dict,
        description="Optional per-agent temperature overrides",
    )


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
        default="openai:gpt-4.1-mini",
        description="AI model name to use for agents"
    )
    
    # Alias for backward compatibility
    @property
    def decision_model_name(self) -> str:
        """Alias for model_name"""
        return self.model_name
    
    evaluation_model: str = Field(
        default="openai:gpt-5-nano",
        description="AI model name to use for evaluators"
    )
    
    # Alias for backward compatibility
    @property
    def evaluator_model_name(self) -> str:
        """Alias for evaluation_model"""
        return self.evaluation_model

    fallback_model_name: Optional[str] = Field(
        default=None,
        description="Optional fallback model for agent calls when primary fails"
    )

    # Retry & Circuit Breaker Configuration
    agent_max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retry attempts for agent execution (excluding fallback)"
    )

    agent_retry_backoff: float = Field(
        default=0.5,
        ge=0.0,
        description="Exponential backoff base (seconds) for retries"
    )

    agent_retry_max_backoff: float = Field(
        default=8.0,
        ge=0.0,
        description="Maximum backoff delay between retries (seconds)"
    )

    agent_retry_jitter: float = Field(
        default=0.2,
        ge=0.0,
        description="Random jitter added to retry backoff (seconds)"
    )

    circuit_breaker_failure_threshold: int = Field(
        default=3,
        ge=1,
        description="Number of consecutive failures before tripping the circuit breaker"
    )

    circuit_breaker_recovery_time: float = Field(
        default=30.0,
        ge=1.0,
        description="Time window (seconds) before a half-open probe is allowed"
    )

    circuit_breaker_half_open_success_threshold: int = Field(
        default=1,
        ge=1,
        description="Number of successful half-open calls needed to close the breaker"
    )

    # ===== Tooling Configuration =====
    tool_rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum tool executions per minute across the process"
    )

    tool_execution_timeout: float = Field(
        default=15.0,
        ge=1.0,
        description="Per-tool execution timeout in seconds"
    )

    enable_tool_audit_log: bool = Field(
        default=True,
        description="Enable audit logging for tool executions"
    )
    
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

    # ===== Agent Lifecycle & Configuration =====
    agent_config: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Lifecycle, concurrency, and per-agent configuration",
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
    
    # ===== Observability & Tracing Configuration =====
    enable_tracing: bool = Field(
        default=True,
        description="Enable distributed tracing with logfire"
    )
    
    logfire_token: Optional[str] = Field(
        None,
        description="Logfire API token for tracing (optional, will use environment variable if not set)"
    )
    
    trace_sampling_rate: float = Field(
        default=1.0,
        description="Sampling rate for traces (0.0 to 1.0, where 1.0 traces all requests)"
    )
    
    track_token_usage: bool = Field(
        default=True,
        description="Track token usage and cost for each agent execution"
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

    persistence_backend: str = Field(
        default="memory",
        description="Persistence backend: memory | redis | postgres",
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

    # Convenience accessors for agent configuration
    @property
    def max_concurrent_agents(self) -> int:
        return self.agent_config.max_concurrent_agents

    @property
    def agent_timeout(self) -> int:
        return self.agent_config.agent_timeout

    @property
    def agent_model_mapping(self) -> dict[str, str]:
        return self.agent_config.agent_model_mapping

    @property
    def agent_temperature_mapping(self) -> dict[str, float]:
        return self.agent_config.agent_temperature_mapping

    def get_agent_model(self, agent_name: str) -> str:
        """Return the configured model for an agent, falling back to decision model."""
        return self.agent_model_mapping.get(agent_name, self.decision_model_name)
    
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
