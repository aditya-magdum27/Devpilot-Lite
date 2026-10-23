"""config.py - load settings from .env and check them. Never prints secrets."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = APP_DIR / "workspace"
RUNS_DIR = WORKSPACE_DIR / ".runs"  # managed folder for isolated working copies
SUPPORTED_PROVIDERS = ("openai", "anthropic", "groq")


def _int_env(name: str, default: int, low: int, high: int) -> int:
    """Read an integer from the environment; fall back to default and keep it in range."""
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


@dataclass(frozen=True)
class Config:
    provider: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)  # repr=False: never shown by accident
    base_url: str = ""
    langsmith_tracing: bool = False
    langsmith_api_key: str = field(default="", repr=False)
    max_repair_attempts: int = 1
    timeout_seconds: int = 60
    max_files: int = 15
    max_file_bytes: int = 100_000
    workspace_dir: Path = WORKSPACE_DIR
    runs_dir: Path = RUNS_DIR

    def problems(self) -> list[str]:
        """List what is missing or wrong. Empty list means the model is ready to use."""
        found = []
        if not self.provider:
            found.append(f"LLM_PROVIDER is not set (use one of: {', '.join(SUPPORTED_PROVIDERS)}).")
        elif self.provider not in SUPPORTED_PROVIDERS:
            found.append(f"LLM_PROVIDER '{self.provider}' is not supported (use: {', '.join(SUPPORTED_PROVIDERS)}).")
        if not self.model:
            found.append("LLM_MODEL is not set.")
        if not self.api_key:
            found.append("LLM_API_KEY is not set.")
        return found

    @property
    def ready(self) -> bool:
        return not self.problems()

    @property
    def tracing_enabled(self) -> bool:
        return self.langsmith_tracing and bool(self.langsmith_api_key)


def load_config(load_env_file: bool = True) -> Config:
    """Build a Config from environment variables (and the .env file if present)."""
    if load_env_file:
        load_dotenv(APP_DIR / ".env", override=False)
    return Config(
        provider=os.getenv("LLM_PROVIDER", "").strip().lower(),
        model=os.getenv("LLM_MODEL", "").strip(),
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "").strip(),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true",
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", "").strip(),
        max_repair_attempts=_int_env("MAX_REPAIR_ATTEMPTS", 1, 0, 3),
        timeout_seconds=_int_env("EXECUTION_TIMEOUT_SECONDS", 60, 5, 600),
        max_files=_int_env("MAX_FILES_TO_INSPECT", 15, 1, 50),
        max_file_bytes=_int_env("MAX_FILE_BYTES", 100_000, 1_000, 1_000_000),
    )


def setup_tracing(cfg: Config) -> None:
    """LangSmith is optional: switch it on only when both the flag and the key are present."""
    if cfg.tracing_enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = cfg.langsmith_api_key
    else:
        os.environ["LANGSMITH_TRACING"] = "false"


def redact(text: str, cfg: Config) -> str:
    """Remove API key values from any text before it is shown to the user."""
    for secret in (cfg.api_key, cfg.langsmith_api_key):
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text
