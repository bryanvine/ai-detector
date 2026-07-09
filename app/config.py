"""Configuration from environment (.env is loaded by main before import users read these)."""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Tiny .env loader (KEY=VALUE lines, no quoting games) so we avoid a dependency."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

ROUTER_BASE_URL = os.environ.get("ROUTER_BASE_URL", "http://127.0.0.1:24001/v1").rstrip("/")
ROUTER_API_KEY = os.environ.get("ROUTER_API_KEY", "")

SCORING_MODEL = os.environ.get("SCORING_MODEL", "qwen3-30b-tq")
SECONDARY_SCORING_MODEL = os.environ.get("SECONDARY_SCORING_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3-30b-tq")

IMAGE_ML_ENABLED = os.environ.get("IMAGE_ML_ENABLED", "1") == "1"
IMAGE_ML_MODEL = os.environ.get("IMAGE_ML_MODEL", "Organika/sdxl-detector")

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "27000"))
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "10"))
EXPORT_TOKEN = os.environ.get("EXPORT_TOKEN", "")

# Input limits
MAX_TEXT_CHARS = 60_000          # hard reject above this
SCORING_MAX_CHARS = 16_000       # chars actually sent for logprob scoring / judging
MIN_TEXT_CHARS = 120             # too little text to say anything meaningful
MAX_FILE_BYTES = 25 * 1024 * 1024
