import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f) or {}
    

cfg = load_config()

LIMIT = cfg.get("max_episodes") or None


BASE_DATA = Path(cfg['paths']['data_root'])
RAW_AUDIO_DIR: Path = BASE_DATA / cfg['paths']['raw_subfolder'] / cfg['paths']['raw_audio_subfolder']
MANIFEST_DIR: Path = BASE_DATA / cfg['paths']['raw_subfolder'] / cfg['paths']['manifest_subfolder']
WAV_AUDIO_DIR: Path = BASE_DATA / cfg['paths']['raw_subfolder'] / cfg['paths']['wav_audio_subfolder']
TRANSCRIPTS_DIR: Path = BASE_DATA / cfg['paths']['processed_subfolder'] / cfg['paths']['transcripts_subfolder']
DIARIZATIONS_DIR: Path = BASE_DATA / cfg['paths']['processed_subfolder'] / cfg['paths']['diarizations_subfolder']
ALIGNED_SCRIPTS_DIR: Path = BASE_DATA / cfg['paths']['processed_subfolder'] / cfg['paths']['aligned_scripts_subfolder']

RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
WAV_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
DIARIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)