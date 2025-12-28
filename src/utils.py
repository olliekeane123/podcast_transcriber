from src import config
import json

from src.logger import init_logger
logger = init_logger(__name__)

### PIPELINE REGISTRY ###
STAGE_MAP = {
    "ingestion": {
        "audio_folder": config.RAW_AUDIO_DIR,
        "manifest_folder": config.MANIFEST_DIR,
    },
    "processing": {
        "folder": config.WAV_AUDIO_DIR,
        "ready": lambda m: m.get("audio_path") and not m.get("wav_path"),
    },
    "transcription": {
        "folder": config.TRANSCRIPTS_DIR,
        "ready": lambda m: m.get("wav_path") and not m.get("transcription_complete"),
    },
    "diarization": {
        "folder": config.DIARIZATIONS_DIR,
        "ready": lambda m: m.get("wav_path") and not m.get("diarization_complete"),
    },
    "alignment": {
        "folder": config.ALIGNED_SCRIPTS_DIR,
        "ready": lambda m: m.get("transcription_complete") and 
                           m.get("diarization_complete") and 
                           not m.get("alignment_complete"),
    }
}


def get_stage_todo(stage_name):
    """
    Returns a list of (manifest_path, metadata) for a specific stage 
    based on the STAGE_MAP definitions.
    """
    conf = STAGE_MAP.get(stage_name)
    if not conf:
        raise ValueError(f"Stage {stage_name} not found in STAGE_MAP")

    todo = []
    # Loop through manifests as the source of truth
    for m_path in config.MANIFEST_DIR.glob("*.json"):
        try:
            with open(m_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Check the "ready" lambda defined at the top
            if conf["ready"](metadata):
                todo.append((m_path, metadata))
        except Exception as e:
            logger.error(f"Error reading manifest {m_path.name}: {e}")
            
    return todo, conf["folder"]