import json

from src.logger import init_logger
logger = init_logger(__name__)

### Save episode to disk functions ###

def save_ep_audio_stream(audio_stream, save_path):
    try:
        with open(save_path, "wb") as f:
            for chunk in audio_stream.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Saved audio to {save_path}")
    except Exception as err:
        logger.error(f"Failed to save audio: {err}")
        

def save_ep_manifest(manifest_data, save_path):
    try:
        save_to_json(save_path, manifest_data)
        logger.info(f"Saved manifest to {save_path}")
    except Exception as err:
        logger.error(f"Failed to save manifest: {err}")


def save_transcription_assets(base_path, transcription_result):
    try:
        ### 1. Save full (word level) JSON ###
        full_path = f"{base_path}_full.json"
        save_to_json(full_path, transcription_result)


        ### 2. Save Lite (removing words and their timestamps) JSON ###
        lite_result = {
            "text": transcription_result["text"],
            "chunks": [
                {key: value for key, value in chunk.items() if key != "words"} 
                for chunk in transcription_result["chunks"]
            ]
        }
        lite_path = f"{base_path}_lite.json"
        save_to_json(lite_path, lite_result)


        ### 3. Save raw text as txt file ###
        txt_path = f"{base_path}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcription_result["text"])

        
        ### Finish and return paths for manifest ###
        logger.info(f"Saved all transcription assets for {base_path}")
        return {"full": full_path, "lite": lite_path, "txt": txt_path}
    
    except Exception as err:
        logger.error(f"Failed to save transcription assets: {err}")
        return None
    

def save_diarization(save_path, diarization_data):
    try:
        save_to_json(save_path, diarization_data)
        logger.info(f"Saved diarization data to {save_path}")
        return True
    except Exception as err:
        logger.error(f"Error saving diarization: {err}")
        return None
    
    
def save_aligned_script(save_path, aligned_script):
    try:
        save_to_json(save_path, aligned_script)
        logger.info(f"Saved aligned script to {save_path}")
        return True
    except Exception as err:
        logger.error(f"Error saving aligned script: {err}")
        return None

def save_readable_script(save_path, readable_script):
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(readable_script)
        logger.info(f"Saved readable script to {save_path}")
        return True
    except Exception as err:
        logger.error(f"Error saving readable script: {err}")
        return None


### Helper functions ###

def save_to_json(save_path, data):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

        