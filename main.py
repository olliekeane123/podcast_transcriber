from src import config, extract, transform, load
import multiprocessing
import datetime
import json
from pathlib import Path
import torch
import gc

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


def ingest_stage():
    """Stage 1: RSS -> MP3 & Manifest"""
    conf = STAGE_MAP["ingestion"]
    limit = config.LIMIT

    # Get data
    rss_text = extract.fetch_rss_feed()
    ep_xml_list = extract.get_ep_xml_list(rss_text)

    for xml in ep_xml_list[:limit]:
        try:
            ep_data = extract.get_ep_metadata(xml)

            if not ep_data:
                continue
                
            ep_id = ep_data["episode_id"]
            manifest_path = conf["manifest_folder"] / f"{ep_id}.json"
            audio_path = conf["audio_folder"] / f"{ep_id}.mp3"

            if manifest_path.exists():
                continue

            logger.info(f"New episode detected: {ep_data["title"]}")
            audio_stream = extract.stream_audio(ep_data["audio_url"])

            if audio_stream:
                ep_data["audio_path"] = str(audio_path)
                ep_data["manifest_path"] = str(manifest_path)

                load.save_ep_audio_stream(audio_stream, ep_data["audio_path"])
                load.save_ep_manifest(ep_data, manifest_path)

        except Exception as err:
            logger.error(f"Ingestion error: {err}")

def process_stage():
    """Stage 2: MP3 -> WAV (16k Mono)"""

    to_process, wav_folder = get_stage_todo("processing")

    if not to_process:
        logger.info("Processing: No new MP3s to convert.")
        return

    for m_path, metadata in to_process:
        try:
            mp3_path = metadata["audio_path"]
            wav_path = wav_folder / f"{metadata['episode_id']}.wav"

            logger.info(f"Converting: {metadata['title']}")
            success = transform.convert_to_wav_ffmpeg(str(mp3_path), str(wav_path))

            if success:
                metadata["wav_path"] = str(wav_path)
                load.save_ep_manifest(metadata, m_path)
                
        except Exception as err:
            logger.error(f"Failed to convert {metadata['title']}: {err}")
    

def transcription_stage():
    """Stage 3: WAV -> Transcription"""

    to_process, save_folder = get_stage_todo("transcription")

    if not to_process:
        logger.info("Transcription: No work found.")
        return

    logger.info(f"Loading Whisper for {len(to_process)} items...")
    model = transform.init_faster_whisper()

    for m_path, metadata in to_process:
        try:

            logger.info(f"Starting transcription: {metadata["title"]}")

            result = transform.run_whisper_pipeline(model, metadata["wav_path"])

            # Save transcription assets
            base_name = save_folder / metadata['episode_id']
            paths = load.save_transcription_assets(str(base_name), result)
            
            # If save completed, update manifest file to mark transcription as completed
            if paths:
                metadata.update({
                    "transcript_path_full": paths["full"],
                    "transcript_path_lite": paths["lite"],
                    "transcript_path_txt": paths["txt"],
                    "transcription_complete": True 
                }) 
                load.save_ep_manifest(metadata, m_path)

        except Exception as err:
            logger.error(f"Failed to transcribe {metadata["title"]}: {err}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def diarization_stage():
    """Stage 4: WAV -> Diarization"""

    to_process, save_folder = get_stage_todo("diarization")

    if not to_process:
        logger.info("Diarization: No work found.")
        return
    

    ### If diarization needs to be done, load model and diarize each episode ###
    logger.info(f"Loading pyannote diarization for {len(to_process)} items...")
    pyannote_pipe = transform.init_pyannote()

    for m_path, metadata in to_process:
        try:
            logger.info(f"Starting diarization: {metadata["title"]}")

            result = transform.run_pyannote(pyannote_pipe, metadata["wav_path"])

            # Save diarization to JSON
            diarize_path = save_folder / f"{metadata['episode_id']}_diarization.json"
            success = load.save_diarization(str(diarize_path), result)
            
            # If save completed, update manifest file to mark diarization as complete
            if success:
                metadata["diarization_path"] = str(diarize_path)
                # One check for if diarization completed
                metadata["diarization_complete"] = True 
                load.save_ep_manifest(metadata, m_path)

        except Exception as err:
            logger.error(f"Failed to diarize {metadata["title"]}: {err}")

    del pyannote_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    
def alignment_stage():
    """Stage 5: Merge Transcript + Diarization -> Final Script"""
    to_process, save_folder = get_stage_todo("alignment")
    
    if not to_process:
        logger.info("Alignment: No work found.")
        return
    
    ### If alignments needs to be done, loop over files to process and save ###
    logger.info("Aligning new transcriptions and diarizations...")

    for m_path, metadata in to_process:
        try:
            logger.info(f"Starting alignment: {metadata["title"]}")

            with open(metadata["transcript_path_full"], "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            with open(metadata["diarization_path"], "r", encoding="utf-8") as f:
                diarization_data = json.load(f)

            aligned_script = transform.merge_transcript_and_diarization(
                transcript_data["chunks"],
                diarization_data
            )
            
            # Also create aligned script in human readable format for post-processing with LLM
            readable_script = transform.format_to_human_readable_script(aligned_script)

            # Save aligned script to JSON and readable script to txt
            aligned_script_path = save_folder / f"{metadata['episode_id']}_aligned_script.json"
            readable_script_path = save_folder / f"{metadata["episode_id"]}_readable_script.txt"
            
            success_json_save = load.save_aligned_script(str(aligned_script_path), aligned_script)
            success_txt_save = load.save_readable_script(str(readable_script_path), readable_script)
                
            # If save completed, update manifest file to mark diarization as complete
            if success_json_save and success_txt_save:
                metadata["aligned_script_path"] = str(aligned_script_path)
                metadata["readable_script_path"] = str(readable_script_path)
                # One check for if alignment completed
                metadata["alignment_complete"] = True 
                load.save_ep_manifest(metadata, m_path)

        except Exception as err:
            logger.error(f"Failed to align script for: {metadata["title"]}: {err}")

        
### Subprocess wrappers ###

def transcription_worker():
    """Isolated worker for Stage 3"""
    try:
        logger.info("Child Process: Starting Transcription Stage")
        transcription_stage()
        logger.info("Child Process: Transcription Stage Finished")
    except Exception as err:
        logger.error(f"Transcription Process Failed: {err}")
        exit(1) 

def diarization_worker():
    """Isolated worker for Stage 4"""
    try:
        logger.info("Child Process: Starting Diarization Stage")
        diarization_stage()
        logger.info("Child Process: Diarization Stage Finished")
    except Exception as err:
        logger.error(f"Diarization Process Failed: {err}")
        exit(1)



### Main ###

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    start = datetime.datetime.now()
    logger.info("Pipeline started")
    
    try:
        ### Stage 1: Ingestion ###
        logger.info("--- Stage 1: Ingestion ---")
        ingest_stage()


        ### Stage 2: Pre-processing
        logger.info("--- Stage 2: Pre-processing ---")
        process_stage()

        """ 
        Note: There seems to be an issue relating to faster-whisper's use of ctranslate2 to
        manage memory and the clean up process - possibly the use of 'del f_whisper_model'
        which leads to a crash. For safety, both transcription and diarization are managed
        by subprocesses which enforces the OS to act as the ultimate garbage collector. And
        
        As the transcription is saved to disk before the crash, we can ignore it and continue
        the script
         """

        ### Stage 3: Transcription (Subprocess) ###
        logger.info("--- Stage 3: Transcription [Isolated Process] ---")
        p_trans = multiprocessing.Process(target=transcription_worker)
        p_trans.start()
        p_trans.join()
        
        if p_trans.exitcode != 0:
            logger.error(f"Transcription crashed (Code {p_trans.exitcode}).")
        
        # Ensure cleanup between processes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ### Stage 4: Diarization (Subprocess) ###
        logger.info("--- Stage 4: Diarization [Isolated Process] ---")
        p_diar = multiprocessing.Process(target=diarization_worker)
        p_diar.start()
        p_diar.join()
        
        if p_diar.exitcode != 0:
            logger.error(f"Diarization crashed (Code {p_diar.exitcode}).")
        
        ### Stage 5: Alignment
        logger.info("--- Stage 5: Alignment ---")
        alignment_stage() 


        complete = True
        
    except Exception as err:
        logger.error(f"Pipeline Error: {err}")
        complete = False
    
    if complete:
        end = datetime.datetime.now()
        logger.info(f"Pipeline finished successfully in {end - start}")