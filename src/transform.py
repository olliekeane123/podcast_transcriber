import subprocess

import torch
import torchaudio
from faster_whisper import WhisperModel

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

import os
from dotenv import load_dotenv

from src.logger import init_logger
logger = init_logger(__name__)



### Convert audio files from mp3 to AI optimised wav format using ffmpeg and subprocess ###

def convert_to_wav_ffmpeg(input_path: str, output_path: str):
    command = [
        "ffmpeg",
        "-y",               # Overwrite output file if it exists
        "-i", input_path,
        "-ar", "16000",     # Audio rate
        "-ac", "1",         # Audio channels (Mono)
        "-c:a", "pcm_s16le",# Codec: 16-bit PCM
        output_path
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg Error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg is not installed or not in your PATH.")
        return False
    

def init_faster_whisper(model_size="large-v3-turbo"):
    
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    return model
    

def run_whisper_pipeline(model, audio_path):
    style_prompt = (
    "Hello and thank you for joining us on Why Theory. I am Ryan Engley, "
    "joined by Todd McGowan. In this episode, we explore the work of "
    "Lacan, Marx, Althusser, Benjamin, and Freud, specifically looking at "
    "the symbolic, the imaginary, and the real."
    )
    segments, info = model.transcribe(
        str(audio_path), 
        beam_size=5, 
        word_timestamps=True,
        language="en",
        initial_prompt=style_prompt,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    segments = list(segments)

    output = {"text": "", "chunks": []}

    for segment in segments:
        output["text"] += segment.text + " "
        
        chunk = {
            "text": segment.text,
            "timestamp": [segment.start, segment.end],
            "words": []
        }
        
        if segment.words:
            for word in segment.words:
                chunk["words"].append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })
        
        output["chunks"].append(chunk)

    return output

def init_pyannote():

    load_dotenv()
    access_token = os.getenv("PYANNOTE_LOCAL_ACCESS_TOKEN")

    pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=access_token)

    # send pipeline to GPU (when available) 
    pipeline.to(torch.device("cuda"))

    return pipeline

def run_pyannote(pipeline, audio_path):

    waveform, sample_rate = torchaudio.load(audio_path)
    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

    result = pipeline(audio_in_memory)

    annotation = result.speaker_diarization
   
    diarization_list = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        diarization_list.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })
    
    return diarization_list



def merge_transcript_and_diarization(transcript_chunks, diarization_segments):
    """
    transcript_chunks: list of dicts with 'text' and 'timestamp' [start, end]
    diarization_segments: list of dicts with 'start', 'end', and 'speaker'
    """
    merged_script = []

    for chunk in transcript_chunks:
        c_start, c_end = chunk["timestamp"]
        c_text = chunk["text"].strip()
        
        # Track which speaker owns the most 'time' in this chunk
        speaker_overlap = {}

        for segment in diarization_segments:
            # Calculate the intersection of the chunk and the speaker segment
            overlap_start = max(c_start, segment["start"])
            overlap_end = min(c_end, segment["end"])
            
            if overlap_end > overlap_start:
                duration = overlap_end - overlap_start
                speaker = segment["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + duration

        # Assign the speaker with the maximum overlap duration
        if speaker_overlap:
            assigned_speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            assigned_speaker = "UNKNOWN"

        merged_script.append({
            "speaker": assigned_speaker,
            "text": c_text,
            "start": round(c_start, 2),
            "end": round(c_end, 2)
        })

    return merged_script


def format_to_human_readable_script(merged_script):

    readable_lines = []

    def format_time(seconds):
        minutes = int(seconds // 60)
        # Using :02d ensures 5 seconds becomes "05" instead of "5"
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    for entry in merged_script:
        speaker = entry["speaker"]
        text = entry["text"]
        start_t = format_time(entry["start"])
        
        # Creates a consistent "Play Script" look
        formatted_line = f"[{start_t}] {speaker}: {text}"
        readable_lines.append(formatted_line)

    return "\n".join(readable_lines)

