# example.py
import os
from audio.eleven_labs.client import elevenlabs_client


def infer_audio(audio_file_path):
    '''
    Get .wav file or .mp3 file
    '''
    try:
        with open(audio_file_path, "rb") as f:
            result = elevenlabs_client.speech_to_text.convert(file=f, 
                                                model_id="scribe_v2", 
                                                language_code="eng", 
                                                diarize = True)
            
        return result.text

    except Exception as e:
        print(f'[ERROR LOG] Error at {os.getcwd()}/speech_to_text.py : {e}')
    
