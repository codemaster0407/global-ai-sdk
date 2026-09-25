from audio.eleven_labs.client import elevenlabs_client
import os

def generate_audio_from_speech(text):
    try:
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George" - browse voices at elevenlabs.io/app/voice-library
            model_id="eleven_v3",
            output_format="mp3_44100_128",
        )


        return audio 

    except Exception as e:
        print(f'Exception at {os.getcwd()}/text_to_speech.py : {e}')