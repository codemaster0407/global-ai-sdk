from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os

load_dotenv()

elevenlabs_client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)
