from audio.eleven_labs.speech_to_text import infer_audio
from audio.eleven_labs.text_to_speech import generate_audio_from_speech
from elevenlabs.play import play 
from llm.z_ai.inference import streaming_response
from llm.z_ai.get_models import list_models
import asyncio
# print(list_models())

# streaming_response(system_prompt = 'You are a financial agent who suggests what to purchase', user_prompt = 'I have 10 lakhs INR, where should I invest?')
# output = infer_audio(f'data/audio_files/recording.wav')
# print(output)

# audio = generate_audio_from_speech(text = 'Hi John, How are you Doing? Hope you are doing well.')

# play(audio)

from llm.redis_caching import functions as red


# asyncio.run(red.push_data(prompt = 'Who is the founder of Boat?', response = 'The founder of Boat is Aman Gupta', attributes = None))

print(red.search_query(input_prompt = 'Who is the founder of Boat',threshold = 0.7))

