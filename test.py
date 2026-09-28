# from audio.eleven_labs.speech_to_text import infer_audio
# from audio.eleven_labs.text_to_speech import generate_audio_from_speech
# from elevenlabs.play import play 
# from llm.z_ai.inference import streaming_response
# from llm.z_ai.get_models import list_models
# import asyncio
# from llm.z_ai import websocket_streaming_response as wss

# wss.main()
# print(list_models())

# streaming_response(system_prompt = 'You are a financial agent who suggests what to purchase', user_prompt = 'I have 10 lakhs INR, where should I invest?')
# output = infer_audio(f'data/audio_files/recording.wav')
# print(output)

# audio = generate_audio_from_speech(text = 'Hi John, How are you Doing? Hope you are doing well.')

# play(audio)

# from llm.redis_caching import functions as red


# asyncio.run(red.push_data(prompt = 'Who is the founder of Boat?', response = 'The founder of Boat is Aman Gupta', attributes = None))

# print(asyncio.run(red.search_query(input_prompt = 'Who is the founder of Boat', threshold = 0.7)))




## LLM Tool Calling Test 


# from llm.z_ai.inference import inference_with_tools


# inference_with_tools(
#     system_prompt = 'You have access to tools and call the appropriate tool as per the user prompt', 
#     user_prompt = 'Add the numbers 3,4,5 and give me the output'
# )

from llm.google.ai_studio import generate 

generate(user_prompt = 'WHat is the capital of france?', system_prompt = 'You are a helpful agent.')