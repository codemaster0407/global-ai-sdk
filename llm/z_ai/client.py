from zai import ZaiClient
from dotenv import load_dotenv 
import os 

load_dotenv()
zai_client = ZaiClient(api_key=os.getenv('Z_AI_API_KEY'))  

