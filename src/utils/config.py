import os
from dotenv import load_dotenv
load_dotenv()
import cohere
class Settings:
    COHERE_API_KEY=os.getenv("COHERE_API_KEY")    
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    API_KEY=os.getenv("API_KEY","genai-intern")
    MODEL_NAME='gemini-2.0-flash'
    MAX_TOKENS=2000
    TEMPERATURE=0.3
    MAX_RETRIES=3
    RETRY_DELAY=1
    KEYWORD_RELEVANCE_WEIGHT=0.4
    READABILITY_WEIGHT=0.3
    USER_PROFILE_WEIGHT=0.3

settings=Settings()