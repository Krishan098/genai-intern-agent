import json
import logging
import cohere 
from typing import Any,Optional
from src.utils.config import settings
from src.utils.retry_handler import retry_with_exponential_backoff
from src.agents.prompts import(
    SENTIMENT_ANALYSIS_PROMPT,
    TOPIC_EXTRACTION_PROMPT,
    KEYWORD_GENERATION_PROMPT,
    KEYWORD_RECOMMENDATION_PROMPT,
    DRAFT_ANALYSIS_PROMPT,
)
logger=logging.getLogger(__name__)

from openai import AsyncOpenAI
class LLMService:
    def __init__(self):
        self.Cohere_client=cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        self.client=AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_message="You periodically analyze the evolving blog and refine suggestions by referencing patterns learned from the past blog data.You should suggest new keywords inline or highlight weak sections."
        self.total_tokens=0
    async def make_openai_api_call(self,prompt:str,max_tokens:int=None)->tuple[str,int]:
        async def gpt_call():
            response=await self.client.chat.completions.ceate(
                model=settings.MODEL_NAME,
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens or settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            return response.choices[0].message.content,response.usage.total_tokens
        return await retry_with_exponential_backoff(gpt_call)
    async def make_cohere_api_call(self,prompt:str,max_tokens:int=None)->tuple[str,int]:
        async def cohere_call():
            res=await self.Cohere_client.chat(model="command-a-03-2025",messages=[{"role":"system","content":self.system_message},
                                                                                  {"role":"user","content":prompt}],max_tokens=max_tokens or settings.MAX_TOKENS,temperature=settings.TEMPERATURE)
            return res.choices[0].message.content,res.usage.total_tokens
        return retry_with_exponential_backoff(cohere_call)
    async def analyze_sentiment(self,text:str)->tuple[dict[str,float],int]:
        prompt=SENTIMENT_ANALYSIS_PROMPT.format(text=text[:1000])
        response,tokens=await self.make_cohere_api_call(prompt,max_tokens=50)
        try:
            sentiment =json.loads(response)
            self.total_tokens+=tokens
            return sentiment,tokens
        except json.JSONDecodeError:
            logger.error(f"Failed to parse sentiment response:{response}")
            return {"polarity":0.0,"subjectivity":0.5},tokens
    async def extract_topics(self,text:str)->tuple[list[str],int]:
        prompt=TOPIC_EXTRACTION_PROMPT.format(text=text[:1500])
        response,tokens=await self.make_cohere_api_call(prompt,max_tokens=100)
        try:
            topics=json.loads(response)
            self.total_tokens+=tokens
            return topics,tokens
        except json.JSONDecodeError:
            logger.error(f"Failed to parse topics response: {response}")
            return ['general'], tokens
    async def generate_initial_keywords(self,text:str)->tuple[list[str],int]:
        """Generate initial keyword suggestions"""
        prompt=KEYWORD_GENERATION_PROMPT.format(text=text[:1500])
        response,tokens=await self.make_cohere_api_call(prompt,max_tokens=100)
        try:
            keywords=json.loads(response)
            self.total_tokens+=tokens
            return keywords,tokens
        except json.JSONDecodeError:
            logger.error(f"Failed to parse keywords response:{response}")
            return ["content","blog"],tokens
    async def recommend_keywords(
        self,
        draft_text:str,
        cursor_context:Optional[str],
        preferred_topics:list[str],
        reading_level:str,
        historical_data:Optional[dict]=None
    )->tuple[list[str],int]:
        prompt=KEYWORD_RECOMMENDATION_PROMPT.format(
            draft_text=draft_text[:1500],
            cursor_context=cursor_context or "None",
            preferred_topics=", ".join(preferred_topics),
            reading_level=reading_level,
            historical_data=str(historical_data) if historical_data else "None"
        )
        response,tokens=await self.make_cohere_api_call(prompt,max_tokens=150)
        try:
            keywords=json.loads(response)
            self.total_tokens+=tokens
            return keywords,tokens
        except json.JSONDecodeError:
            logger.error(f"Failed to parse keyword recommendations: {response}")
            return ["keyword","content"], tokens
    async def analyze_draft(self,draft_text:str)->tuple[dict[str,Any],int]:
        """Analyze draft quality and structure"""
        prompt=DRAFT_ANALYSIS_PROMPT.format(draft_text=draft_text[:1500])
        response,tokens=await self.make_cohere_api_call(prompt,max_tokens=200)
        try:
            analysis=json.loads(response)
            self.total_tokens+=tokens
            return analysis,tokens
        except json.JSONDecodeError:
            logger.error(f"Failed to parse draft analysis:{response}")
            return{
                "quality_score":0.5,
                "structure_notes":"Unable to analyze",
                "improvement_areas":["clarity"]
            },tokens
    def get_token_usage(self)->int:
        """Get total token usage for this session"""        
        return self.total_tokens
    def reset_token_counter(self):
        """Reset token counter"""
        self.total_tokens=0