import json
import logging
import cohere 
import re
from typing import Any, Optional
from src.utils.config import settings
from src.utils.retry_handler import retry_with_exponential_backoff
from src.agents.prompts import (
    SENTIMENT_ANALYSIS_PROMPT,
    TOPIC_EXTRACTION_PROMPT,
    KEYWORD_GENERATION_PROMPT,
    KEYWORD_RECOMMENDATION_PROMPT,
    DRAFT_ANALYSIS_PROMPT,
)

logger = logging.getLogger(__name__)

from openai import AsyncOpenAI

class LLMService:
    def __init__(self):
        
        try:
            self.cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        except Exception:
        
            self.cohere_client = cohere.Client(api_key=settings.COHERE_API_KEY)
        
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_message = "You periodically analyze the evolving blog and refine suggestions by referencing patterns learned from the past blog data. You should suggest new keywords inline or highlight weak sections."
        self.total_tokens = 0

    def clean_json_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response"""
        cleaned = response.strip()
        
        
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            cleaned = cleaned[start_idx:end_idx+1]
        elif '[' in cleaned and ']' in cleaned:
        
            start_idx = cleaned.find('[')
            end_idx = cleaned.rfind(']')
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                cleaned = cleaned[start_idx:end_idx+1]
        
        
        cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)  
        cleaned = re.sub(r':\s*([^",\[\]{}\d\-][^",\[\]{}]*)', r': "\1"', cleaned)  
        cleaned = re.sub(r'"\s*"', r'""', cleaned)  
        
        return cleaned

    def parse_json_safely(self, response: str, fallback_value):
        """Safely parse JSON with multiple fallback strategies"""
        try:
            cleaned = self.clean_json_response(response)
            logger.debug(f"Cleaned JSON: {cleaned}")
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}. Trying manual extraction...")
            
            if isinstance(fallback_value, dict):
                return self.extract_dict_values(response, fallback_value)
            elif isinstance(fallback_value, list):
                return self.extract_list_values(response, fallback_value)
            else:
                return fallback_value
        except Exception as e:
            logger.error(f"Unexpected parsing error: {e}")
            return fallback_value

    def extract_dict_values(self, response: str, fallback: dict) -> dict:
        """Extract dictionary values using regex"""
        result = fallback.copy()
        
        
        polarity_match = re.search(r'["\']?polarity["\']?\s*:?\s*(-?\d*\.?\d+)', response, re.IGNORECASE)
        if polarity_match:
            try:
                result["polarity"] = float(polarity_match.group(1))
            except ValueError:
                pass
        
        
        subjectivity_match = re.search(r'["\']?subjectivity["\']?\s*:?\s*(-?\d*\.?\d+)', response, re.IGNORECASE)
        if subjectivity_match:
            try:
                result["subjectivity"] = float(subjectivity_match.group(1))
            except ValueError:
                pass
        
        
        quality_match = re.search(r'["\']?quality_score["\']?\s*:?\s*(-?\d*\.?\d+)', response, re.IGNORECASE)
        if quality_match:
            try:
                result["quality_score"] = float(quality_match.group(1))
            except ValueError:
                pass
        
        return result

    def extract_list_values(self, response: str, fallback: list) -> list:
        """Extract list values using regex"""
        
        array_matches = re.findall(r'\[([^\]]+)\]', response)
        if array_matches:
            array_content = array_matches[0]
        
            items = re.findall(r'["\']([^"\']+)["\']', array_content)
            if items:
                return items
            
        
            items = [item.strip().strip('"\'') for item in array_content.split(',')]
            return [item for item in items if item]
        
        
        quoted_items = re.findall(r'["\']([^"\']{2,})["\']', response)
        if quoted_items and len(quoted_items) > 1:
            return quoted_items[:7]  
        
        return fallback

    def make_openai_api_call(self, prompt: str, max_tokens: int = None) -> tuple[str, int]:
        def gpt_call():
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            return response.choices[0].message.content, response.usage.total_tokens
        return retry_with_exponential_backoff(gpt_call)

    def make_cohere_api_call(self, prompt: str, max_tokens: int = None) -> tuple[str, int]:
        def cohere_call():
            try:
                
                if hasattr(self.cohere_client, 'chat'):
                    try:
                        res = self.cohere_client.chat(
                            model="command-r-plus",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens or settings.MAX_TOKENS,
                            temperature=settings.TEMPERATURE
                        )
                        
                        
                        content = None
                        if hasattr(res, 'message') and hasattr(res.message, 'content'):
                            if isinstance(res.message.content, list) and len(res.message.content) > 0:
                                content = res.message.content[0].text
                            else:
                                content = str(res.message.content)
                        
                    
                        tokens = 0
                        if hasattr(res, 'usage') and hasattr(res.usage, 'tokens'):
                            tokens = getattr(res.usage.tokens, 'output_tokens', 0)
                        
                        return content, tokens
                        
                    except Exception as v2_error:
                        logger.warning(f"ClientV2 failed: {v2_error}")
                        raise v2_error
                
                
                import asyncio
                res = asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: cohere.Client(api_key=settings.COHERE_API_KEY).generate(
                        model="command-r-plus",
                        prompt=prompt,
                        max_tokens=max_tokens or settings.MAX_TOKENS,
                        temperature=settings.TEMPERATURE
                    )
                )
                
                content = res.generations[0].text if res.generations else ""
                tokens = getattr(res.meta, 'tokens', {}).get('output_tokens', 0) if hasattr(res, 'meta') else 0
                
                return content, tokens
                
            except Exception as e:
                logger.error(f"Cohere API error: {e}")
                raise
        
        return retry_with_exponential_backoff(cohere_call)

    def analyze_sentiment(self, text: str) -> tuple[dict[str, float], int]:
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(text=text[:1000])
        response, tokens = self.make_cohere_api_call(prompt, max_tokens=50)
        
        logger.debug(f"Raw sentiment response: {repr(response)}")
        
        fallback_sentiment = {"polarity": 0.0, "subjectivity": 0.5}
        sentiment = self.parse_json_safely(response, fallback_sentiment)
        if not isinstance(sentiment, dict):
            sentiment = fallback_sentiment
        sentiment = {
            "polarity": float(sentiment.get("polarity", 0.0)),
            "subjectivity": float(sentiment.get("subjectivity", 0.5))
        }

        
        self.total_tokens += tokens
        return sentiment, tokens

    def extract_topics(self, text: str) -> tuple[list[str], int]:
        prompt = TOPIC_EXTRACTION_PROMPT.format(text=text[:1500])
        response, tokens = self.make_cohere_api_call(prompt, max_tokens=100)
        
        fallback_topics = ['general']
        topics = self.parse_json_safely(response, fallback_topics)
        
        if not isinstance(topics, list) or len(topics) == 0:
            topics = fallback_topics
        
        self.total_tokens += tokens
        return topics, tokens

    def generate_initial_keywords(self, text: str) -> tuple[list[str], int]:
        prompt = KEYWORD_GENERATION_PROMPT.format(text=text[:1500])
        response, tokens =  self.make_cohere_api_call(prompt, max_tokens=100)
        
        fallback_keywords = ["content", "blog"]
        keywords = self.parse_json_safely(response, fallback_keywords)
        
        if not isinstance(keywords, list) or len(keywords) == 0:
            keywords = fallback_keywords
        
        self.total_tokens += tokens
        return keywords, tokens

    def recommend_keywords(
        self,
        draft_text: str,
        cursor_context: Optional[str],
        preferred_topics: list[str],
        reading_level: str,
        historical_data: Optional[dict] = None
    ) -> tuple[list[str], int]:
        prompt = KEYWORD_RECOMMENDATION_PROMPT.format(
            draft_text=draft_text[:1500],
            cursor_context=cursor_context or "None",
            preferred_topics=", ".join(preferred_topics),
            reading_level=reading_level,
            historical_data=str(historical_data) if historical_data else "None"
        )
        response, tokens = self.make_cohere_api_call(prompt, max_tokens=150)
        
        fallback_keywords = ["keyword", "content"]
        keywords = self.parse_json_safely(response, fallback_keywords)
        
        if not isinstance(keywords, list) or len(keywords) == 0:
            keywords = fallback_keywords
        
        self.total_tokens += tokens
        return keywords, tokens

    def analyze_draft(self, draft_text: str) -> tuple[dict[str, Any], int]:
        prompt = DRAFT_ANALYSIS_PROMPT.format(draft_text=draft_text[:1500])
        response, tokens = self.make_cohere_api_call(prompt, max_tokens=200)
        
        fallback_analysis = {
            "quality_score": 0.5,
            "structure_notes": "Unable to analyze",
            "improvement_areas": ["clarity"]
        }
        analysis = self.parse_json_safely(response, fallback_analysis)
        
        if not isinstance(analysis, dict):
            analysis = fallback_analysis
        
        self.total_tokens += tokens
        return analysis, tokens

    def get_token_usage(self) -> int:
        return self.total_tokens

    def reset_token_counter(self):
        self.total_tokens = 0