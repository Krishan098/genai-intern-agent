import json
import logging
import cohere 
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
        # Try both sync and async Cohere clients
        try:
            self.cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        except Exception:
            # Fallback to sync client
            self.cohere_client = cohere.Client(api_key=settings.COHERE_API_KEY)
        
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_message = "You periodically analyze the evolving blog and refine suggestions by referencing patterns learned from the past blog data. You should suggest new keywords inline or highlight weak sections."
        self.total_tokens = 0

    async def make_openai_api_call(self, prompt: str, max_tokens: int = None) -> tuple[str, int]:
        async def gpt_call():
            response = await self.client.chat.completions.create(  # Fixed typo: ceate -> create
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            return response.choices[0].message.content, response.usage.total_tokens
        return await retry_with_exponential_backoff(gpt_call)

    async def make_cohere_api_call(self, prompt: str, max_tokens: int = None) -> tuple[str, int]:
        async def cohere_call():
            try:
                # Check if we have async or sync client
                is_async_client = hasattr(self.cohere_client, 'chat') and callable(getattr(self.cohere_client, 'chat'))
                
                # Prepare the call parameters
                call_params = {
                    "model": "command-r-plus",  # Use the most reliable model
                    "message": prompt,  # For sync client
                    "max_tokens": max_tokens or settings.MAX_TOKENS,
                    "temperature": settings.TEMPERATURE
                }
                
                # For ClientV2 (async), use messages format
                if hasattr(self.cohere_client, 'chat'):
                    try:
                        res = await self.cohere_client.chat(
                            model="command-r-plus",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens or settings.MAX_TOKENS,
                            temperature=settings.TEMPERATURE
                        )
                    except Exception as v2_error:
                        logger.warning(f"ClientV2 failed, trying sync client: {v2_error}")
                        # Fallback to sync client
                        import asyncio
                        res = await asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: cohere.Client(api_key=settings.COHERE_API_KEY).generate(
                                model="command-r-plus",
                                prompt=prompt,
                                max_tokens=max_tokens or settings.MAX_TOKENS,
                                temperature=settings.TEMPERATURE
                            )
                        )
                else:
                    # Use sync client in executor
                    import asyncio
                    res = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.cohere_client.generate(
                            model="command-r-plus",
                            prompt=prompt,
                            max_tokens=max_tokens or settings.MAX_TOKENS,
                            temperature=settings.TEMPERATURE
                        )
                    )
                
                logger.debug(f"Cohere response type: {type(res)}")
                logger.debug(f"Cohere response: {res}")
                
                # Handle different response structures
                content = None
                tokens = 0
                
                # For ClientV2 responses
                if hasattr(res, 'message'):
                    if hasattr(res.message, 'content'):
                        if isinstance(res.message.content, list) and len(res.message.content) > 0:
                            if hasattr(res.message.content[0], 'text'):
                                content = res.message.content[0].text
                            else:
                                content = str(res.message.content[0])
                        else:
                            content = str(res.message.content)
                    elif hasattr(res.message, 'text'):
                        content = res.message.text
                
                # For sync client responses
                elif hasattr(res, 'generations') and len(res.generations) > 0:
                    content = res.generations[0].text
                elif hasattr(res, 'text'):
                    content = res.text
                
                # Extract token usage
                if hasattr(res, 'usage'):
                    if hasattr(res.usage, 'tokens'):
                        if hasattr(res.usage.tokens, 'output_tokens'):
                            tokens = res.usage.tokens.output_tokens
                        elif hasattr(res.usage.tokens, 'total_tokens'):
                            tokens = res.usage.tokens.total_tokens
                    elif hasattr(res.usage, 'total_tokens'):
                        tokens = res.usage.total_tokens
                elif hasattr(res, 'meta') and hasattr(res.meta, 'tokens'):
                    if hasattr(res.meta.tokens, 'output_tokens'):
                        tokens = res.meta.tokens.output_tokens
                    elif hasattr(res.meta.tokens, 'total_tokens'):
                        tokens = res.meta.tokens.total_tokens
                
                if content is None:
                    logger.error(f"Could not extract content from response: {res}")
                    logger.error(f"Response attributes: {[attr for attr in dir(res) if not attr.startswith('_')]}")
                    raise ValueError("Unable to extract content from Cohere response")
                
                logger.debug(f"Extracted content: {repr(content)}")
                logger.debug(f"Extracted tokens: {tokens}")
                
                return content, tokens
                
            except Exception as e:
                logger.error(f"Error in Cohere API call: {e}")
                logger.error(f"Error type: {type(e)}")
                raise
        
        return await retry_with_exponential_backoff(cohere_call)

    async def analyze_sentiment(self, text: str) -> tuple[dict[str, float], int]:
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(text=text[:1000])
        response, tokens = await self.make_cohere_api_call(prompt, max_tokens=50)
        
        # DEBUG: Log the raw response
        logger.debug(f"Raw Cohere sentiment response: {repr(response)}")
        
        try:
            # Clean the response in case it has markdown formatting
            cleaned_response = response.strip()
            
            # Handle various markdown formats
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
            
            # Remove any additional text before/after JSON
            # Look for the first { and last }
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                cleaned_response = cleaned_response[start_idx:end_idx+1]
            
            logger.debug(f"Cleaned sentiment response: {repr(cleaned_response)}")
            
            sentiment = json.loads(cleaned_response)
            
            # Ensure the response has the correct structure and data types
            if isinstance(sentiment, dict):
                sentiment = {
                    "polarity": float(sentiment.get("polarity", 0.0)),
                    "subjectivity": float(sentiment.get("subjectivity", 0.5))
                }
            else:
                logger.error(f"Sentiment response is not a dict: {sentiment}")
                sentiment = {"polarity": 0.0, "subjectivity": 0.5}
            
            self.total_tokens += tokens
            return sentiment, tokens
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse sentiment response: {repr(response)}")
            logger.error(f"Cleaned response was: {repr(cleaned_response) if 'cleaned_response' in locals() else 'N/A'}")
            logger.error(f"JSON decode error: {e}")
            
            # Try to extract values manually if JSON parsing fails
            try:
                polarity = 0.0
                subjectivity = 0.5
                
                # Look for polarity value in the response
                if "polarity" in response.lower():
                    import re
                    polarity_match = re.search(r'"?polarity"?\s*:?\s*(-?\d*\.?\d+)', response.lower())
                    if polarity_match:
                        polarity = float(polarity_match.group(1))
                
                # Look for subjectivity value in the response
                if "subjectivity" in response.lower():
                    subjectivity_match = re.search(r'"?subjectivity"?\s*:?\s*(-?\d*\.?\d+)', response.lower())
                    if subjectivity_match:
                        subjectivity = float(subjectivity_match.group(1))
                
                logger.info(f"Manually extracted sentiment: polarity={polarity}, subjectivity={subjectivity}")
                return {"polarity": polarity, "subjectivity": subjectivity}, tokens
                
            except Exception as manual_e:
                logger.error(f"Manual extraction also failed: {manual_e}")
                return {"polarity": 0.0, "subjectivity": 0.5}, tokens
                
        except Exception as e:
            logger.error(f"Unexpected error in sentiment analysis: {e}")
            return {"polarity": 0.0, "subjectivity": 0.5}, tokens

    async def extract_topics(self, text: str) -> tuple[list[str], int]:
        prompt = TOPIC_EXTRACTION_PROMPT.format(text=text[:1500])
        response, tokens = await self.make_cohere_api_call(prompt, max_tokens=100)
        try:
            # Clean the response
            cleaned_response = response.strip()
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
            
            topics = json.loads(cleaned_response)
            
            # Ensure it's a list
            if not isinstance(topics, list):
                logger.error(f"Topics response is not a list: {topics}")
                topics = ['general']
            
            self.total_tokens += tokens
            return topics, tokens
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse topics response: {response}")
            logger.error(f"JSON decode error: {e}")
            return ['general'], tokens
        except Exception as e:
            logger.error(f"Unexpected error in topic extraction: {e}")
            return ['general'], tokens

    async def generate_initial_keywords(self, text: str) -> tuple[list[str], int]:
        """Generate initial keyword suggestions"""
        prompt = KEYWORD_GENERATION_PROMPT.format(text=text[:1500])
        response, tokens = await self.make_cohere_api_call(prompt, max_tokens=100)
        try:
            # Clean the response
            cleaned_response = response.strip()
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
            
            keywords = json.loads(cleaned_response)
            
            # Ensure it's a list
            if not isinstance(keywords, list):
                logger.error(f"Keywords response is not a list: {keywords}")
                keywords = ["content", "blog"]
            
            self.total_tokens += tokens
            return keywords, tokens
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse keywords response: {response}")
            logger.error(f"JSON decode error: {e}")
            return ["content", "blog"], tokens
        except Exception as e:
            logger.error(f"Unexpected error in keyword generation: {e}")
            return ["content", "blog"], tokens

    async def recommend_keywords(
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
        response, tokens = await self.make_cohere_api_call(prompt, max_tokens=150)
        try:
            # Clean the response
            cleaned_response = response.strip()
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
            
            keywords = json.loads(cleaned_response)
            
            # Ensure it's a list
            if not isinstance(keywords, list):
                logger.error(f"Keyword recommendations response is not a list: {keywords}")
                keywords = ["keyword", "content"]
            
            self.total_tokens += tokens
            return keywords, tokens
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse keyword recommendations: {response}")
            logger.error(f"JSON decode error: {e}")
            return ["keyword", "content"], tokens
        except Exception as e:
            logger.error(f"Unexpected error in keyword recommendations: {e}")
            return ["keyword", "content"], tokens

    async def analyze_draft(self, draft_text: str) -> tuple[dict[str, Any], int]:
        """Analyze draft quality and structure"""
        prompt = DRAFT_ANALYSIS_PROMPT.format(draft_text=draft_text[:1500])
        response, tokens = await self.make_cohere_api_call(prompt, max_tokens=200)
        try:
            # Clean the response
            cleaned_response = response.strip()
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(cleaned_response)
            
            # Ensure it's a dict with required fields
            if not isinstance(analysis, dict):
                logger.error(f"Draft analysis response is not a dict: {analysis}")
                analysis = {
                    "quality_score": 0.5,
                    "structure_notes": "Unable to analyze",
                    "improvement_areas": ["clarity"]
                }
            
            self.total_tokens += tokens
            return analysis, tokens
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse draft analysis: {response}")
            logger.error(f"JSON decode error: {e}")
            return {
                "quality_score": 0.5,
                "structure_notes": "Unable to analyze",
                "improvement_areas": ["clarity"]
            }, tokens
        except Exception as e:
            logger.error(f"Unexpected error in draft analysis: {e}")
            return {
                "quality_score": 0.5,
                "structure_notes": "Unable to analyze",
                "improvement_areas": ["clarity"]
            }, tokens

    def get_token_usage(self) -> int:
        """Get total token usage for this session"""        
        return self.total_tokens

    def reset_token_counter(self):
        """Reset token counter"""
        self.total_tokens = 0
    
    async def debug_cohere_response(self, test_prompt: str = "Analyze sentiment: This is a great day!"):
        """Debug method to see raw Cohere responses"""
        try:
            logger.info("=== DEBUGGING COHERE RESPONSE ===")
            
            # Test direct API call
            res = await self.cohere_client.chat(
                model="command-r-03-2024",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": test_prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            logger.info(f"Response type: {type(res)}")
            logger.info(f"Response dir: {[attr for attr in dir(res) if not attr.startswith('_')]}")
            logger.info(f"Full response: {res}")
            
            return res
            
        except Exception as e:
            logger.error(f"Debug error: {e}")
            logger.error(f"Error type: {type(e)}")
            return None