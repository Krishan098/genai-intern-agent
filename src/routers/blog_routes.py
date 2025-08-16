from fastapi import APIRouter, Depends, HTTPException
from typing import List
import logging

from src.auth.authentication import verify_api_key
from src.models.request_models import analyzeBlogsRequest, recommendKeywordsRequest
from src.models.response_model import blogAnalysisResponse, keywordRecommendationResponse
from src.services.llm_service import LLMService
from src.agents.blog_agent import BlogAgent
from src.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)

router = APIRouter()


_blog_agent = None

def get_blog_agent():
    global _blog_agent
    if _blog_agent is None:
        _blog_agent = BlogAgent()
    return _blog_agent

@router.post("/analyze-blogs", response_model=List[blogAnalysisResponse])
def analyze_blogs(
    request: analyzeBlogsRequest,
    api_key: str = Depends(verify_api_key)
) -> List[blogAnalysisResponse]:
    """
    Analyze multiple blog texts for sentiment, topics, and initial keywords
    """
    try:
        llm_service = LLMService()
        results = []
        
        for blog_text in request.blog_texts:
            if not blog_text.strip():
                continue
            sentiment, _ = llm_service.analyze_sentiment(blog_text)
            topics, _ = llm_service.extract_topics(blog_text)
            keywords, _ = llm_service.generate_initial_keywords(blog_text)
            
            results.append(blogAnalysisResponse(
                sentiment_metrics={
                    "polarity": sentiment.get("polarity", 0.0),
                    "subjectivity": sentiment.get("subjectivity", 0.5)
                },
                key_topics=topics,
                initial_keywords=keywords
            ))
        
        logger.info(f"Successfully analyzed {len(results)} blog texts")
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing blogs: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/recommend-keywords", response_model=keywordRecommendationResponse)
def recommend_keywords(
    request: recommendKeywordsRequest,
    api_key: str = Depends(verify_api_key)
) -> keywordRecommendationResponse:
    """
    Generate keyword recommendations using the agentic workflow
    """
    try:
        blog_agent = get_blog_agent()
        result =blog_agent.process_recommendation_request(
            draft_text=request.draft_text,
            cursor_context=request.cursor_context,
            preferred_topics=request.user_profile.preferred_topics,
            reading_level=request.user_profile.reading_level
        )
        
        if result.get("error"):
            logger.warning(f"Agent workflow completed with error: {result['error']}")
        
        response = keywordRecommendationResponse(
            suggested_keywords=result["suggested_keywords"],
            readability_score=result["readability_score"],
            relevance_score=result["relevance_score"],
            token_usage=result["token_usage"]
        )
        
        logger.info(f"Successfully generated {len(response.suggested_keywords)} keyword recommendations")
        return response
        
    except Exception as e:
        logger.error(f"Error generating keyword recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")