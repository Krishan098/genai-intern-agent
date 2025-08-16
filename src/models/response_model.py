from pydantic import BaseModel
class sentimentMetrics(BaseModel):
    polarity: float
    subjectivity: float
class blogAnalysisResponse(BaseModel):
    sentiment_metrics: sentimentMetrics
    key_topics: list[str]
    initial_keywords:list[str]
class keywordRecommendationResponse(BaseModel):
    suggested_keywords:list[str]
    readibility_score:float
    relevance_score:float
    token_usage: int
    