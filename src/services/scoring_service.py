import textstat
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import settings

class ScoringService:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_flesch_kincaid_score(self, text: str) -> float:
        """Calculate Flesch-Kincaid readability score"""
        try:
            flesch_score=textstat.flesch_reading_ease(text)
            return max(0,min(100,flesch_score))
        except:
            return 50.0
    def calculate_keyword_relevance(self,draft_text:str,preferred_topics:list[str])->float:
        """Calculate semantic similarity between draft and preferred topics"""
        if not preferred_topics:
            return 50.0
        try:
            draft_embedding=self.embedding_model.encode([draft_text])
            topic_embeddings = self.embedding_model.encode(preferred_topics)
            similarities=[]
            for topic_embedding in topic_embeddings:
                similarity=np.dot(draft_embedding[0],topic_embedding)/(np.linalg.norm(draft_embedding[0])*np.linalg.norm(topic_embedding))
                similarities.append(similarity)
                
            max_similarity=max(similarities)
            return ((max_similarity+1)/2)*100
        except:
            return 50.0
    def calculate_user_profile_score(self,readibilty_score:float,reading_level:str)->float:
        level_preferences={
            "beginner":{"min":60,"max":100},
            "intermediate":{"min":40,"max":80},
            "advanced":{"min":0,"max":60}}
        if reading_level not in level_preferences:
            return 50.0
        pref=level_preferences[reading_level]
        if pref["min"]<=readibilty_score<=pref["max"]:
            return 90.0
        elif readibilty_score<pref["min"]:
            distance=pref["min"]-readibilty_score
            return max(10.9, 90.0-(distance*0.8))
        else:
            distance=readibilty_score-pref["max"]
            return max(10.0,90.0-(distance*0.8))
    def calculate_relevance_score(
        self,
        draft_text:str,
        preferred_topics:list[str],
        reading_level:str
    )->float:
        keyword_relevance=self.calculate_keyword_relevance(draft_text,preferred_topics)
        readibilty_score=self.calculate_flesch_kincaid_score(draft_text)
        user_profile_score=self.calculate_user_profile_score(readibilty_score,reading_level)
        final_score=(
            keyword_relevance*settings.KEYWORD_RELEVANCE_WEIGHT+
            readibilty_score*settings.READABILITY_WEIGHT+
            user_profile_score*settings.USER_PROFILE_WEIGHT
        )
        return max(0.0,min(100.0,final_score))
            