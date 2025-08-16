from pydantic import BaseModel,Field
class analyzeBlogsRequest(BaseModel):
    blog_texts:list[str]=Field(...,description="List of blog texts to analyze")
class userProfile(BaseModel):
    preferred_topics:list[str]=Field(...,description="User's preferred topics")
    reading_level: str=Field(...,description="User's reading level (beginner,intermediate,advanced)")
from typing import Optional
class recommendKeywordsRequest(BaseModel):
    draft_text:str=Field(...,description="Draft text to analyze")
    cursor_context: Optional[str]=Field(None,description="Context around cursor position")
    user_profile: userProfile=Field(...,description="User profile information")