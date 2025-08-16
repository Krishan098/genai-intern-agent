"""
Optimized prompt templates for blog analysis and keyword recommendation.
Templates are designed to minimize token usage while maintaining accuracy.
"""

SENTIMENT_ANALYSIS_PROMPT = """Analyze sentiment of this blog text. Return JSON only:
{{"polarity": float(-1 to 1), "subjectivity": float(0 to 1)}}

Text: {text}"""

TOPIC_EXTRACTION_PROMPT = """Extract 3-5 key topics from this blog. Return JSON array of strings only:
["topic1", "topic2", ...]

Text: {text}"""

KEYWORD_GENERATION_PROMPT = """Generate 5-7 relevant keywords for this blog. Return JSON array only:
["keyword1", "keyword2", ...]

Text: {text}"""

KEYWORD_RECOMMENDATION_PROMPT = """Given:
- Draft: {draft_text}
- Context: {cursor_context}
- User topics: {preferred_topics}
- Reading level: {reading_level}
- Past analysis: {historical_data}

Generate 8-10 ranked keywords optimized for user preferences. Return JSON array only:
["keyword1", "keyword2", ...]"""

DRAFT_ANALYSIS_PROMPT = """Analyze this draft for content quality and structure. Return JSON:
{"quality_score": float(0-1), "structure_notes": "brief notes", "improvement_areas": ["area1", "area2"]}

Draft: {draft_text}"""