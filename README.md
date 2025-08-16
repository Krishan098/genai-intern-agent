# GenAI Intern Agent

An agentic blog support system built with FastAPI, LangGraph, and OpenAI GPT-4o for intelligent blog analysis and keyword recommendation.

## Features

- **Blog Analysis**: Sentiment analysis, topic extraction, and initial keyword generation
- **Intelligent Keyword Recommendations**: Context-aware keyword suggestions using LangGraph agents
- **Advanced Scoring System**: Multi-factor relevance scoring combining semantic similarity, readability, and user preferences
- **Retry Mechanism**: Exponential backoff for robust API calls
- **Token Usage Tracking**: Monitor and optimize LLM token consumption

## Quick Start

### Prerequisites

- Python 3.9+
- Cohere API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genai-intern-agent.git
cd genai-intern-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env
# Edit .env and add your OPENAI_API_KEY
```

4. Run the application:
```bash
python -m src.main
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## API Endpoints

### 1. Analyze Blogs
**POST** `/api/analyze-blogs`

Analyze multiple blog texts for sentiment, topics, and keywords.

**Request Body:**
```json
{
  "blog_texts": ["Blog content 1", "Blog content 2"]
}
```

**Response:**
```json
[
  {
    "sentiment_metrics": {
      "polarity": 0.8,
      "subjectivity": 0.5
    },
    "key_topics": ["topic1", "topic2"],
    "initial_keywords": ["keyword1", "keyword2"]
  }
]
```

### 2. Recommend Keywords
**POST** `/api/recommend-keywords`

Generate intelligent keyword recommendations using the agentic workflow.

**Request Body:**
```json
{
  "draft_text": "Your draft content here",
  "cursor_context": "Optional context",
  "user_profile": {
    "preferred_topics": ["AI", "technology"],
    "reading_level": "intermediate"
  }
}
```

**Response:**
```json
{
  "suggested_keywords": ["keyword1", "keyword2"],
  "readability_score": 45.2,
  "relevance_score": 87.3,
  "token_usage": 245
}
```

## Authentication

All endpoints require API key authentication via the Authorization header:
```
Authorization: Bearer your-api-key
```

Default API key: `genai-intern-2024`

## Architecture

- **FastAPI**: Modern web framework for building APIs
- **LangGraph**: Stateful agent workflow orchestration
- **Sentence Transformers**: Semantic similarity calculations
- **TextStat**: Readability analysis

## Project Structure

```
src/
├── main.py                 # FastAPI application entry point
├── auth/                   # API key authentication
├── models/                 # Pydantic request/response models
├── services/              # Core business logic
├── agents/                # LangGraph agent workflows
├── routers/               # API route handlers
└── utils/                 # Configuration and utilities
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black src/
isort src/
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| COHERE_API_KEY | COHERE API key | Required |
| API_KEY | API authentication key | genai-intern-2024 |
| MAX_TOKENS | Max tokens per request | 2000 |
| TEMPERATURE | Model temperature | 0.3 |

