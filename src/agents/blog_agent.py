from typing import  Any, Optional,TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from src.services.llm_service import LLMService
from src.services.scoring_service import ScoringService
import logging

logger = logging.getLogger(__name__)

class BlogAgentState(TypedDict):
    draft_text: str
    cursor_context: Optional[str]
    preferred_topics: list[str]
    reading_level: str
    historical_data: Optional[dict]
    analysis_results: dict[str, Any]
    keywords: list[str]
    readability_score: float
    relevance_score: float
    token_usage: int
    error: Optional[str]

class BlogAgent:
    def __init__(self):
        self.llm_service = LLMService()
        self.scoring_service = ScoringService()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(BlogAgentState)
        workflow.add_node("analyze_draft", self._analyze_draft)
        workflow.add_node("generate_keywords", self._generate_keywords)
        workflow.add_node("calculate_scores", self._calculate_scores)
        workflow.add_node("format_output", self._format_output)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_edge("analyze_draft", "generate_keywords")
        workflow.add_edge("generate_keywords", "calculate_scores")
        workflow.add_edge("calculate_scores", "format_output")
        workflow.add_edge("format_output", END)
        workflow.add_edge("handle_error", END)
        workflow.set_entry_point("analyze_draft")
        
        return workflow.compile()
    
    async def _analyze_draft(self, state: BlogAgentState) -> BlogAgentState:
        """Analyze the draft text for quality and structure"""
        try:
            logger.info("Analyzing draft text")
            analysis, tokens = await self.llm_service.analyze_draft(state["draft_text"])
            state["analysis_results"] = analysis
            state["token_usage"] += tokens
            logger.info(f"Draft analysis completed. Tokens used: {tokens}")
        except Exception as e:
            logger.error(f"Error analyzing draft: {e}")
            state["error"] = f"Draft analysis failed: {str(e)}"
            # Set default values
            state["analysis_results"] = {
                "quality_score": 0.5,
                "structure_notes": "Unable to analyze",
                "improvement_areas": ["clarity"]
            }
        
        return state
    
    async def _generate_keywords(self, state: BlogAgentState) -> BlogAgentState:
        """Generate keyword recommendations"""
        if state.get("error"):
            return state
            
        try:
            logger.info("Generating keyword recommendations")
            keywords, tokens = await self.llm_service.recommend_keywords(
                state["draft_text"],
                state["cursor_context"],
                state["preferred_topics"],
                state["reading_level"],
                state["historical_data"]
            )
            state["keywords"] = keywords
            state["token_usage"] += tokens
            logger.info(f"Keywords generated: {len(keywords)} keywords. Tokens used: {tokens}")
        except Exception as e:
            logger.error(f"Error generating keywords: {e}")
            state["error"] = f"Keyword generation failed: {str(e)}"
            state["keywords"] = ["content", "blog", "article"]
        
        return state
    
    async def _calculate_scores(self, state: BlogAgentState) -> BlogAgentState:
        """Calculate readability and relevance scores"""
        if state.get("error"):
            return state
            
        try:
            logger.info("Calculating scores")
            state["readability_score"] = self.scoring_service.calculate_flesch_kincaid_score(
                state["draft_text"]
            )
            state["relevance_score"] = self.scoring_service.calculate_relevance_score(
                state["draft_text"],
                state["preferred_topics"],
                state["reading_level"]
            )
            logger.info(f"Scores calculated - Readability: {state['readability_score']:.2f}, Relevance: {state['relevance_score']:.2f}")
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
            state["error"] = f"Score calculation failed: {str(e)}"
            state["readability_score"] = 50.0
            state["relevance_score"] = 50.0
        
        return state
    
    async def _format_output(self, state: BlogAgentState) -> BlogAgentState:
        """Format the final output"""
        logger.info("Formatting output")
        if not state.get("keywords"):
            state["keywords"] = ["content", "blog"]
        if state.get("readability_score", 0) == 0:
            state["readability_score"] = 50.0
        if state.get("relevance_score", 0) == 0:
            state["relevance_score"] = 50.0
        return state
    
    async def _handle_error(self, state: BlogAgentState) -> BlogAgentState:
        """Handle errors by providing default values"""
        logger.warning(f"Handling error: {state.get('error', 'Unknown error')}")
        if not state.get("keywords"):
            state["keywords"] = ["content", "blog", "article"]
        if state.get("readability_score", 0) == 0:
            state["readability_score"] = 50.0
        if state.get("relevance_score", 0) == 0:
            state["relevance_score"] = 50.0
        return state
    
    async def process_recommendation_request(
        self,
        draft_text: str,
        cursor_context: Optional[str],
        preferred_topics: list[str],
        reading_level: str,
        historical_data: Optional[dict] = None
    ) -> dict[str, Any]:
        """Process a keyword recommendation request through the agent workflow"""
        # Initialize state
        state: BlogAgentState = {
            "draft_text": draft_text,
            "cursor_context": cursor_context,
            "preferred_topics": preferred_topics,
            "reading_level": reading_level,
            "historical_data": historical_data,
            "analysis_results": {},
            "keywords": [],
            "readability_score": 0.0,
            "relevance_score": 0.0,
            "token_usage": 0,
            "error": None
        }
        
        # Reset token counter
        self.llm_service.reset_token_counter()
        
        # Run the workflow
        try:
            final_state = await self.graph.ainvoke(state)
            
            return {
                "suggested_keywords": final_state["keywords"],
                "readability_score": final_state["readability_score"],
                "relevance_score": final_state["relevance_score"],
                "token_usage": final_state["token_usage"],
                "error": final_state.get("error")
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "suggested_keywords": ["content", "blog"],
                "readability_score": 50.0,
                "relevance_score": 50.0,
                "token_usage": 0,
                "error": str(e)
            }