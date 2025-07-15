"""
Review Agent for Translation Quality Assessment

This module provides a ReviewAgent class that evaluates translation quality,
provides feedback, and generates ratings for translated text chunks.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .state import ReviewState, ReviewRating
from .utils import VietnameseTextChunker


class ReviewCriteria(Enum):
    """Review criteria for translation quality assessment."""
    ACCURACY = "accuracy"
    FLUENCY = "fluency"
    STYLE = "style"
    CONSISTENCY = "consistency"
    CULTURAL_ADAPTATION = "cultural_adaptation"


@dataclass
class ReviewFeedback:
    """Structured feedback for translation review."""
    criteria: ReviewCriteria
    score: int  # 1-5 scale
    comments: str
    suggestions: List[str]


@dataclass
class ReviewResult:
    """Complete review result for a translation chunk."""
    overall_rating: ReviewRating
    feedback: List[ReviewFeedback]
    summary: str
    confidence: float  # 0.0-1.0
    requires_revision: bool


class ReviewAgent:
    """
    Agent responsible for reviewing translation quality and providing feedback.
    
    This agent uses LLM integration to evaluate translations based on multiple
    criteria and generate structured feedback for improvement.
    """
    
    def __init__(
        self,
        llm_client: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ReviewAgent.
        
        Args:
            llm_client: LLM client for generating reviews
            config: Configuration dictionary for review settings
        """
        self.llm_client = llm_client
        self.config = config or self._get_default_config()
        self.chunker = VietnameseTextChunker()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the review agent."""
        return {
            "max_chunk_size": 1000,  # Maximum words per review chunk
            "min_confidence_threshold": 0.7,
            "review_criteria": [criteria.value for criteria in ReviewCriteria],
            "rating_scale": 5,
            "prompt_template": self._get_default_prompt_template(),
            "temperature": 0.3,
            "max_tokens": 2000
        }
    
    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for review generation."""
        return """
You are a professional Vietnamese translation reviewer. Your task is to evaluate the quality of a Vietnamese translation and provide detailed feedback.

ORIGINAL TEXT (English):
{original_text}

TRANSLATED TEXT (Vietnamese):
{translated_text}

CONTEXT (if available):
{context}

Please evaluate the translation based on the following criteria:

1. ACCURACY (1-5): How accurately does the translation convey the original meaning?
2. FLUENCY (1-5): How natural and readable is the Vietnamese text?
3. STYLE (1-5): How well does the translation match the original style and tone?
4. CONSISTENCY (1-5): How consistent is the terminology and style throughout?
5. CULTURAL_ADAPTATION (1-5): How well is the content adapted for Vietnamese culture?

For each criterion, provide:
- Score (1-5, where 1=poor, 5=excellent)
- Specific comments explaining the score
- Concrete suggestions for improvement

Respond in the following JSON format:
{{
    "overall_rating": "excellent|good|fair|poor|very_poor",
    "confidence": 0.85,
    "requires_revision": false,
    "summary": "Brief overall assessment",
    "feedback": [
        {{
            "criteria": "accuracy",
            "score": 4,
            "comments": "Translation accurately conveys the main meaning...",
            "suggestions": ["Consider using more natural Vietnamese phrasing..."]
        }}
    ]
}}

Focus on providing constructive, actionable feedback that will help improve the translation quality.
"""
    
    def review_translation(
        self,
        original_text: str,
        translated_text: str,
        context: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> ReviewResult:
        """
        Review a translation and generate feedback.
        
        Args:
            original_text: Original English text
            translated_text: Translated Vietnamese text
            context: Additional context for the translation
            chunk_id: Optional identifier for the chunk being reviewed
            
        Returns:
            ReviewResult containing rating, feedback, and recommendations
        """
        try:
            # Validate inputs
            if not original_text.strip() or not translated_text.strip():
                raise ValueError("Both original and translated text must be non-empty")
            
            # Prepare prompt
            prompt = self._prepare_review_prompt(original_text, translated_text, context)
            
            # Generate review using LLM
            response = self._generate_review(prompt)
            
            # Parse and validate response
            review_data = self._parse_review_response(response)
            
            # Create ReviewResult
            result = self._create_review_result(review_data)
            
            self.logger.info(f"Review completed for chunk {chunk_id}: {result.overall_rating.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error reviewing translation: {e}")
            return self._create_error_result(str(e))
    
    def review_chunk(
        self,
        review_state: ReviewState
    ) -> ReviewState:
        """
        Review a specific chunk and update the review state.
        
        Args:
            review_state: Current review state for the chunk
            
        Returns:
            Updated review state with review results
        """
        import logging
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"ReviewAgent input: original_text=\n{review_state['original_text']}\n---\ntranslated_text=\n{review_state['translated_text']}\n---")
            if not review_state['original_text'] or not review_state['translated_text']:
                raise ValueError("Both original and translated text must be non-empty")
            # Perform the review
            result = self.review_translation(
                original_text=review_state["original_text"],
                translated_text=review_state["translated_text"],
                context=review_state.get("context"),
                chunk_id=review_state["chunk_id"]
            )
            
            # Update the review state
            updated_state = ReviewState(
                chunk_id=review_state["chunk_id"],
                original_text=review_state["original_text"],
                translated_text=review_state["translated_text"],
                context=review_state.get("context"),
                rating=result.overall_rating,
                feedback=result.summary,
                confidence=result.confidence,
                requires_revision=result.requires_revision,
                review_timestamp=review_state.get("review_timestamp"),
                reviewer_id=review_state.get("reviewer_id")
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error reviewing chunk {review_state['chunk_id']}: {e}")
            # Return state with error information
            return ReviewState(
                chunk_id=review_state["chunk_id"],
                original_text=review_state["original_text"],
                translated_text=review_state["translated_text"],
                context=review_state.get("context"),
                rating=ReviewRating.POOR,
                feedback=f"Review failed: {str(e)}",
                confidence=0.0,
                requires_revision=True,
                review_timestamp=review_state.get("review_timestamp"),
                reviewer_id=review_state.get("reviewer_id")
            )
    
    def batch_review(
        self,
        review_states: List[ReviewState]
    ) -> List[ReviewState]:
        """
        Review multiple chunks in batch.
        
        Args:
            review_states: List of review states to process
            
        Returns:
            List of updated review states
        """
        results = []
        
        for review_state in review_states:
            try:
                result = self.review_chunk(review_state)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in batch review for chunk {review_state['chunk_id']}: {e}")
                # Add error state
                error_state = ReviewState(
                    chunk_id=review_state["chunk_id"],
                    original_text=review_state["original_text"],
                    translated_text=review_state["translated_text"],
                    context=review_state.get("context"),
                    rating=ReviewRating.POOR,
                    feedback=f"Batch review failed: {str(e)}",
                    confidence=0.0,
                    requires_revision=True,
                    review_timestamp=review_state.get("review_timestamp"),
                    reviewer_id=review_state.get("reviewer_id")
                )
                results.append(error_state)
        
        return results
    
    def _prepare_review_prompt(
        self,
        original_text: str,
        translated_text: str,
        context: Optional[str] = None
    ) -> str:
        """Prepare the review prompt with the given texts."""
        template = self.config["prompt_template"]
        context_text = context or "No additional context provided."
        
        return template.format(
            original_text=original_text,
            translated_text=translated_text,
            context=context_text
        )
    
    def _generate_review(self, prompt: str) -> str:
        """Generate review using the LLM client."""
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm_client.invoke([
                HumanMessage(content=prompt)
            ])
            return response.content
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"Failed to generate review: {e}")
    
    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["overall_rating", "confidence", "requires_revision", "summary", "feedback"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            self.logger.error(f"Failed to parse review response: {e}")
            raise ValueError(f"Failed to parse response: {e}")
    
    def _create_review_result(self, review_data: Dict[str, Any]) -> ReviewResult:
        """Create a ReviewResult from parsed review data."""
        try:
            # Parse overall rating
            rating_str = review_data["overall_rating"].lower()
            rating_map = {
                "excellent": ReviewRating.EXCELLENT,
                "good": ReviewRating.GOOD,
                "fair": ReviewRating.FAIR,
                "poor": ReviewRating.POOR,
                "very_poor": ReviewRating.VERY_POOR
            }
            overall_rating = rating_map.get(rating_str, ReviewRating.FAIR)
            
            # Parse feedback
            feedback_list = []
            for fb_data in review_data["feedback"]:
                criteria = ReviewCriteria(fb_data["criteria"])
                feedback = ReviewFeedback(
                    criteria=criteria,
                    score=fb_data["score"],
                    comments=fb_data["comments"],
                    suggestions=fb_data.get("suggestions", [])
                )
                feedback_list.append(feedback)
            
            return ReviewResult(
                overall_rating=overall_rating,
                feedback=feedback_list,
                summary=review_data["summary"],
                confidence=float(review_data["confidence"]),
                requires_revision=bool(review_data["requires_revision"])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create review result: {e}")
            raise ValueError(f"Invalid review data structure: {e}")
    
    def _create_error_result(self, error_message: str) -> ReviewResult:
        """Create an error result when review fails."""
        return ReviewResult(
            overall_rating=ReviewRating.POOR,
            feedback=[],
            summary=f"Review failed: {error_message}",
            confidence=0.0,
            requires_revision=True
        )
    
    def get_review_statistics(
        self,
        review_states: List[ReviewState]
    ) -> Dict[str, Any]:
        """
        Generate statistics from a list of review states.
        
        Args:
            review_states: List of review states to analyze
            
        Returns:
            Dictionary containing review statistics
        """
        if not review_states:
            return {
                "total_reviews": 0,
                "average_confidence": 0.0,
                "rating_distribution": {rating.value: 0 for rating in ReviewRating},
                "revision_rate": 0.0
            }
        
        total_reviews = len(review_states)
        total_confidence = sum(state["confidence"] for state in review_states)
        average_confidence = total_confidence / total_reviews
        
        # Rating distribution
        rating_counts = {rating.value: 0 for rating in ReviewRating}
        
        for state in review_states:
            if state.get("rating"):
                rating_counts[state["rating"].value] += 1
        
        # Revision rate
        revisions_needed = sum(1 for state in review_states if state["requires_revision"])
        revision_rate = revisions_needed / total_reviews
        
        return {
            "total_reviews": total_reviews,
            "average_confidence": average_confidence,
            "rating_distribution": rating_counts,
            "revision_rate": revision_rate
        } 