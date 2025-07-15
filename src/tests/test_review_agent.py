"""
Unit tests for ReviewAgent class.

Tests the review agent functionality including translation review,
feedback generation, error handling, and statistics calculation.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from src.agent.review_agent import (
    ReviewAgent,
    ReviewCriteria,
    ReviewFeedback,
    ReviewResult,
    ReviewRating
)
from src.agent.state import ReviewState


class TestReviewCriteria:
    """Test ReviewCriteria enum."""
    
    def test_review_criteria_values(self):
        """Test that all review criteria have correct values."""
        assert ReviewCriteria.ACCURACY.value == "accuracy"
        assert ReviewCriteria.FLUENCY.value == "fluency"
        assert ReviewCriteria.STYLE.value == "style"
        assert ReviewCriteria.CONSISTENCY.value == "consistency"
        assert ReviewCriteria.CULTURAL_ADAPTATION.value == "cultural_adaptation"


class TestReviewFeedback:
    """Test ReviewFeedback dataclass."""
    
    def test_review_feedback_creation(self):
        """Test creating a ReviewFeedback instance."""
        feedback = ReviewFeedback(
            criteria=ReviewCriteria.ACCURACY,
            score=4,
            comments="Good translation accuracy",
            suggestions=["Consider using more natural phrasing"]
        )
        
        assert feedback.criteria == ReviewCriteria.ACCURACY
        assert feedback.score == 4
        assert feedback.comments == "Good translation accuracy"
        assert feedback.suggestions == ["Consider using more natural phrasing"]


class TestReviewResult:
    """Test ReviewResult dataclass."""
    
    def test_review_result_creation(self):
        """Test creating a ReviewResult instance."""
        feedback = ReviewFeedback(
            criteria=ReviewCriteria.ACCURACY,
            score=4,
            comments="Good accuracy",
            suggestions=[]
        )
        
        result = ReviewResult(
            overall_rating=ReviewRating.GOOD,
            feedback=[feedback],
            summary="Overall good translation",
            confidence=0.8,
            requires_revision=False
        )
        
        assert result.overall_rating == ReviewRating.GOOD
        assert len(result.feedback) == 1
        assert result.summary == "Overall good translation"
        assert result.confidence == 0.8
        assert result.requires_revision is False


class TestReviewAgent:
    """Test ReviewAgent class."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.generate.return_value = self._get_sample_llm_response()
        return client
    
    @pytest.fixture
    def review_agent(self, mock_llm_client):
        """Create a ReviewAgent instance with mock LLM client."""
        return ReviewAgent(llm_client=mock_llm_client)
    
    def _get_sample_llm_response(self) -> str:
        """Get a sample LLM response for testing."""
        return '''
        {
            "overall_rating": "good",
            "confidence": 0.85,
            "requires_revision": false,
            "summary": "Good translation with minor issues",
            "feedback": [
                {
                    "criteria": "accuracy",
                    "score": 4,
                    "comments": "Translation accurately conveys the main meaning",
                    "suggestions": ["Consider using more natural Vietnamese phrasing"]
                },
                {
                    "criteria": "fluency",
                    "score": 3,
                    "comments": "Generally readable but could be more natural",
                    "suggestions": ["Improve sentence flow"]
                }
            ]
        }
        '''
    
    def test_review_agent_initialization(self, mock_llm_client):
        """Test ReviewAgent initialization."""
        agent = ReviewAgent(llm_client=mock_llm_client)
        
        assert agent.llm_client == mock_llm_client
        assert agent.config is not None
        assert "max_chunk_size" in agent.config
        assert "prompt_template" in agent.config
        assert agent.logger is not None
    
    def test_review_agent_with_custom_config(self, mock_llm_client):
        """Test ReviewAgent initialization with custom config."""
        custom_config = {
            "max_chunk_size": 500,
            "temperature": 0.5,
            "max_tokens": 1000
        }
        
        agent = ReviewAgent(llm_client=mock_llm_client, config=custom_config)
        
        assert agent.config["max_chunk_size"] == 500
        assert agent.config["temperature"] == 0.5
        assert agent.config["max_tokens"] == 1000
    
    def test_prepare_review_prompt(self, review_agent):
        """Test prompt preparation."""
        original_text = "Hello world"
        translated_text = "Xin chào thế giới"
        context = "Greeting context"
        
        prompt = review_agent._prepare_review_prompt(original_text, translated_text, context)
        
        assert original_text in prompt
        assert translated_text in prompt
        assert context in prompt
        assert "ORIGINAL TEXT" in prompt
        assert "TRANSLATED TEXT" in prompt
    
    def test_prepare_review_prompt_no_context(self, review_agent):
        """Test prompt preparation without context."""
        original_text = "Hello world"
        translated_text = "Xin chào thế giới"
        
        prompt = review_agent._prepare_review_prompt(original_text, translated_text)
        
        assert "No additional context provided" in prompt
    
    def test_generate_review_success(self, review_agent, mock_llm_client):
        """Test successful review generation."""
        prompt = "Test prompt"
        
        response = review_agent._generate_review(prompt)
        
        mock_llm_client.generate.assert_called_once_with(
            prompt=prompt,
            temperature=review_agent.config["temperature"],
            max_tokens=review_agent.config["max_tokens"]
        )
        assert response == self._get_sample_llm_response()
    
    def test_generate_review_failure(self, review_agent, mock_llm_client):
        """Test review generation failure."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        
        with pytest.raises(RuntimeError, match="Failed to generate review"):
            review_agent._generate_review("test prompt")
    
    def test_parse_review_response_valid(self, review_agent):
        """Test parsing valid review response."""
        response = self._get_sample_llm_response()
        
        data = review_agent._parse_review_response(response)
        
        assert data["overall_rating"] == "good"
        assert data["confidence"] == 0.85
        assert data["requires_revision"] is False
        assert data["summary"] == "Good translation with minor issues"
        assert len(data["feedback"]) == 2
    
    def test_parse_review_response_no_json(self, review_agent):
        """Test parsing response with no JSON."""
        response = "This is not JSON"
        
        with pytest.raises(ValueError, match="No JSON found in response"):
            review_agent._parse_review_response(response)
    
    def test_parse_review_response_invalid_json(self, review_agent):
        """Test parsing invalid JSON response."""
        response = "{ invalid json }"
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            review_agent._parse_review_response(response)
    
    def test_parse_review_response_missing_fields(self, review_agent):
        """Test parsing response with missing required fields."""
        response = '{"overall_rating": "good"}'
        
        with pytest.raises(ValueError, match="Missing required field"):
            review_agent._parse_review_response(response)
    
    def test_create_review_result_success(self, review_agent):
        """Test creating review result from valid data."""
        review_data = {
            "overall_rating": "good",
            "confidence": 0.85,
            "requires_revision": False,
            "summary": "Good translation",
            "feedback": [
                {
                    "criteria": "accuracy",
                    "score": 4,
                    "comments": "Good accuracy",
                    "suggestions": ["Improve phrasing"]
                }
            ]
        }
        
        result = review_agent._create_review_result(review_data)
        
        assert result.overall_rating == ReviewRating.GOOD
        assert result.confidence == 0.85
        assert result.requires_revision is False
        assert result.summary == "Good translation"
        assert len(result.feedback) == 1
        assert result.feedback[0].criteria == ReviewCriteria.ACCURACY
        assert result.feedback[0].score == 4
    
    def test_create_review_result_unknown_rating(self, review_agent):
        """Test creating review result with unknown rating."""
        review_data = {
            "overall_rating": "unknown_rating",
            "confidence": 0.85,
            "requires_revision": False,
            "summary": "Test",
            "feedback": []
        }
        
        result = review_agent._create_review_result(review_data)
        
        # Should default to FAIR for unknown ratings
        assert result.overall_rating == ReviewRating.FAIR
    
    def test_create_error_result(self, review_agent):
        """Test creating error result."""
        error_message = "Test error"
        
        result = review_agent._create_error_result(error_message)
        
        assert result.overall_rating == ReviewRating.POOR
        assert result.confidence == 0.0
        assert result.requires_revision is True
        assert "Test error" in result.summary
        assert len(result.feedback) == 0
    
    def test_review_translation_success(self, review_agent):
        """Test successful translation review."""
        original_text = "Hello world"
        translated_text = "Xin chào thế giới"
        
        result = review_agent.review_translation(original_text, translated_text)
        
        assert isinstance(result, ReviewResult)
        assert result.overall_rating == ReviewRating.GOOD
        assert result.confidence == 0.85
        assert result.requires_revision is False
    
    def test_review_translation_empty_inputs(self, review_agent):
        """Test translation review with empty inputs."""
        result = review_agent.review_translation("", "translated")
        
        assert result.overall_rating == ReviewRating.POOR
        assert result.requires_revision is True
        assert "non-empty" in result.summary
    
    def test_review_translation_llm_error(self, review_agent, mock_llm_client):
        """Test translation review when LLM fails."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        
        result = review_agent.review_translation("Hello", "Xin chào")
        
        assert result.overall_rating == ReviewRating.POOR
        assert result.requires_revision is True
        assert "LLM error" in result.summary
    
    def test_review_chunk_success(self, review_agent):
        """Test successful chunk review."""
        review_state = ReviewState(
            chunk_id="test_chunk_1",
            original_text="Hello world",
            translated_text="Xin chào thế giới",
            context="Test context",
            rating=ReviewRating.FAIR,
            feedback="",
            confidence=0.0,
            requires_revision=False,
            review_timestamp=datetime.now(),
            reviewer_id="test_reviewer"
        )
        
        updated_state = review_agent.review_chunk(review_state)
        
        assert updated_state["chunk_id"] == "test_chunk_1"
        assert updated_state["original_text"] == "Hello world"
        assert updated_state["translated_text"] == "Xin chào thế giới"
        assert updated_state["rating"] == ReviewRating.GOOD
        assert updated_state["confidence"] == 0.85
        assert updated_state["requires_revision"] is False
        assert "Good translation" in updated_state["feedback"]
    
    def test_review_chunk_error(self, review_agent, mock_llm_client):
        """Test chunk review with error."""
        mock_llm_client.generate.side_effect = Exception("Review error")
        
        review_state = ReviewState(
            chunk_id="test_chunk_1",
            original_text="Hello world",
            translated_text="Xin chào thế giới",
            context="Test context",
            rating=ReviewRating.FAIR,
            feedback="",
            confidence=0.0,
            requires_revision=False,
            review_timestamp=datetime.now(),
            reviewer_id="test_reviewer"
        )
        
        updated_state = review_agent.review_chunk(review_state)
        
        assert updated_state["rating"] == ReviewRating.POOR
        assert updated_state["confidence"] == 0.0
        assert updated_state["requires_revision"] is True
        assert "Review error" in updated_state["feedback"]
    
    def test_batch_review_success(self, review_agent):
        """Test successful batch review."""
        review_states = [
            ReviewState(
                chunk_id="chunk_1",
                original_text="Hello",
                translated_text="Xin chào",
                context="",
                rating=ReviewRating.FAIR,
                feedback="",
                confidence=0.0,
                requires_revision=False,
                review_timestamp=datetime.now(),
                reviewer_id="test_reviewer"
            ),
            ReviewState(
                chunk_id="chunk_2",
                original_text="World",
                translated_text="Thế giới",
                context="",
                rating=ReviewRating.FAIR,
                feedback="",
                confidence=0.0,
                requires_revision=False,
                review_timestamp=datetime.now(),
                reviewer_id="test_reviewer"
            )
        ]
        
        results = review_agent.batch_review(review_states)
        
        assert len(results) == 2
        assert all(result["rating"] == ReviewRating.GOOD for result in results)
        assert all(result["confidence"] == 0.85 for result in results)
    
    def test_batch_review_with_errors(self, review_agent, mock_llm_client):
        """Test batch review with some errors."""
        # Make second call fail
        mock_llm_client.generate.side_effect = [
            self._get_sample_llm_response(),
            Exception("Second review failed")
        ]
        
        review_states = [
            ReviewState(
                chunk_id="chunk_1",
                original_text="Hello",
                translated_text="Xin chào",
                context="",
                rating=ReviewRating.FAIR,
                feedback="",
                confidence=0.0,
                requires_revision=False,
                review_timestamp=datetime.now(),
                reviewer_id="test_reviewer"
            ),
            ReviewState(
                chunk_id="chunk_2",
                original_text="World",
                translated_text="Thế giới",
                context="",
                rating=ReviewRating.FAIR,
                feedback="",
                confidence=0.0,
                requires_revision=False,
                review_timestamp=datetime.now(),
                reviewer_id="test_reviewer"
            )
        ]
        
        results = review_agent.batch_review(review_states)
        
        assert len(results) == 2
        assert results[0]["rating"] == ReviewRating.GOOD
        assert results[1]["rating"] == ReviewRating.POOR
        assert "Second review failed" in results[1]["feedback"]
    
    def test_get_review_statistics_empty(self, review_agent):
        """Test statistics calculation with empty list."""
        stats = review_agent.get_review_statistics([])
        
        assert stats["total_reviews"] == 0
        assert stats["average_confidence"] == 0.0
        assert stats["revision_rate"] == 0.0
        assert len(stats["rating_distribution"]) == 5  # All rating types
    
    def test_get_review_statistics_success(self, review_agent):
        """Test statistics calculation with review states."""
        review_states = [
            ReviewState(
                chunk_id="chunk_1",
                original_text="Hello",
                translated_text="Xin chào",
                context="",
                rating=ReviewRating.GOOD,
                feedback="Good",
                confidence=0.8,
                requires_revision=False,
                review_timestamp=datetime.now(),
                reviewer_id="test_reviewer"
            ),
            ReviewState(
                chunk_id="chunk_2",
                original_text="World",
                translated_text="Thế giới",
                context="",
                rating=ReviewRating.POOR,
                feedback="Poor",
                confidence=0.3,
                requires_revision=True,
                review_timestamp=datetime.now(),
                reviewer_id="test_reviewer"
            ),
            ReviewState(
                chunk_id="chunk_3",
                original_text="Test",
                translated_text="Kiểm tra",
                context="",
                rating=ReviewRating.EXCELLENT,
                feedback="Excellent",
                confidence=0.9,
                requires_revision=False,
                review_timestamp=datetime.now(),
                reviewer_id="test_reviewer"
            )
        ]
        
        stats = review_agent.get_review_statistics(review_states)
        
        assert stats["total_reviews"] == 3
        assert stats["average_confidence"] == pytest.approx(0.67, rel=1e-2)
        assert stats["revision_rate"] == pytest.approx(0.333, rel=1e-2)
        assert stats["rating_distribution"]["good"] == 1
        assert stats["rating_distribution"]["poor"] == 1
        assert stats["rating_distribution"]["excellent"] == 1
        assert stats["rating_distribution"]["fair"] == 0
        assert stats["rating_distribution"]["very_poor"] == 0


class TestReviewAgentIntegration:
    """Integration tests for ReviewAgent."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for integration tests."""
        client = Mock()
        
        def generate_response(prompt, **kwargs):
            # Simulate different responses based on input
            if "Hello world" in prompt:
                return '''
                {
                    "overall_rating": "excellent",
                    "confidence": 0.95,
                    "requires_revision": false,
                    "summary": "Excellent translation",
                    "feedback": [
                        {
                            "criteria": "accuracy",
                            "score": 5,
                            "comments": "Perfect accuracy",
                            "suggestions": []
                        }
                    ]
                }
                '''
            else:
                return '''
                {
                    "overall_rating": "fair",
                    "confidence": 0.6,
                    "requires_revision": true,
                    "summary": "Fair translation needs improvement",
                    "feedback": [
                        {
                            "criteria": "fluency",
                            "score": 2,
                            "comments": "Needs improvement",
                            "suggestions": ["Improve sentence structure"]
                        }
                    ]
                }
                '''
        
        client.generate.side_effect = generate_response
        return client
    
    @pytest.fixture
    def review_agent(self, mock_llm_client):
        """Create ReviewAgent for integration tests."""
        return ReviewAgent(llm_client=mock_llm_client)
    
    def test_full_review_workflow(self, review_agent):
        """Test complete review workflow."""
        # Test excellent translation
        result1 = review_agent.review_translation(
            "Hello world",
            "Xin chào thế giới"
        )
        
        assert result1.overall_rating == ReviewRating.EXCELLENT
        assert result1.confidence == 0.95
        assert result1.requires_revision is False
        
        # Test fair translation
        result2 = review_agent.review_translation(
            "This is a test",
            "Đây là một bài kiểm tra"
        )
        
        assert result2.overall_rating == ReviewRating.FAIR
        assert result2.confidence == 0.6
        assert result2.requires_revision is True
    
    def test_review_state_integration(self, review_agent):
        """Test integration with ReviewState."""
        review_state = ReviewState(
            chunk_id="integration_test",
            original_text="Hello world",
            translated_text="Xin chào thế giới",
            context="Integration test context",
            rating=ReviewRating.FAIR,
            feedback="",
            confidence=0.0,
            requires_revision=False,
            review_timestamp=datetime.now(),
            reviewer_id="integration_tester"
        )
        
        updated_state = review_agent.review_chunk(review_state)
        
        assert updated_state["chunk_id"] == "integration_test"
        assert updated_state["rating"] == ReviewRating.EXCELLENT
        assert updated_state["confidence"] == 0.95
        assert updated_state["requires_revision"] is False
        assert "Excellent translation" in updated_state["feedback"]


class TestReviewAgentRealAPI:
    """Real API integration tests for ReviewAgent using actual Google Gemini API."""
    
    @pytest.fixture
    def real_llm_client(self):
        """Create a real LLM client using Google Gemini API."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Check if Google API key is set
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                pytest.skip("Google API key not configured. Set GOOGLE_API_KEY environment variable.")
            
            # Create real LLM client
            llm_client = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                max_tokens=2000
            )
            
            # Test the connection
            test_response = llm_client.invoke("Say 'Hello'")
            if not test_response or not test_response.content:
                pytest.skip("Could not get response from Google Gemini API")
            
            return llm_client
            
        except Exception as e:
            pytest.skip(f"Could not initialize Google Gemini client: {str(e)}")
    
    @pytest.fixture
    def real_review_agent(self, real_llm_client):
        """Create a ReviewAgent with real LLM client."""
        # Create a wrapper class to match the expected interface
        class RealLLMWrapper:
            def __init__(self, llm_client):
                self.llm_client = llm_client
            
            def generate(self, prompt, temperature=0.3, max_tokens=2000):
                from langchain_core.messages import HumanMessage
                response = self.llm_client.invoke([HumanMessage(content=prompt)])
                return response.content
        
        wrapper = RealLLMWrapper(real_llm_client)
        return ReviewAgent(llm_client=wrapper)
    
    def test_real_api_translation_review(self, real_review_agent):
        """Test translation review using real Google Gemini API."""
        original_text = "Hello world, how are you today?"
        translated_text = "Xin chào thế giới, bạn khỏe không hôm nay?"
        
        result = real_review_agent.review_translation(original_text, translated_text)
        
        # Verify the result structure
        assert isinstance(result, ReviewResult)
        assert result.overall_rating in ReviewRating
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.requires_revision, bool)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert isinstance(result.feedback, list)
        assert len(result.feedback) > 0
        
        # Verify feedback structure
        for feedback in result.feedback:
            assert isinstance(feedback, ReviewFeedback)
            assert feedback.criteria in ReviewCriteria
            assert 1 <= feedback.score <= 5
            assert isinstance(feedback.comments, str)
            assert isinstance(feedback.suggestions, list)
        
        print(f"Real API Review Result:")
        print(f"  Rating: {result.overall_rating.value}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Requires Revision: {result.requires_revision}")
        print(f"  Summary: {result.summary}")
        print(f"  Feedback Count: {len(result.feedback)}")
    
    def test_real_api_review_with_context(self, real_review_agent):
        """Test translation review with context using real API."""
        original_text = "The main character entered the room."
        translated_text = "Nhân vật chính bước vào phòng."
        context = "This is from a fantasy novel. The main character is a wizard."
        
        result = real_review_agent.review_translation(
            original_text, translated_text, context
        )
        
        assert isinstance(result, ReviewResult)
        assert result.overall_rating in ReviewRating
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        
        print(f"Real API Review with Context:")
        print(f"  Rating: {result.overall_rating.value}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Summary: {result.summary}")
    
    def test_real_api_review_state_integration(self, real_review_agent):
        """Test ReviewState integration with real API."""
        review_state = ReviewState(
            chunk_id="real_api_test",
            original_text="The weather is beautiful today.",
            translated_text="Thời tiết hôm nay thật đẹp.",
            context="Casual conversation context",
            rating=ReviewRating.FAIR,
            feedback="",
            confidence=0.0,
            requires_revision=False,
            review_timestamp=datetime.now(),
            reviewer_id="real_api_tester"
        )
        
        updated_state = real_review_agent.review_chunk(review_state)
        
        # Verify the updated state
        assert updated_state["chunk_id"] == "real_api_test"
        assert updated_state["original_text"] == "The weather is beautiful today."
        assert updated_state["translated_text"] == "Thời tiết hôm nay thật đẹp."
        assert updated_state["rating"] in ReviewRating
        assert 0.0 <= updated_state["confidence"] <= 1.0
        assert isinstance(updated_state["requires_revision"], bool)
        assert isinstance(updated_state["feedback"], str)
        assert len(updated_state["feedback"]) > 0
        
        print(f"Real API ReviewState Integration:")
        print(f"  Rating: {updated_state['rating'].value}")
        print(f"  Confidence: {updated_state['confidence']}")
        print(f"  Requires Revision: {updated_state['requires_revision']}")
        print(f"  Feedback: {updated_state['feedback'][:100]}...")
    
    def test_real_api_batch_review(self, real_review_agent):
        """Test batch review with real API."""
        review_states = [
            ReviewState(
                chunk_id="batch_1",
                original_text="Good morning!",
                translated_text="Chào buổi sáng!",
                context="Morning greeting",
                rating=ReviewRating.FAIR,
                feedback="",
                confidence=0.0,
                requires_revision=False,
                review_timestamp=datetime.now(),
                reviewer_id="real_api_tester"
            ),
            ReviewState(
                chunk_id="batch_2",
                original_text="Have a great day!",
                translated_text="Chúc bạn một ngày tốt lành!",
                context="Farewell greeting",
                rating=ReviewRating.FAIR,
                feedback="",
                confidence=0.0,
                requires_revision=False,
                review_timestamp=datetime.now(),
                reviewer_id="real_api_tester"
            )
        ]
        
        results = real_review_agent.batch_review(review_states)
        
        assert len(results) == 2
        
        for i, result in enumerate(results):
            assert result["chunk_id"] == f"batch_{i+1}"
            assert result["rating"] in ReviewRating
            assert 0.0 <= result["confidence"] <= 1.0
            assert isinstance(result["requires_revision"], bool)
            assert isinstance(result["feedback"], str)
            assert len(result["feedback"]) > 0
            
            print(f"Batch Review Result {i+1}:")
            print(f"  Rating: {result['rating'].value}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Feedback: {result['feedback'][:50]}...")
    
    def test_real_api_error_handling(self, real_review_agent):
        """Test error handling with real API."""
        # Test with empty inputs
        result = real_review_agent.review_translation("", "translated")
        
        assert result.overall_rating == ReviewRating.POOR
        assert result.confidence == 0.0
        assert result.requires_revision is True
        assert "non-empty" in result.summary
        
        # Test with very long inputs (should still work)
        long_original = "This is a very long text. " * 100
        long_translated = "Đây là một văn bản rất dài. " * 100
        
        result = real_review_agent.review_translation(long_original, long_translated)
        
        assert isinstance(result, ReviewResult)
        assert result.overall_rating in ReviewRating
        assert 0.0 <= result.confidence <= 1.0
        
        print(f"Long Text Review Result:")
        print(f"  Rating: {result.overall_rating.value}")
        print(f"  Confidence: {result.confidence}")
    
    def test_real_api_statistics(self, real_review_agent):
        """Test statistics calculation with real API results."""
        # Create review states with real API results
        review_states = []
        
        test_cases = [
            ("Hello", "Xin chào"),
            ("Goodbye", "Tạm biệt"),
            ("Thank you", "Cảm ơn bạn"),
            ("You're welcome", "Không có gì"),
            ("How are you?", "Bạn khỏe không?")
        ]
        
        for i, (original, translated) in enumerate(test_cases):
            # Get real review result
            result = real_review_agent.review_translation(original, translated)
            
            # Create review state
            review_state = ReviewState(
                chunk_id=f"stats_test_{i}",
                original_text=original,
                translated_text=translated,
                context="Statistics test",
                rating=result.overall_rating,
                feedback=result.summary,
                confidence=result.confidence,
                requires_revision=result.requires_revision,
                review_timestamp=datetime.now(),
                reviewer_id="real_api_tester"
            )
            review_states.append(review_state)
        
        # Calculate statistics
        stats = real_review_agent.get_review_statistics(review_states)
        
        assert stats["total_reviews"] == 5
        assert 0.0 <= stats["average_confidence"] <= 1.0
        assert 0.0 <= stats["revision_rate"] <= 1.0
        assert len(stats["rating_distribution"]) == 5
        
        # Verify rating distribution
        total_ratings = sum(stats["rating_distribution"].values())
        assert total_ratings == 5
        
        print(f"Real API Statistics:")
        print(f"  Total Reviews: {stats['total_reviews']}")
        print(f"  Average Confidence: {stats['average_confidence']:.3f}")
        print(f"  Revision Rate: {stats['revision_rate']:.3f}")
        print(f"  Rating Distribution: {stats['rating_distribution']}")