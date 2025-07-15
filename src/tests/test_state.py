import pytest
from typing import Dict, Any, List
from src.agent.state import (
    TranslationState,
    MemoryState,
    ChunkState,
    BigChunkState,
    SmallChunkState,
    ReviewState,
    RetryState,
    OverallState,
    KnowledgeNode,
    MemoryOperation,
    TranslationOutput,
    validate_translation_state,
    validate_memory_state,
    validate_chunk_state,
    validate_big_chunk_state,
    validate_small_chunk_state,
    validate_review_state,
    validate_retry_state,
    validate_overall_state,
    validate_knowledge_node,
    create_initial_state,
    is_state_valid,
    get_state_validation_errors
)


class TestTranslationState:
    """Test cases for TranslationState"""
    
    def test_valid_translation_state(self):
        """Test valid TranslationState creation and validation"""
        state = TranslationState(
            chunk_id="chunk_001",
            original_text="Đây là văn bản gốc.",
            translated_text="This is the original text.",
            memory_context=[],
            translation_quality="good",
            processing_status="completed",
            error_message=None
        )
        
        errors = validate_translation_state(state)
        assert len(errors) == 0
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        state = TranslationState(
            chunk_id="",
            original_text="",
            translated_text=None,
            memory_context=[],
            translation_quality=None,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_translation_state(state)
        assert len(errors) == 2
        assert any("chunk_id is required" in error for error in errors)
        assert any("original_text is required" in error for error in errors)
    
    def test_invalid_processing_status(self):
        """Test validation with invalid processing status"""
        state = TranslationState(
            chunk_id="chunk_001",
            original_text="Test text",
            translated_text=None,
            memory_context=[],
            translation_quality=None,
            processing_status="invalid_status",
            error_message=None
        )
        
        errors = validate_translation_state(state)
        assert len(errors) == 1
        assert "processing_status must be one of" in errors[0]
    
    def test_valid_processing_statuses(self):
        """Test all valid processing statuses"""
        valid_statuses = ["pending", "processing", "completed", "failed"]
        
        for status in valid_statuses:
            state = TranslationState(
                chunk_id="chunk_001",
                original_text="Test text",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status=status,
                error_message=None
            )
            
            errors = validate_translation_state(state)
            assert len(errors) == 0
    
    def test_invalid_memory_context(self):
        """Test validation with invalid memory context"""
        state = TranslationState(
            chunk_id="chunk_001",
            original_text="Test text",
            translated_text=None,
            memory_context="not_a_list",  #type: ignore # Should be a list
            translation_quality=None,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_translation_state(state)
        assert len(errors) == 1
        assert "memory_context must be a list" in errors[0]


class TestMemoryState:
    """Test cases for MemoryState"""
    
    def test_valid_memory_state(self):
        """Test valid MemoryState creation and validation"""
        state = MemoryState(
            retrieved_nodes=[],
            created_nodes=[],
            updated_nodes=[],
            search_queries=[],
            memory_operations=[]
        )
        
        errors = validate_memory_state(state)
        assert len(errors) == 0
    
    def test_memory_state_with_operations(self):
        """Test MemoryState with actual memory operations"""
        operations = [
            {
                "operation_type": "search",
                "node_type": "GlossaryTerm",
                "query_or_content": "test query",
                "result": {"found": True},
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ]
        
        state = MemoryState(
            retrieved_nodes=[{"id": "1", "content": "test"}],
            created_nodes=[],
            updated_nodes=[],
            search_queries=["test query"],
            memory_operations=operations
        )
        
        errors = validate_memory_state(state)
        assert len(errors) == 0
    
    def test_invalid_list_fields(self):
        """Test validation with invalid list fields"""
        state = MemoryState(
            retrieved_nodes="not_a_list",  #type: ignore # Should be a list
            created_nodes=[],
            updated_nodes=[],
            search_queries=[],
            memory_operations=[]
        )
        
        errors = validate_memory_state(state)
        assert len(errors) == 1
        assert "retrieved_nodes must be a list" in errors[0]
    
    def test_invalid_memory_operations(self):
        """Test validation with invalid memory operations"""
        state = MemoryState(
            retrieved_nodes=[],
            created_nodes=[],
            updated_nodes=[],
            search_queries=[],
            memory_operations=["not_a_dict"]  #type: ignore  # Should be a list of dicts
        )
        
        errors = validate_memory_state(state)
        assert len(errors) == 1
        assert "memory_operations must contain dictionaries" in errors[0]
    
    def test_missing_operation_fields(self):
        """Test validation with missing operation fields"""
        state = MemoryState(
            retrieved_nodes=[],
            created_nodes=[],
            updated_nodes=[],
            search_queries=[],
            memory_operations=[{"operation_type": "search"}]  # Missing required fields
        )
        
        errors = validate_memory_state(state)
        assert len(errors) > 0
        assert any("must contain" in error for error in errors)


class TestChunkState:
    """Test cases for ChunkState"""
    
    def test_valid_chunk_state(self):
        """Test valid ChunkState creation and validation"""
        state = ChunkState(
            chunk_index=0,
            chunk_text="Test chunk text",
            chunk_size=15,
            is_processed=False,
            translation_attempts=0,
            max_attempts=3
        )
        
        errors = validate_chunk_state(state)
        assert len(errors) == 0
    
    def test_invalid_chunk_index(self):
        """Test validation with invalid chunk index"""
        state = ChunkState(
            chunk_index="not_an_int",  #type: ignore  # Should be an integer
            chunk_text="Test text",
            chunk_size=10,
            is_processed=False,
            translation_attempts=0,
            max_attempts=3
        )
        
        errors = validate_chunk_state(state)
        assert len(errors) == 1
        assert "chunk_index must be an integer" in errors[0]
    
    def test_empty_chunk_text(self):
        """Test validation with empty chunk text"""
        state = ChunkState(
            chunk_index=0,
            chunk_text="",  # Should not be empty
            chunk_size=0,
            is_processed=False,
            translation_attempts=0,
            max_attempts=3
        )
        
        errors = validate_chunk_state(state)
        assert len(errors) == 2  # Both empty text and invalid chunk size
        assert any("chunk_text is required" in error for error in errors)
    
    def test_invalid_chunk_size(self):
        """Test validation with invalid chunk size"""
        state = ChunkState(
            chunk_index=0,
            chunk_text="Test text",
            chunk_size=-1,  # Should be positive
            is_processed=False,
            translation_attempts=0,
            max_attempts=3
        )
        
        errors = validate_chunk_state(state)
        assert len(errors) == 1
        assert "chunk_size must be a positive integer" in errors[0]
    
    def test_invalid_boolean_fields(self):
        """Test validation with invalid boolean fields"""
        state = ChunkState(
            chunk_index=0,
            chunk_text="Test text",
            chunk_size=10,
            is_processed="not_a_bool",  #type: ignore  # Should be a boolean
            translation_attempts=0,
            max_attempts=3
        )
        
        errors = validate_chunk_state(state)
        assert len(errors) == 1
        assert "is_processed must be a boolean" in errors[0]
    
    def test_invalid_attempts(self):
        """Test validation with invalid attempts"""
        state = ChunkState(
            chunk_index=0,
            chunk_text="Test text",
            chunk_size=10,
            is_processed=False,
            translation_attempts="not_an_int",  #type: ignore  # Should be an integer
            max_attempts=3
        )
        
        errors = validate_chunk_state(state)
        assert len(errors) == 1
        assert "translation_attempts must be a non-negative integer" in errors[0]
    
    def test_attempts_exceed_max(self):
        """Test validation when attempts exceed max attempts"""
        state = ChunkState(
            chunk_index=0,
            chunk_text="Test text",
            chunk_size=10,
            is_processed=False,
            translation_attempts=5,
            max_attempts=3
        )
        
        errors = validate_chunk_state(state)
        assert len(errors) == 1
        assert "translation_attempts cannot exceed max_attempts" in errors[0]


class TestBigChunkState:
    """Test cases for BigChunkState"""
    
    def test_valid_big_chunk_state(self):
        """Test valid BigChunkState creation and validation"""
        state = BigChunkState(
            big_chunk_id="big_chunk_001",
            big_chunk_text="This is a big chunk of text for testing purposes.",
            big_chunk_size=16000,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_big_chunk_state(state)
        assert len(errors) == 0
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        state = BigChunkState(
            big_chunk_id="",
            big_chunk_text="",
            big_chunk_size=16000,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_big_chunk_state(state)
        assert len(errors) == 2
        assert any("big_chunk_id is required" in error for error in errors)
        assert any("big_chunk_text is required" in error for error in errors)
    
    def test_invalid_big_chunk_size(self):
        """Test validation with invalid big chunk size"""
        state = BigChunkState(
            big_chunk_id="big_chunk_001",
            big_chunk_text="Test text",
            big_chunk_size=-1,  # Should be positive
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_big_chunk_state(state)
        assert len(errors) == 1
        assert "big_chunk_size must be a positive integer" in errors[0]
    
    def test_invalid_processing_status(self):
        """Test validation with invalid processing status"""
        state = BigChunkState(
            big_chunk_id="big_chunk_001",
            big_chunk_text="Test text",
            big_chunk_size=16000,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="invalid_status",
            error_message=None
        )
        
        errors = validate_big_chunk_state(state)
        assert len(errors) == 1
        assert "processing_status must be one of" in errors[0]
    
    def test_big_chunk_with_small_chunks(self):
        """Test BigChunkState with valid small chunks"""
        small_chunk = SmallChunkState(
            small_chunk_id="small_chunk_001",
            big_chunk_id="big_chunk_001",
            small_chunk_text="Small chunk text",
            small_chunk_size=500,
            position_in_big_chunk=0,
            translated_text=None,
            recent_context=[],
            translation_attempts=0,
            max_attempts=3,
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        state = BigChunkState(
            big_chunk_id="big_chunk_001",
            big_chunk_text="Big chunk text",
            big_chunk_size=16000,
            memory_context=[],
            small_chunks=[small_chunk],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_big_chunk_state(state)
        assert len(errors) == 0


class TestSmallChunkState:
    """Test cases for SmallChunkState"""
    
    def test_valid_small_chunk_state(self):
        """Test valid SmallChunkState creation and validation"""
        state = SmallChunkState(
            small_chunk_id="small_chunk_001",
            big_chunk_id="big_chunk_001",
            small_chunk_text="This is a small chunk of text.",
            small_chunk_size=500,
            position_in_big_chunk=0,
            translated_text=None,
            recent_context=[],
            translation_attempts=0,
            max_attempts=3,
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_small_chunk_state(state)
        assert len(errors) == 0
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        state = SmallChunkState(
            small_chunk_id="",
            big_chunk_id="",
            small_chunk_text="",
            small_chunk_size=500,
            position_in_big_chunk=0,
            translated_text=None,
            recent_context=[],
            translation_attempts=0,
            max_attempts=3,
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_small_chunk_state(state)
        assert len(errors) == 3
        assert any("small_chunk_id is required" in error for error in errors)
        assert any("big_chunk_id is required" in error for error in errors)
        assert any("small_chunk_text is required" in error for error in errors)
    
    def test_invalid_position_in_big_chunk(self):
        """Test validation with invalid position"""
        state = SmallChunkState(
            small_chunk_id="small_chunk_001",
            big_chunk_id="big_chunk_001",
            small_chunk_text="Test text",
            small_chunk_size=500,
            position_in_big_chunk=-1,  # Should be non-negative
            translated_text=None,
            recent_context=[],
            translation_attempts=0,
            max_attempts=3,
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_small_chunk_state(state)
        assert len(errors) == 1
        assert "position_in_big_chunk must be a non-negative integer" in errors[0]
    
    def test_small_chunk_with_translation(self):
        """Test SmallChunkState with translated text"""
        state = SmallChunkState(
            small_chunk_id="small_chunk_001",
            big_chunk_id="big_chunk_001",
            small_chunk_text="Vietnamese text",
            small_chunk_size=500,
            position_in_big_chunk=0,
            translated_text="English translation",
            recent_context=[],
            translation_attempts=1,
            max_attempts=3,
            is_processed=True,
            processing_status="completed",
            error_message=None
        )
        
        errors = validate_small_chunk_state(state)
        assert len(errors) == 0


class TestReviewState:
    """Test cases for ReviewState"""
    
    def test_valid_review_state(self):
        """Test valid ReviewState creation and validation"""
        state = ReviewState( # type: ignore
            review_id="review_001",
            small_chunk_id="small_chunk_001",
            translated_text="This is the translated text to review.",
            rating=8.5,
            feedback="Good translation with minor improvements needed.",
            review_status="completed",
            reviewer_model="gpt-4",
            review_timestamp="2024-01-01T00:00:00Z",
            error_message=None
        )
        
        errors = validate_review_state(state)
        assert len(errors) == 0
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        state = ReviewState( # type: ignore
            review_id="",
            small_chunk_id="",
            translated_text="",
            rating=None,
            feedback=None,
            review_status="pending",
            reviewer_model="",
            review_timestamp=None,
            error_message=None
        )
        
        errors = validate_review_state(state)
        assert len(errors) == 4
        assert any("review_id is required" in error for error in errors)
        assert any("small_chunk_id is required" in error for error in errors)
        assert any("translated_text is required" in error for error in errors)
        assert any("reviewer_model is required" in error for error in errors)
    
    def test_invalid_rating(self):
        """Test validation with invalid rating"""
        state = ReviewState( # type: ignore
            review_id="review_001",
            small_chunk_id="small_chunk_001",
            translated_text="Test translation",
            rating=15.0,  # Should be between 0.0 and 10.0
            feedback=None,
            review_status="completed",
            reviewer_model="gpt-4",
            review_timestamp=None,
            error_message=None
        )
        
        errors = validate_review_state(state)
        assert len(errors) == 1
        assert "rating must be a number between 0.0 and 10.0" in errors[0]
    
    def test_valid_rating_range(self):
        """Test validation with valid rating range"""
        valid_ratings = [0.0, 5.0, 7.0, 10.0]
        
        for rating in valid_ratings:
            state = ReviewState(    # type: ignore
                review_id="review_001",
                small_chunk_id="small_chunk_001",
                translated_text="Test translation",
                rating=rating,
                feedback=None,
                review_status="completed",
                reviewer_model="gpt-4",
                review_timestamp=None,
                error_message=None
            )
            
            errors = validate_review_state(state)
            assert len(errors) == 0
    
    def test_invalid_review_status(self):
        """Test validation with invalid review status"""
        state = ReviewState(  # type: ignore
            review_id="review_001",
            small_chunk_id="small_chunk_001",
            translated_text="Test translation",
            rating=None,
            feedback=None,
            review_status="invalid_status",
            reviewer_model="gpt-4",
            review_timestamp=None,
            error_message=None
        )
        
        errors = validate_review_state(state)
        assert len(errors) == 1
        assert "review_status must be one of" in errors[0]


class TestRetryState:
    """Test cases for RetryState"""
    
    def test_valid_retry_state(self):
        """Test valid RetryState creation and validation"""
        state = RetryState(
            retry_id="retry_001",
            small_chunk_id="small_chunk_001",
            original_translation="Original failed translation",
            feedback="Translation needs improvement in accuracy",
            retry_attempt=1,
            max_retries=3,
            new_translation="Improved translation",
            retry_status="completed",
            error_message=None
        )
        
        errors = validate_retry_state(state)
        assert len(errors) == 0
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        state = RetryState(
            retry_id="",
            small_chunk_id="",
            original_translation="",
            feedback="",
            retry_attempt=0,
            max_retries=3,
            new_translation=None,
            retry_status="pending",
            error_message=None
        )
        
        errors = validate_retry_state(state)
        assert len(errors) == 4
        assert any("retry_id is required" in error for error in errors)
        assert any("small_chunk_id is required" in error for error in errors)
        assert any("original_translation is required" in error for error in errors)
        assert any("feedback is required" in error for error in errors)
    
    def test_invalid_retry_attempt(self):
        """Test validation with invalid retry attempt"""
        state = RetryState(
            retry_id="retry_001",
            small_chunk_id="small_chunk_001",
            original_translation="Original translation",
            feedback="Feedback",
            retry_attempt=-1,  # Should be non-negative
            max_retries=3,
            new_translation=None,
            retry_status="pending",
            error_message=None
        )
        
        errors = validate_retry_state(state)
        assert len(errors) == 1
        assert "retry_attempt must be a non-negative integer" in errors[0]
    
    def test_retry_attempt_exceeds_max(self):
        """Test validation when retry attempt exceeds max retries"""
        state = RetryState(
            retry_id="retry_001",
            small_chunk_id="small_chunk_001",
            original_translation="Original translation",
            feedback="Feedback",
            retry_attempt=5,
            max_retries=3,
            new_translation=None,
            retry_status="pending",
            error_message=None
        )
        
        errors = validate_retry_state(state)
        assert len(errors) == 1
        assert "retry_attempt cannot exceed max_retries" in errors[0]
    
    def test_invalid_retry_status(self):
        """Test validation with invalid retry status"""
        state = RetryState(
            retry_id="retry_001",
            small_chunk_id="small_chunk_001",
            original_translation="Original translation",
            feedback="Feedback",
            retry_attempt=0,
            max_retries=3,
            new_translation=None,
            retry_status="invalid_status",
            error_message=None
        )
        
        errors = validate_retry_state(state)
        assert len(errors) == 1
        assert "retry_status must be one of" in errors[0]


class TestOverallState:
    """Test cases for OverallState"""
    
    def test_valid_overall_state(self):
        """Test valid OverallState creation and validation"""
        state = OverallState(
            messages=[],
            input_text="Vietnamese input text",
            big_chunks=[],
            small_chunks=[],
            translated_chunks=[],
            memory_context=[],
            review_states=[],
            retry_states=[],
            failed_translations=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="initial",
                original_text="Vietnamese input text",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            big_chunk_size=16000,
            small_chunk_size=500,
            current_big_chunk_index=0,
            current_small_chunk_index=0,
            total_big_chunks=0,
            total_small_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) == 0
    
    def test_missing_input_text(self):
        """Test validation with missing input text"""
        state = OverallState(
            messages=[],
            input_text="",  # Should not be empty
            big_chunks=[],
            small_chunks=[],
            translated_chunks=[],
            memory_context=[],
            review_states=[],
            retry_states=[],
            failed_translations=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="initial",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            big_chunk_size=16000,
            small_chunk_size=500,
            current_big_chunk_index=0,
            current_small_chunk_index=0,
            total_big_chunks=0,
            total_small_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) == 1
        assert "input_text is required" in errors[0]
    
    def test_invalid_big_chunks_field(self):
        """Test validation with invalid big_chunks field"""
        state = OverallState(
            messages=[],
            input_text="Test input",
            big_chunks="not_a_list",  #type: ignore  # Should be a list
            small_chunks=[],
            translated_chunks=[],
            memory_context=[],
            review_states=[],
            retry_states=[],
            failed_translations=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="initial",
                original_text="Test input",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            big_chunk_size=16000,
            small_chunk_size=500,
            current_big_chunk_index=0,
            current_small_chunk_index=0,
            total_big_chunks=0,
            total_small_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) == 1
        assert "big_chunks must be a list" in errors[0]
    
    def test_big_chunks_with_invalid_states(self):
        """Test validation with invalid big chunk states"""
        invalid_big_chunk = BigChunkState(
            big_chunk_id="",  # Invalid: empty ID
            big_chunk_text="",  # Invalid: empty text
            big_chunk_size=16000,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        state = OverallState(
            messages=[],
            input_text="Test input",
            big_chunks=[invalid_big_chunk],
            small_chunks=[],
            translated_chunks=[],
            memory_context=[],
            review_states=[],
            retry_states=[],
            failed_translations=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="initial",
                original_text="Test input",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            big_chunk_size=16000,
            small_chunk_size=500,
            current_big_chunk_index=0,
            current_small_chunk_index=0,
            total_big_chunks=0,
            total_small_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) == 2
        assert any("big_chunk_id is required" in error for error in errors)
        assert any("big_chunk_text is required" in error for error in errors)
    
    def test_invalid_configuration_values(self):
        """Test validation with invalid configuration values"""
        state = OverallState(
            messages=[],
            input_text="Test input",
            big_chunks=[],
            small_chunks=[],
            translated_chunks=[],
            memory_context=[],
            review_states=[],
            retry_states=[],
            failed_translations=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="initial",
                original_text="Test input",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            big_chunk_size=-1,  # Invalid: negative size
            small_chunk_size=0,  # Invalid: zero size
            current_big_chunk_index=0,
            current_small_chunk_index=0,
            total_big_chunks=0,
            total_small_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) == 2
        assert any("big_chunk_size must be a positive integer" in error for error in errors)
        assert any("small_chunk_size must be a positive integer" in error for error in errors)
    
    def test_logical_validation_errors(self):
        """Test validation with logical errors"""
        state = OverallState(
            messages=[],
            input_text="Test input",
            big_chunks=[],
            small_chunks=[],
            translated_chunks=[],
            memory_context=[],
            review_states=[],
            retry_states=[],
            failed_translations=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="initial",
                original_text="Test input",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            big_chunk_size=16000,
            small_chunk_size=500,
            current_big_chunk_index=-1,  # Invalid: negative index
            current_small_chunk_index=-1,  # Invalid: negative index
            total_big_chunks=-1,  # Invalid: negative total
            total_small_chunks=-1,  # Invalid: negative total
            processing_complete="not_a_bool",  #type: ignore  # Invalid: not boolean
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) == 5
        assert any("current_big_chunk_index must be a non-negative integer" in error for error in errors)
        assert any("current_small_chunk_index must be a non-negative integer" in error for error in errors)
        assert any("total_big_chunks must be a non-negative integer" in error for error in errors)
        assert any("total_small_chunks must be a non-negative integer" in error for error in errors)
        assert any("processing_complete must be a boolean" in error for error in errors)


class TestKnowledgeNode:
    """Test cases for KnowledgeNode"""
    
    def test_valid_knowledge_node(self):
        """Test valid KnowledgeNode creation and validation"""
        node = KnowledgeNode(
            id="node_001",
            type="GlossaryTerm",
            label="Test Term",
            content="This is a test glossary term",
            aliases=["test", "term"],
            metadata={"source": "test"}
        )
        
        errors = validate_knowledge_node(node)
        assert len(errors) == 0
    
    def test_invalid_node_type(self):
        """Test validation with invalid node type"""
        node = KnowledgeNode(
            id="node_001",
            type="InvalidType",  # Should be one of valid types
            label="Test Term",
            content="Test content",
            aliases=[],
            metadata={}
        )
        
        errors = validate_knowledge_node(node)
        assert len(errors) == 1
        assert "type must be one of" in errors[0]
    
    def test_valid_node_types(self):
        """Test all valid node types"""
        valid_types = ["GlossaryTerm", "Character", "Event"]
        
        for node_type in valid_types:
            node = KnowledgeNode(
                id="node_001",
                type=node_type,
                label="Test",
                content="Test content",
                aliases=[],
                metadata={}
            )
            
            errors = validate_knowledge_node(node)
            assert len(errors) == 0
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        node = KnowledgeNode(
            id="node_001",
            type="",  # Missing required
            label="",  # Missing required
            content="",  # Missing required
            aliases=[],
            metadata={}
        )
        
        errors = validate_knowledge_node(node)
        assert len(errors) == 3
        assert any("type is required" in error for error in errors)
        assert any("label is required" in error for error in errors)
        assert any("content is required" in error for error in errors)


class TestStateCreationAndValidation:
    """Test cases for state creation and validation utilities"""
    
    def test_create_initial_state(self):
        """Test create_initial_state function"""
        state = create_initial_state("Test input text", 16000, 500)
        
        assert state["input_text"] == "Test input text"
        assert state["big_chunk_size"] == 16000
        assert state["small_chunk_size"] == 500
        assert state["big_chunks"] == []
        assert state["small_chunks"] == []
        assert state["review_states"] == []
        assert state["retry_states"] == []
        assert state["failed_translations"] == []
        assert state["current_big_chunk_index"] == 0
        assert state["current_small_chunk_index"] == 0
        assert state["total_big_chunks"] == 0
        assert state["total_small_chunks"] == 0
        assert state["processing_complete"] == False
        assert state["retry_count"] == 0
    
    def test_create_initial_state_defaults(self):
        """Test create_initial_state with default parameters"""
        state = create_initial_state("Test input text")
        
        assert state["big_chunk_size"] == 16000  # Default value
        assert state["small_chunk_size"] == 500  # Default value
    
    def test_is_state_valid(self):
        """Test is_state_valid function"""
        # Valid state
        valid_state = create_initial_state("Test input")
        assert is_state_valid(valid_state) == True
        
        # Invalid state
        invalid_state = OverallState(
            messages=[],
            input_text="",  # Invalid: empty input
            big_chunks=[],
            small_chunks=[],
            translated_chunks=[],
            memory_context=[],
            review_states=[],
            retry_states=[],
            failed_translations=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="initial",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            big_chunk_size=16000,
            small_chunk_size=500,
            current_big_chunk_index=0,
            current_small_chunk_index=0,
            total_big_chunks=0,
            total_small_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        assert is_state_valid(invalid_state) == False
    
    def test_get_state_validation_errors(self):
        """Test get_state_validation_errors function"""
        # Valid state should have no errors
        valid_state = create_initial_state("Test input")
        errors = get_state_validation_errors(valid_state)
        assert len(errors) == 0
        
        # Invalid state should have errors
        invalid_state = OverallState(
            messages=[],
            input_text="",  # Invalid: empty input
            big_chunks=[],
            small_chunks=[],
            translated_chunks=[],
            memory_context=[],
            review_states=[],
            retry_states=[],
            failed_translations=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="initial",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            big_chunk_size=16000,
            small_chunk_size=500,
            current_big_chunk_index=0,
            current_small_chunk_index=0,
            total_big_chunks=0,
            total_small_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        errors = get_state_validation_errors(invalid_state)
        assert len(errors) > 0
        assert any("input_text is required" in error for error in errors)


class TestTranslationOutput:
    """Test cases for TranslationOutput"""
    
    def test_translation_output_creation(self):
        """Test TranslationOutput creation with defaults"""
        output = TranslationOutput()
        
        assert output.final_translation == ""
        assert output.memory_summary == {}
        assert output.processing_stats == {}
        assert output.quality_metrics == {}
    
    def test_translation_output_with_data(self):
        """Test TranslationOutput with actual data"""
        output = TranslationOutput(
            final_translation="Final translated text",
            memory_summary={"nodes_retrieved": 5},
            processing_stats={"chunks_processed": 10},
            quality_metrics={"average_rating": 8.5}
        )
        
        assert output.final_translation == "Final translated text"
        assert output.memory_summary["nodes_retrieved"] == 5
        assert output.processing_stats["chunks_processed"] == 10
        assert output.quality_metrics["average_rating"] == 8.5


if __name__ == "__main__":
    pytest.main([__file__]) 