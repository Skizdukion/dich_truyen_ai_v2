import pytest
from typing import Dict, Any, List
from src.agent.state import (
    TranslationState,
    MemoryState,
    ChunkState,
    OverallState,
    KnowledgeNode,
    MemoryOperation,
    TranslationOutput,
    validate_translation_state,
    validate_memory_state,
    validate_chunk_state,
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
            is_processed="not_a_bool",   #type: ignore # Should be a boolean
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
            translation_attempts=-1,  # Should be non-negative
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
            translation_attempts=5,  # Exceeds max_attempts
            max_attempts=3
        )
        
        errors = validate_chunk_state(state)
        assert len(errors) == 1
        assert "translation_attempts cannot exceed max_attempts" in errors[0]


class TestOverallState:
    """Test cases for OverallState"""
    
    def test_valid_overall_state(self):
        """Test valid OverallState creation and validation"""
        state = OverallState(
            messages=[],
            input_text="Test input text",
            chunks=[],
            translated_text=[],
            memory_context=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="valid_id",
                original_text="Test input text",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            chunk_size=1000,
            current_chunk_index=0,
            total_chunks=0,
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
            chunks=[],
            translated_text=[],
            memory_context=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            chunk_size=1000,
            current_chunk_index=0,
            total_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) >= 1
        assert any("input_text is required" in error for error in errors)
    
    def test_invalid_chunks_field(self):
        """Test validation with invalid chunks field"""
        state = OverallState(
            messages=[],
            input_text="Test text",
            chunks="not_a_list",  #type: ignore  # Should be a list
            translated_text=[],
            memory_context=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            chunk_size=1000,
            current_chunk_index=0,
            total_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) >= 1
        assert any("chunks must be a list" in error for error in errors)
    
    def test_chunks_with_invalid_states(self):
        """Test validation with chunks containing invalid states"""
        invalid_chunk = ChunkState(
            chunk_index=0,
            chunk_text="",  # Invalid: empty text
            chunk_size=0,
            is_processed=False,
            translation_attempts=0,
            max_attempts=3
        )
        
        state = OverallState(
            messages=[],
            input_text="Test text",
            chunks=[invalid_chunk],
            translated_text=[],
            memory_context=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            chunk_size=1000,
            current_chunk_index=0,
            total_chunks=1,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) > 0
        assert any("chunk[0]" in error for error in errors)
    
    def test_invalid_configuration_values(self):
        """Test validation with invalid configuration values"""
        state = OverallState(
            messages=[],
            input_text="Test text",
            chunks=[],
            translated_text=[],
            memory_context=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            chunk_size=-1,  # Invalid: negative
            current_chunk_index=0,
            total_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) >= 1
        assert any("chunk_size must be a positive integer" in error for error in errors)
    
    def test_logical_validation_errors(self):
        """Test logical validation errors"""
        state = OverallState(
            messages=[],
            input_text="Test text",
            chunks=[],
            translated_text=[],
            memory_context=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            chunk_size=1000,
            current_chunk_index=5,  # Invalid: exceeds total_chunks
            total_chunks=3,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = validate_overall_state(state)
        assert len(errors) >= 1
        assert any("current_chunk_index cannot exceed total_chunks" in error for error in errors)


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
            type="",  # Missing
            label="",  # Missing
            content="",  # Missing
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
        """Test initial state creation"""
        state = create_initial_state("Test input text", chunk_size=500)
        
        assert state['input_text'] == "Test input text"
        assert state['chunk_size'] == 500
        assert state['current_chunk_index'] == 0
        assert state['total_chunks'] == 0
        assert state['processing_complete'] is False
        assert state['retry_count'] == 0
        assert len(state['chunks']) == 0
        assert len(state['translated_text']) == 0
        assert len(state['memory_context']) == 0
        assert len(state['failed_chunks']) == 0
    
    def test_create_initial_state_defaults(self):
        """Test initial state creation with default values"""
        state = create_initial_state("Test input text")
        
        assert state['chunk_size'] == 1000  # Default value (not changed in create_initial_state)
        assert state['memory_state']['retrieved_nodes'] == []
        assert state['translation_state']['processing_status'] == "pending"
    
    def test_is_state_valid(self):
        """Test state validity check"""
        valid_state = create_initial_state("Test text")
        assert is_state_valid(valid_state) is True
        
        # Create invalid state
        invalid_state = OverallState(
            messages=[],
            input_text="",  # Invalid: empty
            chunks=[],
            translated_text=[],
            memory_context=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            chunk_size=1000,
            current_chunk_index=0,
            total_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        assert is_state_valid(invalid_state) is False
    
    def test_get_state_validation_errors(self):
        """Test getting detailed validation errors"""
        invalid_state = OverallState(
            messages=[],
            input_text="",  # Invalid: empty
            chunks="not_a_list",  #type: ignore  # Invalid: not a list
            translated_text=[],
            memory_context=[],
            memory_state=MemoryState(
                retrieved_nodes=[],
                created_nodes=[],
                updated_nodes=[],
                search_queries=[],
                memory_operations=[]
            ),
            translation_state=TranslationState(
                chunk_id="",
                original_text="",
                translated_text=None,
                memory_context=[],
                translation_quality=None,
                processing_status="pending",
                error_message=None
            ),
            chunk_size=1000,
            current_chunk_index=0,
            total_chunks=0,
            processing_complete=False,
            failed_chunks=[],
            retry_count=0
        )
        
        errors = get_state_validation_errors(invalid_state)
        assert len(errors) >= 2
        assert any("input_text is required" in error for error in errors)
        assert any("chunks must be a list" in error for error in errors)


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
        """Test TranslationOutput creation with data"""
        output = TranslationOutput(
            final_translation="Translated text",
            memory_summary={"nodes_created": 5},
            processing_stats={"chunks_processed": 10},
            quality_metrics={"average_quality": 0.85}
        )
        
        assert output.final_translation == "Translated text"
        assert output.memory_summary["nodes_created"] == 5
        assert output.processing_stats["chunks_processed"] == 10
        assert output.quality_metrics["average_quality"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__]) 