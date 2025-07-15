"""
Unit tests for enhanced translation graph workflow.
Tests the LangGraph nodes and workflow execution with big chunks, small chunks, review, and retry.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import os

from src.agent.graph import (
    create_translation_graph,
    big_chunk_input_node,
    small_chunk_input_node,
    search_memory_node,
    translate_small_chunk_node,
    review_chunk_node,
    retry_translation_node,
    memory_update_node,
    recent_context_update_node,
    should_retry_or_continue
)
from src.agent.state import OverallState, create_initial_state, BigChunkState, SmallChunkState
from src.agent.configuration import Configuration
from src.agent.translation_tools import TranslationTools
from src.agent.review_agent import ReviewAgent
from src.agent.utils import VietnameseTextChunker


class TestEnhancedTranslationGraph:
    """Test cases for enhanced translation graph workflow."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Configuration(
            translation_model="gemini-2.0-flash",
            memory_search_model="gemini-2.0-flash",
            memory_update_model="gemini-2.0-flash",
            context_summary_model="gemini-2.0-flash"
        )
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        input_text = "这是第一段。这是第二段。这是第三段。这是第四段。这是第五段。"
        state = create_initial_state(input_text, big_chunk_size=100, small_chunk_size=50)
        return state
    
    @pytest.fixture
    def mock_weaviate_client(self):
        """Create a mock Weaviate client."""
        mock_client = Mock()
        mock_client.search_nodes_by_text.return_value = [
            {
                "id": "node1",
                "type": "character",
                "label": "Nhân vật chính",
                "name": "Nguyễn Văn A",
                "content": "Nhân vật chính trong truyện"
            }
        ]
        mock_client.insert_knowledge_node.return_value = "new_node_id"
        mock_client.update_node.return_value = True
        mock_client.count_objects.return_value = 10
        return mock_client
    
    @pytest.fixture
    def mock_translation_tools(self):
        """Create a mock TranslationTools instance."""
        mock_tools = Mock(spec=TranslationTools)
        mock_tools.generate_search_queries.return_value = ["nhân vật", "hành trình"]
        mock_tools.translate_small_chunk.return_value = ("Đây là bản dịch thành công.", Mock(), Mock())
        mock_tools.retranslate_with_feedback.return_value = ("Đây là bản dịch cải thiện.", Mock(), Mock())
        mock_tools.generate_memory_operations.return_value = {
            "create_nodes": [],
            "update_nodes": []
        }
        mock_tools.generate_context_summary.return_value = "Tóm tắt ngữ cảnh mới."
        mock_tools._format_recent_context.return_value = "Ngữ cảnh gần đây"
        return mock_tools
    
    @pytest.fixture
    def mock_review_agent(self):
        """Create a mock ReviewAgent instance."""
        mock_agent = Mock(spec=ReviewAgent)
        def review_chunk_side_effect(review_state):
            # Return the review state with the actual chunk_id from the input
            return {
                'chunk_id': review_state['chunk_id'],
                'original_text': review_state['original_text'],
                'translated_text': review_state['translated_text'],
                'rating': 'good',
                'feedback': 'Good translation',
                'confidence': 0.8,
                'requires_revision': False,
                'review_timestamp': None,
                'reviewer_id': None
            }
        mock_agent.review_chunk.side_effect = review_chunk_side_effect
        return mock_agent
    
    def test_create_translation_graph(self, config):
        """Test enhanced translation graph creation."""
        with patch('src.agent.graph.WeaviateWrapperClient') as mock_weaviate, \
             patch('src.agent.graph.TranslationTools') as mock_tools, \
             patch('src.agent.graph.ReviewAgent') as mock_review_agent, \
             patch('src.agent.graph.VietnameseTextChunker') as mock_chunker:
            
            graph = create_translation_graph(config)
            
            assert graph is not None
            mock_tools.assert_called_once_with(config)
            mock_review_agent.assert_called_once()
            mock_chunker.assert_called_once()
    
    def test_big_chunk_input_node(self, sample_state):
        """Test big chunk input node processing."""
        chunker = VietnameseTextChunker()
        
        result = big_chunk_input_node(sample_state, chunker, Configuration())
        
        # Verify big chunks were created
        assert len(result['big_chunks']) > 0
        assert result['total_big_chunks'] > 0
        assert result['current_big_chunk_index'] == 0
        
        # Verify each big chunk has the correct structure
        for big_chunk in result['big_chunks']:
            assert 'big_chunk_id' in big_chunk
            assert 'big_chunk_text' in big_chunk
            assert 'big_chunk_size' in big_chunk
            assert 'memory_context' in big_chunk
            assert 'small_chunks' in big_chunk
            assert 'is_processed' in big_chunk
            assert 'processing_status' in big_chunk
    
    def test_small_chunk_input_node(self, sample_state):
        """Test small chunk input node processing."""
        chunker = VietnameseTextChunker()
        
        # First create big chunks
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        
        # Then create small chunks
        result = small_chunk_input_node(state, chunker, Configuration())
        
        # Verify small chunks were created
        assert len(result['small_chunks']) > 0
        assert result['total_small_chunks'] > 0
        assert result['current_small_chunk_index'] == 0
        
        # Verify each small chunk has the correct structure and flexible size
        for small_chunk in result['small_chunks']:
            assert 'small_chunk_id' in small_chunk
            assert 'big_chunk_id' in small_chunk
            assert 'small_chunk_text' in small_chunk
            assert 'small_chunk_size' in small_chunk
            assert 'position_in_big_chunk' in small_chunk
            assert 'translated_text' in small_chunk
            assert 'recent_context' in small_chunk
            assert 'translation_attempts' in small_chunk
            assert 'max_attempts' in small_chunk
            assert 'is_processed' in small_chunk
            assert 'processing_status' in small_chunk
            # Flexible chunk size assertion
            assert 400 <= small_chunk['small_chunk_size'] <= 700
    
    def test_translate_small_chunk_node(self, sample_state, mock_translation_tools):
        """Test small chunk translation node."""
        # Set up state with small chunks
        chunker = VietnameseTextChunker()
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        state = small_chunk_input_node(state, chunker, Configuration())
        
        result = translate_small_chunk_node(state, mock_translation_tools)
        
        # Verify translation was performed
        mock_translation_tools.translate_small_chunk.assert_called_once()
        assert result['small_chunks'][0]['translated_text'] == "Đây là bản dịch thành công."
        assert result['small_chunks'][0]['is_processed'] is True
        assert result['small_chunks'][0]['processing_status'] == 'completed'
    
    def test_review_chunk_node(self, sample_state, mock_review_agent):
        """Test review chunk node."""
        # Set up state with translated small chunk
        chunker = VietnameseTextChunker()
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        state = small_chunk_input_node(state, chunker, Configuration())
        state['small_chunks'][0]['translated_text'] = "Translated text"
        
        result = review_chunk_node(state, mock_review_agent)
        
        # Verify review was performed
        mock_review_agent.review_chunk.assert_called_once()
        assert len(result['review_states']) == 1
        assert result['review_states'][0]['chunk_id'] == state['small_chunks'][0]['small_chunk_id']
    
    def test_retry_translation_node(self, sample_state, mock_translation_tools):
        """Test retry translation node."""
        # Set up state with review that requires revision
        chunker = VietnameseTextChunker()
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        state = small_chunk_input_node(state, chunker, Configuration())
        state['small_chunks'][0]['translated_text'] = "Original translation"
        state['review_states'] = [{
            'chunk_id': state['small_chunks'][0]['small_chunk_id'],
            'original_text': 'Test text',
            'translated_text': 'Original translation',
            'feedback': 'Improve accuracy',
            'requires_revision': True,
            'confidence': 0.5,
            'rating': None,
            'context': None,
            'review_timestamp': None,
            'reviewer_id': None
        }]
        
        result = retry_translation_node(state, mock_translation_tools)
        
        # Verify retranslation was performed
        mock_translation_tools.retranslate_with_feedback.assert_called_once()
        assert result['small_chunks'][0]['translated_text'] == "Đây là bản dịch cải thiện."
        assert result['small_chunks'][0]['is_processed'] is True
        assert result['small_chunks'][0]['processing_status'] == 'completed'
    
    def test_should_retry_or_continue_retry(self, sample_state):
        """Test retry decision when review requires revision."""
        # Set up state with review requiring revision
        chunker = VietnameseTextChunker()
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        state = small_chunk_input_node(state, chunker, Configuration())
        state['review_states'] = [{
            'chunk_id': state['small_chunks'][0]['small_chunk_id'],
            'original_text': 'Test text',
            'translated_text': 'Original translation',
            'context': None,
            'rating': None,
            'feedback': 'Improve accuracy',
            'confidence': 0.5,
            'requires_revision': True,
            'review_timestamp': None,
            'reviewer_id': None
        }]
        
        result = should_retry_or_continue(state)
        
        assert result == "retry"
    
    def test_should_retry_or_continue_continue(self, sample_state):
        """Test continue decision when review is satisfactory."""
        # Set up state with satisfactory review
        chunker = VietnameseTextChunker()
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        state = small_chunk_input_node(state, chunker, Configuration())
        state['review_states'] = [{
            'chunk_id': state['small_chunks'][0]['small_chunk_id'],
            'original_text': 'Test text',
            'translated_text': 'Good translation',
            'context': None,
            'rating': None,
            'feedback': 'Good translation',
            'confidence': 0.8,
            'requires_revision': False,
            'review_timestamp': None,
            'reviewer_id': None
        }]
        
        result = should_retry_or_continue(state)
        
        assert result == "continue"
    
    def test_search_memory_node_success(self, sample_state, mock_weaviate_client, mock_translation_tools):
        """Test successful memory search node."""
        # Set up state with small chunks
        chunker = VietnameseTextChunker()
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        state = small_chunk_input_node(state, chunker, Configuration())
        
        with patch('src.agent.graph.WeaviateWrapperClient', return_value=mock_weaviate_client):
            result = search_memory_node(state, mock_translation_tools)
        
        # Verify memory search was performed
        mock_translation_tools.generate_search_queries.assert_called_once()
        mock_weaviate_client.search_nodes_by_text.assert_called()
        assert len(result['translation_state']['memory_context']) > 0
    
    def test_memory_update_node_success(self, sample_state, mock_weaviate_client, mock_translation_tools):
        """Test successful memory update node."""
        # Set up state with translation result
        chunker = VietnameseTextChunker()
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        state = small_chunk_input_node(state, chunker, Configuration())
        state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': 'Đây là bản dịch thành công.',
            'memory_context': [{'type': 'character', 'name': 'Test'}],
            'translation_quality': None,
            'processing_status': 'completed',
            'error_message': None
        }
        
        with patch('src.agent.graph.WeaviateWrapperClient', return_value=mock_weaviate_client):
            result = memory_update_node(state, mock_translation_tools)
        
        # Verify memory operations were generated
        mock_translation_tools.generate_memory_operations.assert_called_once()
        assert len(result['memory_state']['memory_operations']) == 1
    
    def test_recent_context_update_node_success(self, sample_state, mock_translation_tools):
        """Test successful recent context update node."""
        # Set up state with translation result
        chunker = VietnameseTextChunker()
        state = big_chunk_input_node(sample_state, chunker, Configuration())
        state = small_chunk_input_node(state, chunker, Configuration())
        state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': 'Đây là bản dịch thành công.',
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'completed',
            'error_message': None
        }
        
        result = recent_context_update_node(state, mock_translation_tools)
        
        # Verify context was updated
        mock_translation_tools.generate_context_summary.assert_called_once()
        assert len(result['memory_context']) == 1


class TestEnhancedTranslationGraphIntegration:
    """Integration tests for enhanced translation graph with real API calls."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration with real API keys."""
        return Configuration(
            translation_model="gemini-2.0-flash",
            memory_search_model="gemini-2.0-flash",
            memory_update_model="gemini-2.0-flash",
            context_summary_model="gemini-2.0-flash"
        )
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        这是第一段测试文本。它包含了一些基本的句子。
        这是第二段测试文本。它继续了第一段的内容。
        这是第三段测试文本。它提供了更多的上下文信息。
        """
    
    @pytest.mark.integration
    def test_full_enhanced_workflow(self, config, sample_text):
        """Test the complete enhanced workflow with real API calls."""
        # Skip if no API key
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("No Google API key available")
        
        # Create initial state
        state = create_initial_state(sample_text, big_chunk_size=200, small_chunk_size=100)
        
        # Create the enhanced graph
        graph = create_translation_graph(config)
        
        # Run the workflow
        result = graph.invoke(state)
        
        # Verify the result
        assert result is not None
        assert 'big_chunks' in result
        assert 'small_chunks' in result
        assert 'review_states' in result
        assert 'translated_chunks' in result
        assert result['processing_complete'] is True
        
        # Verify big chunks were created
        assert len(result['big_chunks']) > 0
        assert result['total_big_chunks'] > 0
        
        # Verify small chunks were created
        assert len(result['small_chunks']) > 0
        assert result['total_small_chunks'] > 0
        
        # Verify translations were performed
        assert len(result['translated_chunks']) > 0
        
        # Verify reviews were performed
        assert len(result['review_states']) > 0
        
        print(f"Enhanced workflow completed:")
        print(f"  Big chunks: {result['total_big_chunks']}")
        print(f"  Small chunks: {result['total_small_chunks']}")
        print(f"  Translated chunks: {len(result['translated_chunks'])}")
        print(f"  Reviews: {len(result['review_states'])}")
        print(f"  Memory operations: {len(result['memory_state']['memory_operations'])}")
    
    @pytest.mark.integration
    def test_enhanced_workflow_with_retry(self, config):
        """Test the enhanced workflow with retry functionality."""
        # Skip if no API key
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("No Google API key available")
        
        # Use a more complex text that might trigger review feedback
        complex_text = """
        这是一个复杂的测试文本。它包含了一些技术术语和复杂的句子结构。
        这段文本可能会被翻译得不够准确，从而触发重新翻译的流程。
        我们希望通过这个测试来验证整个增强工作流程的功能。
        """
        
        # Create initial state
        state = create_initial_state(complex_text, big_chunk_size=300, small_chunk_size=150)
        
        # Create the enhanced graph
        graph = create_translation_graph(config)
        
        # Run the workflow
        result = graph.invoke(state)
        
        # Verify the result
        assert result is not None
        assert result['processing_complete'] is True
        
        # Check if any retries occurred (this is probabilistic)
        retry_states = [r for r in result.get('retry_states', []) if r.get('retry_status') == 'completed']
        
        print(f"Enhanced workflow with retry completed:")
        print(f"  Total reviews: {len(result['review_states'])}")
        print(f"  Retry attempts: {len(retry_states)}")
        print(f"  Failed translations: {len(result.get('failed_translations', []))}")
        
        # Verify that the workflow completed successfully regardless of retries
        assert len(result['translated_chunks']) > 0
    
    @pytest.mark.integration
    def test_enhanced_workflow_performance(self, config, sample_text):
        """Test the performance of the enhanced workflow."""
        # Skip if no API key
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("No Google API key available")
        
        import time
        
        # Create initial state
        state = create_initial_state(sample_text, big_chunk_size=200, small_chunk_size=100)
        
        # Create the enhanced graph
        graph = create_translation_graph(config)
        
        # Measure execution time
        start_time = time.time()
        result = graph.invoke(state)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Verify the result
        assert result is not None
        assert result['processing_complete'] is True
        
        print(f"Enhanced workflow performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Big chunks processed: {result['total_big_chunks']}")
        print(f"  Small chunks processed: {result['total_small_chunks']}")
        print(f"  Average time per chunk: {execution_time / max(result['total_small_chunks'], 1):.2f} seconds")
        
        # Performance assertions (adjust based on expected performance)
        assert execution_time > 0
        assert result['total_small_chunks'] > 0 