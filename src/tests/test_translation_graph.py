"""
Unit tests for translation graph workflow.
Tests the LangGraph nodes and workflow execution.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from agent.graph import (
    create_translation_graph,
    chunk_input_node,
    search_memory_node,
    translate_chunk_node,
    memory_update_node,
    recent_context_update_node,
    should_continue_to_next_chunk
)
from agent.state import OverallState, create_initial_state
from agent.configuration import Configuration
from agent.translation_tools import TranslationTools


class TestTranslationGraph:
    """Test cases for translation graph workflow."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Configuration(
            translation_model="gemini-2.0-flash",
            memory_search_model="gemini-2.0-flash",
            memory_update_model="gemini-2.0-flash",
            context_summary_model="gemini-2.0-flash",
            default_chunk_size=100
        )
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        input_text = "这是第一段。这是第二段。这是第三段。"
        state = create_initial_state(input_text, chunk_size=50)
        # Ensure chunks are properly initialized
        if not state['chunks']:
            state['chunks'] = [
                {
                    'chunk_index': 0,
                    'chunk_text': '这是第一段。',
                    'chunk_size': 50,
                    'is_processed': False,
                    'translation_attempts': 0,
                    'max_attempts': 3
                },
                {
                    'chunk_index': 1,
                    'chunk_text': '这是第二段。',
                    'chunk_size': 50,
                    'is_processed': False,
                    'translation_attempts': 0,
                    'max_attempts': 3
                }
            ]
            state['total_chunks'] = len(state['chunks'])
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
        return mock_client
    
    @pytest.fixture
    def mock_translation_tools(self):
        """Create a mock TranslationTools instance."""
        mock_tools = Mock(spec=TranslationTools)
        mock_tools.generate_search_queries.return_value = ["nhân vật", "hành trình"]
        mock_tools.translate_chunk.return_value = "Đây là bản dịch thành công."
        mock_tools.generate_memory_operations.return_value = {
            "create_nodes": [],
            "update_nodes": []
        }
        mock_tools.generate_context_summary.return_value = "Tóm tắt ngữ cảnh mới."
        mock_tools._format_recent_context.return_value = "Ngữ cảnh gần đây"
        return mock_tools
    
    def test_create_translation_graph(self, config):
        """Test translation graph creation."""
        with patch('agent.graph.WeaviateWrapperClient') as mock_weaviate, \
             patch('agent.graph.TranslationTools') as mock_tools:
            
            graph = create_translation_graph(config)
            
            assert graph is not None
            mock_weaviate.assert_called_once()
            mock_tools.assert_called_once_with(config)
    
    def test_chunk_input_node(self, sample_state):
        """Test chunk input node processing."""
        # Set up initial state
        sample_state['current_chunk_index'] = 0
        sample_state['chunks'][0]['translation_attempts'] = 0
        
        result = chunk_input_node(sample_state)
        
        # Verify state updates
        assert result['current_chunk_index'] == 0
        assert result['translation_state']['chunk_id'] == "chunk_0"
        assert result['translation_state']['original_text'] == sample_state['chunks'][0]['chunk_text']
        assert result['translation_state']['processing_status'] == 'processing'
        assert result['chunks'][0]['is_processed'] is True
        assert result['chunks'][0]['translation_attempts'] == 1
    
    def test_search_memory_node_success(self, sample_state, mock_weaviate_client, mock_translation_tools):
        """Test successful memory search node."""
        # Set up state
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': None,
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'processing',
            'error_message': None
        }
        
        with patch('agent.graph.WeaviateWrapperClient', return_value=mock_weaviate_client):
            result = search_memory_node(sample_state, mock_translation_tools)
        
        # Verify memory search was performed
        mock_translation_tools.generate_search_queries.assert_called_once()
        mock_weaviate_client.search_nodes_by_text.assert_called()
        assert len(result['translation_state']['memory_context']) > 0
        assert result['translation_state']['processing_status'] == 'processing'
    
    def test_search_memory_node_failure(self, sample_state, mock_translation_tools):
        """Test memory search node with failure."""
        # Set up state
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': None,
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'processing',
            'error_message': None
        }
        
        # Mock Weaviate client to raise exception
        with patch('agent.graph.WeaviateWrapperClient', side_effect=Exception("Connection failed")):
            result = search_memory_node(sample_state, mock_translation_tools)
        
        # Verify error handling
        assert result['translation_state']['processing_status'] == 'failed'
        assert result['translation_state']['error_message'] is not None
        assert "Connection failed" in result['translation_state']['error_message']
    
    def test_translate_chunk_node_success(self, sample_state, mock_translation_tools):
        """Test successful translation chunk node."""
        # Set up state with memory context
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': None,
            'memory_context': [{'type': 'character', 'name': 'Test'}],
            'translation_quality': None,
            'processing_status': 'processing',
            'error_message': None
        }
        sample_state['memory_context'] = [{'summary': 'Previous context'}]
        
        result = translate_chunk_node(sample_state, mock_translation_tools)
        
        # Verify translation was performed
        mock_translation_tools.translate_chunk.assert_called_once()
        assert result['translation_state']['translated_text'] == "Đây là bản dịch thành công."
        assert result['translation_state']['processing_status'] == 'completed'
        assert len(result['translated_text']) == 1
    
    def test_translate_chunk_node_failure(self, sample_state, mock_translation_tools):
        """Test translation chunk node with failure."""
        # Set up state
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': None,
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'processing',
            'error_message': None
        }
        
        # Mock translation to fail
        mock_translation_tools.translate_chunk.side_effect = Exception("Translation failed")
        
        result = translate_chunk_node(sample_state, mock_translation_tools)
        
        # Verify error handling
        assert result['translation_state']['processing_status'] == 'failed'
        assert result['translation_state']['error_message'] is not None
        assert "Translation failed" in result['translation_state']['error_message']
    
    def test_memory_update_node_success(self, sample_state, mock_weaviate_client, mock_translation_tools):
        """Test successful memory update node."""
        # Set up state with translation result
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': 'Đây là bản dịch thành công.',
            'memory_context': [{'type': 'character', 'name': 'Test'}],
            'translation_quality': None,
            'processing_status': 'completed',
            'error_message': None
        }
        
        with patch('agent.graph.WeaviateWrapperClient', return_value=mock_weaviate_client):
            result = memory_update_node(sample_state, mock_translation_tools)
        
        # Verify memory operations were generated
        mock_translation_tools.generate_memory_operations.assert_called_once()
        assert len(result['memory_state']['memory_operations']) == 1
    
    def test_memory_update_node_no_translation(self, sample_state, mock_translation_tools):
        """Test memory update node with no translation."""
        # Set up state without translation
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': None,
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'failed',
            'error_message': 'Translation failed'
        }
        
        result = memory_update_node(sample_state, mock_translation_tools)
        
        # Verify no memory operations were generated
        mock_translation_tools.generate_memory_operations.assert_not_called()
    
    def test_recent_context_update_node_success(self, sample_state, mock_translation_tools):
        """Test successful recent context update node."""
        # Set up state with translation
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': 'Đây là bản dịch thành công.',
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'completed',
            'error_message': None
        }
        sample_state['memory_context'] = [{'summary': 'Previous context'}]
        
        result = recent_context_update_node(sample_state, mock_translation_tools)
        
        # Verify context summary was generated
        mock_translation_tools.generate_context_summary.assert_called_once()
        assert len(result['memory_context']) == 2  # Previous + new
    
    def test_recent_context_update_node_no_translation(self, sample_state, mock_translation_tools):
        """Test recent context update node with no translation."""
        # Set up state without translation
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': None,
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'failed',
            'error_message': 'Translation failed'
        }
        
        result = recent_context_update_node(sample_state, mock_translation_tools)
        
        # Verify fallback summary was used
        assert len(result['memory_context']) == 1
        assert "Không có bản dịch để tóm tắt" in result['memory_context'][0]['summary']
    
    def test_should_continue_to_next_chunk_continue(self, sample_state):
        """Test should_continue_to_next_chunk when more chunks exist."""
        # Set up state with more chunks to process
        sample_state['current_chunk_index'] = 0
        sample_state['total_chunks'] = 3
        
        result = should_continue_to_next_chunk(sample_state)
        
        assert result == "continue"
        assert sample_state['current_chunk_index'] == 1
    
    def test_should_continue_to_next_chunk_end(self, sample_state):
        """Test should_continue_to_next_chunk when all chunks processed."""
        # Set up state with last chunk
        sample_state['current_chunk_index'] = 2
        sample_state['total_chunks'] = 3
        
        result = should_continue_to_next_chunk(sample_state)
        
        assert result == "end"
        assert sample_state['processing_complete'] is True
    
    def test_context_overflow_handling(self, sample_state, mock_translation_tools):
        """Test that context doesn't grow beyond limit."""
        # Set up state with many context items
        sample_state['memory_context'] = [
            {'summary': f'Context {i}', 'timestamp': f'2024-01-01T00:0{i}:00'}
            for i in range(10)  # More than the limit of 5
        ]
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': 'Đây là bản dịch thành công.',
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'completed',
            'error_message': None
        }
        
        result = recent_context_update_node(sample_state, mock_translation_tools)
        
        # Verify context is limited to 5 items
        assert len(result['memory_context']) <= 5
    
    def test_memory_operations_logging(self, sample_state, mock_weaviate_client, mock_translation_tools):
        """Test that memory operations are properly logged."""
        # Set up state
        sample_state['current_chunk_index'] = 0
        sample_state['translation_state'] = {
            'chunk_id': 'chunk_0',
            'original_text': '这是测试文本',
            'translated_text': 'Đây là bản dịch thành công.',
            'memory_context': [],
            'translation_quality': None,
            'processing_status': 'completed',
            'error_message': None
        }
        
        with patch('agent.graph.WeaviateWrapperClient', return_value=mock_weaviate_client):
            result = memory_update_node(sample_state, mock_translation_tools)
        
        # Verify memory operation was logged
        memory_ops = result['memory_state']['memory_operations']
        assert len(memory_ops) == 1
        assert memory_ops[0]['operation_type'] == 'memory_update'
        assert 'chunk 0' in memory_ops[0]['query_or_content'] 