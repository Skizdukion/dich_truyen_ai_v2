"""
Unit tests for TranslationTools class.
Tests LLM integration and translation operations.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agent.translation_tools import TranslationTools
from agent.configuration import Configuration


class TestTranslationTools:
    """Test cases for TranslationTools class."""
    
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
    def translation_tools(self, config):
        """Create TranslationTools instance with mocked LLMs."""
        with patch('agent.translation_tools.ChatGoogleGenerativeAI') as mock_llm:
            # Mock LLM instances
            mock_translation_llm = Mock()
            mock_search_llm = Mock()
            mock_update_llm = Mock()
            mock_summary_llm = Mock()
            
            # Configure mock to return different instances
            mock_llm.side_effect = [
                mock_translation_llm,
                mock_search_llm,
                mock_update_llm,
                mock_summary_llm
            ]
            
            tools = TranslationTools(config)
            tools.translation_llm = mock_translation_llm
            tools.memory_search_llm = mock_search_llm
            tools.memory_update_llm = mock_update_llm
            tools.context_summary_llm = mock_summary_llm
            
            return tools
    
    @pytest.fixture
    def sample_memory_context(self):
        """Sample memory context for testing."""
        return [
            {
                "type": "character",
                "label": "Nhân vật chính",
                "name": "Nguyễn Văn A",
                "content": "Nhân vật chính trong truyện"
            },
            {
                "type": "term",
                "label": "Thuật ngữ kỹ thuật",
                "name": "Công nghệ AI",
                "content": "Trí tuệ nhân tạo"
            }
        ]
    
    @pytest.fixture
    def sample_recent_context(self):
        """Sample recent context for testing."""
        return [
            {
                "chunk_index": 0,
                "summary": "Đoạn đầu giới thiệu nhân vật chính",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "chunk_index": 1,
                "summary": "Nhân vật bắt đầu hành trình",
                "timestamp": "2024-01-01T00:01:00"
            }
        ]
    
    def test_initialization(self, config):
        """Test TranslationTools initialization."""
        with patch('agent.translation_tools.ChatGoogleGenerativeAI') as mock_llm:
            tools = TranslationTools(config)
            
            assert tools.config == config
            assert tools.translation_llm is not None
            assert tools.memory_search_llm is not None
            assert tools.memory_update_llm is not None
            assert tools.context_summary_llm is not None
    
    def test_initialization_failure(self, config):
        """Test TranslationTools initialization failure."""
        with patch('agent.translation_tools.ChatGoogleGenerativeAI', side_effect=Exception("LLM init failed")):
            with pytest.raises(Exception, match="LLM init failed"):
                TranslationTools(config)
    
    def test_translate_chunk_success(self, translation_tools, sample_memory_context, sample_recent_context):
        """Test successful chunk translation."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Đây là bản dịch thành công của văn bản gốc."
        translation_tools.translation_llm.invoke.return_value = mock_response
        
        original_text = "这是原文"
        result = translation_tools.translate_chunk(original_text, sample_memory_context, sample_recent_context)
        
        assert result == "Đây là bản dịch thành công của văn bản gốc."
        translation_tools.translation_llm.invoke.assert_called_once()
    
    def test_translate_chunk_fallback(self, translation_tools, sample_memory_context, sample_recent_context):
        """Test translation fallback when LLM is not available."""
        translation_tools.translation_llm = None
        
        original_text = "这是原文"
        result = translation_tools.translate_chunk(original_text, sample_memory_context, sample_recent_context)
        
        assert result == "[DỊCH] 这是原文"
    
    def test_translate_chunk_exception(self, translation_tools, sample_memory_context, sample_recent_context):
        """Test translation with exception handling."""
        translation_tools.translation_llm.invoke.side_effect = Exception("Translation failed")
        
        original_text = "这是原文"
        with pytest.raises(Exception, match="Translation failed"):
            translation_tools.translate_chunk(original_text, sample_memory_context, sample_recent_context)
    
    def test_generate_search_queries_success(self, translation_tools):
        """Test successful search query generation."""
        # Mock LLM response with valid JSON
        mock_response = Mock()
        mock_response.content = '["nhân vật chính", "công nghệ AI", "hành trình"]'
        translation_tools.memory_search_llm.invoke.return_value = mock_response
        
        chunk_text = "Nhân vật chính bắt đầu hành trình với công nghệ AI."
        result = translation_tools.generate_search_queries(chunk_text)
        
        assert result == ["nhân vật chính", "công nghệ AI", "hành trình"]
        translation_tools.memory_search_llm.invoke.assert_called_once()
    
    def test_generate_search_queries_invalid_json(self, translation_tools):
        """Test search query generation with invalid JSON response."""
        # Mock LLM response with invalid JSON
        mock_response = Mock()
        mock_response.content = "invalid json response"
        translation_tools.memory_search_llm.invoke.return_value = mock_response
        
        chunk_text = "Nhân vật chính bắt đầu hành trình."
        result = translation_tools.generate_search_queries(chunk_text)
        
        # Should fall back to heuristic queries
        assert isinstance(result, list)
        assert len(result) <= 3
    
    def test_generate_search_queries_wrong_format(self, translation_tools):
        """Test search query generation with wrong JSON format."""
        # Mock LLM response with wrong format (not a list)
        mock_response = Mock()
        mock_response.content = '{"queries": ["test"]}'
        translation_tools.memory_search_llm.invoke.return_value = mock_response
        
        chunk_text = "Nhân vật chính bắt đầu hành trình."
        result = translation_tools.generate_search_queries(chunk_text)
        
        # Should fall back to heuristic queries
        assert isinstance(result, list)
        assert len(result) <= 3
    
    def test_generate_search_queries_fallback(self, translation_tools):
        """Test search query generation fallback when LLM is not available."""
        translation_tools.memory_search_llm = None
        
        chunk_text = "Nhân vật chính bắt đầu hành trình."
        result = translation_tools.generate_search_queries(chunk_text)
        
        assert isinstance(result, list)
        assert len(result) <= 3
    
    def test_generate_memory_operations_success(self, translation_tools, sample_memory_context):
        """Test successful memory operations generation."""
        # Mock LLM response with valid JSON
        mock_response = Mock()
        mock_response.content = json.dumps({
            "create_nodes": [
                {
                    "type": "character",
                    "label": "Nhân vật mới",
                    "name": "Trần Thị B",
                    "content": "Nhân vật phụ trong truyện",
                    "alias": ["B", "Trần B"]
                }
            ],
            "update_nodes": []
        })
        translation_tools.memory_update_llm.invoke.return_value = mock_response
        
        original_text = "原文"
        translated_text = "Văn bản đã dịch"
        result = translation_tools.generate_memory_operations(original_text, translated_text, sample_memory_context)
        
        assert "create_nodes" in result
        assert "update_nodes" in result
        assert len(result["create_nodes"]) == 1
        assert result["create_nodes"][0]["type"] == "character"
        translation_tools.memory_update_llm.invoke.assert_called_once()
    
    def test_generate_memory_operations_invalid_json(self, translation_tools, sample_memory_context):
        """Test memory operations generation with invalid JSON."""
        # Mock LLM response with invalid JSON
        mock_response = Mock()
        mock_response.content = "invalid json"
        translation_tools.memory_update_llm.invoke.return_value = mock_response
        
        original_text = "原文"
        translated_text = "Văn bản đã dịch"
        result = translation_tools.generate_memory_operations(original_text, translated_text, sample_memory_context)
        
        assert result == {"create_nodes": [], "update_nodes": []}
    
    def test_generate_memory_operations_fallback(self, translation_tools, sample_memory_context):
        """Test memory operations generation fallback when LLM is not available."""
        translation_tools.memory_update_llm = None
        
        original_text = "原文"
        translated_text = "Văn bản đã dịch"
        result = translation_tools.generate_memory_operations(original_text, translated_text, sample_memory_context)
        
        assert result == {"create_nodes": [], "update_nodes": []}
    
    def test_generate_context_summary_success(self, translation_tools):
        """Test successful context summary generation."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Tóm tắt ngữ cảnh: Nhân vật chính tiếp tục hành trình."
        translation_tools.context_summary_llm.invoke.return_value = mock_response
        
        recent_context = "Ngữ cảnh gần đây"
        current_translation = "Bản dịch hiện tại"
        result = translation_tools.generate_context_summary(recent_context, current_translation)
        
        assert result == "Tóm tắt ngữ cảnh: Nhân vật chính tiếp tục hành trình."
        translation_tools.context_summary_llm.invoke.assert_called_once()
    
    def test_generate_context_summary_fallback(self, translation_tools):
        """Test context summary generation fallback when LLM is not available."""
        translation_tools.context_summary_llm = None
        
        recent_context = "Ngữ cảnh gần đây"
        current_translation = "Bản dịch hiện tại"
        result = translation_tools.generate_context_summary(recent_context, current_translation)
        
        assert "Ngữ cảnh gần đây:" in result
        assert str(len(current_translation)) in result
    
    def test_format_memory_context(self, translation_tools, sample_memory_context):
        """Test memory context formatting."""
        result = translation_tools._format_memory_context(sample_memory_context)
        
        assert "character: Nhân vật chính (Nguyễn Văn A)" in result
        assert "term: Thuật ngữ kỹ thuật (Công nghệ AI)" in result
        assert "Trí tuệ nhân tạo" in result
    
    def test_format_memory_context_empty(self, translation_tools):
        """Test memory context formatting with empty context."""
        result = translation_tools._format_memory_context([])
        
        assert result == "Không tìm thấy ngữ cảnh bộ nhớ liên quan."
    
    def test_format_recent_context(self, translation_tools, sample_recent_context):
        """Test recent context formatting."""
        result = translation_tools._format_recent_context(sample_recent_context)
        
        assert "Đoạn đầu giới thiệu nhân vật chính" in result
        assert "Nhân vật bắt đầu hành trình" in result
    
    def test_format_recent_context_empty(self, translation_tools):
        """Test recent context formatting with empty context."""
        result = translation_tools._format_recent_context([])
        
        assert result == "Không có ngữ cảnh dịch thuật gần đây."
    
    def test_fallback_search_queries(self, translation_tools):
        """Test fallback search query generation."""
        chunk_text = "Nguyễn Văn A và Trần Thị B gặp nhau tại Hà Nội."
        result = translation_tools._fallback_search_queries(chunk_text)
        
        # Should find proper nouns starting with capital letters
        assert isinstance(result, list)
        assert len(result) <= 3
        # Note: This test depends on the heuristic logic in the implementation 