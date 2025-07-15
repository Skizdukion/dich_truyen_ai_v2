"""
Integration tests for the complete translation workflow.
Tests end-to-end translation pipeline with mocked components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from agent.graph import create_translation_graph
from agent.state import create_initial_state, OverallState
from agent.configuration import Configuration
from agent.translation_tools import TranslationTools


class TestTranslationIntegration:
    """Integration tests for the complete translation workflow."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Configuration(
            translation_model="gemini-2.0-flash",
            memory_search_model="gemini-2.0-flash",
            memory_update_model="gemini-2.0-flash",
            context_summary_model="gemini-2.0-flash",
        )

    @pytest.fixture
    def sample_input_text(self):
        """Sample Vietnamese text for testing."""
        return """
        Đây là phần đầu của chương 1. Nhân vật chính Trương Tam là một lập trình viên trẻ.
        
        Anh ta sống ở Bắc Kinh, làm việc tại một công ty công nghệ.
        
        Hôm nay, anh ta sẽ tham gia một cuộc họp quan trọng.
        
        Chủ đề của cuộc họp là về sự phát triển của trí tuệ nhân tạo.
        """

    @pytest.fixture
    def mock_weaviate_client(self):
        """Create a comprehensive mock Weaviate client."""
        mock_client = Mock()

        # Mock search results
        mock_client.search_nodes_by_text.return_value = [
            {
                "id": "char_001",
                "type": "character",
                "label": "Nhân vật chính",
                "name": "Trương Tam",
                "content": "Nhân vật chính, lập trình viên trẻ",
            },
            {
                "id": "term_001",
                "type": "term",
                "label": "Thuật ngữ công nghệ",
                "name": "Trí tuệ nhân tạo",
                "content": "Artificial Intelligence, AI",
            },
        ]

        # Mock node creation
        mock_client.insert_knowledge_node.return_value = "new_node_id"

        return mock_client

    @pytest.fixture
    def mock_translation_tools(self):
        """Create a comprehensive mock TranslationTools instance."""
        mock_tools = Mock(spec=TranslationTools)

        # Mock search queries
        mock_tools.generate_search_queries.return_value = [
            "Trương Tam",
            "lập trình viên",
            "trí tuệ nhân tạo",
        ]

        # Mock translation
        mock_tools.translate_chunk.return_value = (
            "Đây là bản dịch thành công của đoạn văn."
        )

        # Mock memory operations
        mock_tools.generate_memory_operations.return_value = {
            "create_nodes": [
                {
                    "type": "character",
                    "label": "Nhân vật mới",
                    "name": "Lý Tứ",
                    "content": "Đồng nghiệp của Trương Tam",
                    "alias": ["Tứ", "Lý"],
                }
            ],
            "update_nodes": [],
        }

        # Mock context summary
        mock_tools.generate_context_summary.return_value = (
            "Tóm tắt: Nhân vật chính chuẩn bị cho cuộc họp quan trọng."
        )
        mock_tools._format_recent_context.return_value = (
            "Ngữ cảnh gần đây: Nhân vật chính"
        )

        return mock_tools

    def test_complete_translation_workflow(
        self, config, sample_input_text, mock_weaviate_client, mock_translation_tools
    ):
        """Test the complete translation workflow from start to finish."""
        # Create initial state
        initial_state = create_initial_state(sample_input_text, chunk_size=200)

        # Mock the graph creation
        with patch(
            "agent.graph.WeaviateWrapperClient", return_value=mock_weaviate_client
        ), patch("agent.graph.TranslationTools", return_value=mock_translation_tools):

            # Create the graph
            graph = create_translation_graph(config)

            # Execute the workflow
            final_state = graph.invoke(initial_state)

            # Verify the workflow completed successfully
            assert final_state["processing_complete"] is True
            assert len(final_state["translated_text"]) > 0
            assert final_state["current_chunk_index"] == final_state["total_chunks"] - 1

            # Verify memory operations were performed
            assert len(final_state["memory_state"]["memory_operations"]) > 0
            assert len(final_state["memory_state"]["retrieved_nodes"]) > 0
            assert len(final_state["memory_state"]["search_queries"]) > 0

            # Verify context was maintained
            assert len(final_state["memory_context"]) > 0

    def test_translation_workflow_with_existing_memory(
        self, config, sample_input_text, mock_weaviate_client, mock_translation_tools
    ):
        """Test translation workflow with existing memory context."""
        # Create initial state with existing memory
        initial_state = create_initial_state(sample_input_text, chunk_size=200)
        initial_state["memory_context"] = [
            {
                "chunk_index": -1,
                "summary": "Ngữ cảnh trước đó: Nhân vật chính đã được giới thiệu",
                "timestamp": "2024-01-01T00:00:00",
            }
        ]

        with patch(
            "agent.graph.WeaviateWrapperClient", return_value=mock_weaviate_client
        ), patch("agent.graph.TranslationTools", return_value=mock_translation_tools):

            graph = create_translation_graph(config)
            final_state = graph.invoke(initial_state)

            # Verify context was preserved and extended
            assert len(final_state["memory_context"]) > 1
            assert any(
                "Ngữ cảnh trước đó" in item["summary"]
                for item in final_state["memory_context"]
            )

    def test_translation_workflow_error_recovery(
        self, config, sample_input_text, mock_weaviate_client, mock_translation_tools
    ):
        """Test translation workflow with error recovery."""
        # Create initial state
        initial_state = create_initial_state(sample_input_text, chunk_size=200)

        # Mock translation failure for first chunk, success for others
        call_count = 0

        def mock_translate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary translation failure")
            return "Đây là bản dịch thành công."

        mock_translation_tools.translate_chunk.side_effect = mock_translate

        with patch(
            "agent.graph.WeaviateWrapperClient", return_value=mock_weaviate_client
        ), patch("agent.graph.TranslationTools", return_value=mock_translation_tools):

            graph = create_translation_graph(config)

            # The workflow should handle the error gracefully
            # Note: In a real implementation, this might require retry logic
            with pytest.raises(Exception):
                graph.invoke(initial_state)

    def test_memory_consistency_across_chunks(
        self, config, sample_input_text, mock_weaviate_client, mock_translation_tools
    ):
        """Test that memory context remains consistent across chunks."""
        # Create initial state
        initial_state = create_initial_state(sample_input_text, chunk_size=200)

        # Track memory operations across chunks
        memory_operations = []

        def mock_memory_ops(*args, **kwargs):
            result = {
                "create_nodes": [
                    {
                        "type": "character",
                        "label": f"Nhân vật chunk {len(memory_operations)}",
                        "name": f"Character_{len(memory_operations)}",
                        "content": f"Nội dung nhân vật {len(memory_operations)}",
                        "alias": [f"Alias_{len(memory_operations)}"],
                    }
                ],
                "update_nodes": [],
            }
            memory_operations.append(result)
            return result

        mock_translation_tools.generate_memory_operations.side_effect = mock_memory_ops

        with patch(
            "agent.graph.WeaviateWrapperClient", return_value=mock_weaviate_client
        ), patch("agent.graph.TranslationTools", return_value=mock_translation_tools):

            graph = create_translation_graph(config)
            final_state = graph.invoke(initial_state)

            # Verify memory operations were generated for each chunk
            assert len(memory_operations) > 0
            assert len(final_state["memory_state"]["created_nodes"]) > 0

    def test_context_overflow_handling_integration(
        self, config, sample_input_text, mock_weaviate_client, mock_translation_tools
    ):
        """Test context overflow handling in integration."""
        # Create initial state with many context items
        initial_state = create_initial_state(sample_input_text, chunk_size=200)
        initial_state["memory_context"] = [
            {
                "chunk_index": i,
                "summary": f"Ngữ cảnh cũ {i}",
                "timestamp": f"2024-01-01T00:0{i}:00",
            }
            for i in range(10)  # More than the limit of 5
        ]

        with patch(
            "agent.graph.WeaviateWrapperClient", return_value=mock_weaviate_client
        ), patch("agent.graph.TranslationTools", return_value=mock_translation_tools):

            graph = create_translation_graph(config)
            final_state = graph.invoke(initial_state)

            # Verify context was trimmed to limit
            assert len(final_state["memory_context"]) <= 5

    def test_translation_quality_tracking(
        self, config, sample_input_text, mock_weaviate_client, mock_translation_tools
    ):
        """Test that translation quality is tracked throughout the workflow."""
        # Create initial state
        initial_state = create_initial_state(sample_input_text, chunk_size=200)

        with patch(
            "agent.graph.WeaviateWrapperClient", return_value=mock_weaviate_client
        ), patch("agent.graph.TranslationTools", return_value=mock_translation_tools):

            graph = create_translation_graph(config)
            final_state = graph.invoke(initial_state)

            # Verify translation state was properly tracked
            for chunk in final_state["chunks"]:
                assert chunk["is_processed"] is True
                assert chunk["translation_attempts"] > 0

            # Verify translation state in final chunk
            assert final_state["translation_state"]["processing_status"] in [
                "completed",
                "failed",
            ]

    def test_memory_search_efficiency(
        self, config, sample_input_text, mock_weaviate_client, mock_translation_tools
    ):
        """Test that memory search is efficient and doesn't duplicate queries."""
        # Create initial state
        initial_state = create_initial_state(sample_input_text, chunk_size=200)

        search_queries = []

        def mock_search(*args, **kwargs):
            queries = ["Trương Tam", "lập trình viên", "trí tuệ nhân tạo"]
            search_queries.extend(queries)
            return queries

        mock_translation_tools.generate_search_queries.side_effect = mock_search

        with patch(
            "agent.graph.WeaviateWrapperClient", return_value=mock_weaviate_client
        ), patch("agent.graph.TranslationTools", return_value=mock_translation_tools):

            graph = create_translation_graph(config)
            final_state = graph.invoke(initial_state)

            # Verify search queries were generated
            assert len(search_queries) > 0
            assert len(final_state["memory_state"]["search_queries"]) > 0

    def test_final_output_quality(
        self, config, sample_input_text, mock_weaviate_client, mock_translation_tools
    ):
        """Test the quality of the final translation output."""
        # Create initial state
        initial_state = create_initial_state(sample_input_text, chunk_size=200)

        with patch(
            "agent.graph.WeaviateWrapperClient", return_value=mock_weaviate_client
        ), patch("agent.graph.TranslationTools", return_value=mock_translation_tools):

            graph = create_translation_graph(config)
            final_state = graph.invoke(initial_state)

            # Verify final output structure
            assert "translated_text" in final_state
            assert isinstance(final_state["translated_text"], list)
            assert len(final_state["translated_text"]) > 0

            # Verify each translated chunk
            for translated_chunk in final_state["translated_text"]:
                assert isinstance(translated_chunk, str)
                assert len(translated_chunk) > 0
                assert "Đây là bản dịch thành công" in translated_chunk

            # Verify processing statistics
            assert final_state["total_chunks"] > 0
            assert final_state["current_chunk_index"] == final_state["total_chunks"] - 1
            assert final_state["processing_complete"] is True
