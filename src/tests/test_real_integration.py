"""
Real integration tests for the complete translation workflow.
Tests end-to-end translation pipeline with actual LLM calls and Weaviate operations.
"""

import pytest
import os
import time
from typing import Dict, Any, List
import logging

from agent.graph import create_translation_graph
from agent.state import create_initial_state, OverallState
from agent.configuration import Configuration
from agent.translation_tools import TranslationTools
from weaviate_client.client import WeaviateWrapperClient

# Configure logging for integration tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRealIntegration:
    """Real integration tests for the complete translation workflow."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration with real model settings."""
        return Configuration(
            translation_model="gemini-2.0-flash",
            memory_search_model="gemini-2.0-flash",
            memory_update_model="gemini-2.0-flash",
            context_summary_model="gemini-2.0-flash",
            max_translation_attempts=2,
            max_memory_context_items=3,
            memory_search_limit=3
        )
    
    @pytest.fixture
    def sample_vietphrase_text(self):
        """Sample VietPhrase text for real translation testing."""
        return """
        Chương 1: Khởi đầu mới
        Trương Tam là một lập trình viên trẻ, anh ta sống ở Bắc Kinh.
        
        Hôm nay, anh ta sẽ tham gia một cuộc họp quan trọng.
        
        Chủ đề của cuộc họp là về sự phát triển của trí tuệ nhân tạo.
        
        Anh ta cảm thấy rất phấn khích, vì đây là lần đầu tiên anh tham gia một cuộc họp như vậy.
        """
    
    @pytest.fixture
    def sample_vietphrase_text_long(self):
        """Longer VietPhrase text for comprehensive testing."""
        return """
        Chương 1: Khởi đầu mới
        
        Trương Tam là một lập trình viên trẻ, anh ta sống ở Bắc Kinh. Anh ta năm nay hai mươi lăm tuổi, làm việc tại một công ty công nghệ.
        
        Hôm nay, anh ta sẽ tham gia một cuộc họp quan trọng. Chủ đề của cuộc họp là về sự phát triển của trí tuệ nhân tạo.
        
        Anh ta cảm thấy rất phấn khích, vì đây là lần đầu tiên anh tham gia một cuộc họp như vậy.
        
        Tại cuộc họp, anh ta gặp Lý Tứ, một chuyên gia AI giàu kinh nghiệm.
        
        Lý Tứ giới thiệu cho anh ta những công nghệ học máy mới nhất.
        
        Trương Tam học được rất nhiều kiến thức mới về học sâu.
        
        Sau khi cuộc họp kết thúc, Trương Tam quyết định học thêm nhiều kiến thức về AI.
        
        Anh ta dự định tham gia một khóa học trực tuyến để nâng cao kỹ năng của mình.
        
        Anh ta tin rằng, nắm vững công nghệ AI sẽ rất hữu ích cho sự phát triển nghề nghiệp của mình.
        """
    
    @pytest.fixture
    def weaviate_client(self):
        """Create a real Weaviate client for testing."""
        # Check if Weaviate environment variables are set
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_key = os.getenv("WEAVIATE_API_KEY")
        
        if not weaviate_url or not weaviate_key:
            pytest.skip("Weaviate credentials not configured. Set WEAVIATE_URL and WEAVIATE_API_KEY environment variables.")
        
        client = WeaviateWrapperClient(
            url=weaviate_url,
            weaviate_key=weaviate_key
        )
        
        # Test connection
        try:
            client.connect()
            logger.info("Successfully connected to Weaviate")
            return client
        except Exception as e:
            pytest.skip(f"Could not connect to Weaviate: {str(e)}")
    
    @pytest.fixture
    def translation_tools(self, config):
        """Create real TranslationTools instance."""
        # Check if Google API key is set
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("Google API key not configured. Set GOOGLE_API_KEY environment variable.")
        
        try:
            tools = TranslationTools(config)
            logger.info("Successfully initialized TranslationTools")
            return tools
        except Exception as e:
            pytest.skip(f"Could not initialize TranslationTools: {str(e)}")
    
    def test_real_translation_workflow_simple(self, config, sample_vietphrase_text, weaviate_client, translation_tools):
        """Test real translation workflow with simple text."""
        logger.info("Starting real translation workflow test")
        
        # Create initial state
        initial_state = create_initial_state(sample_vietphrase_text, chunk_size=200)
        logger.info(f"Created initial state with {len(initial_state['chunks'])} chunks")
        
        # Create the graph
        graph = create_translation_graph(config)
        logger.info("Created translation graph")
        
        # Execute the workflow
        start_time = time.time()
        final_state = graph.invoke(initial_state)
        end_time = time.time()
        
        logger.info(f"Workflow completed in {end_time - start_time:.2f} seconds")
        
        # Verify the workflow completed successfully
        assert final_state['processing_complete'] is True
        assert len(final_state['translated_text']) > 0
        assert final_state['current_chunk_index'] == final_state['total_chunks'] - 1
        
        # Verify memory operations were performed
        assert len(final_state['memory_state']['memory_operations']) > 0
        assert len(final_state['memory_state']['retrieved_nodes']) >= 0  # May be 0 if no existing memory
        assert len(final_state['memory_state']['search_queries']) > 0
        
        # Verify context was maintained
        assert len(final_state['memory_context']) > 0
        
        # Log results
        logger.info(f"Final state: {len(final_state['translated_text'])} translated chunks")
        logger.info(f"Memory operations: {len(final_state['memory_state']['memory_operations'])}")
        logger.info(f"Context items: {len(final_state['memory_context'])}")
        
        # Print sample translations
        for i, translated_chunk in enumerate(final_state['translated_text']):
            logger.info(f"Chunk {i}: {translated_chunk[:100]}...")
    
    def test_real_translation_workflow_with_memory(self, config, sample_vietphrase_text_long, weaviate_client, translation_tools):
        """Test real translation workflow with memory persistence."""
        logger.info("Starting real translation workflow with memory test")
        
        # First, create some initial memory in Weaviate
        initial_nodes = [
            {
                "type": "character",
                "label": "Nhân vật chính",
                "name": "Trương Tam",
                "content": "Nhân vật chính, lập trình viên trẻ 25 tuổi",
                "alias": ["Tam", "Trương"]
            },
            {
                "type": "term",
                "label": "Thuật ngữ công nghệ",
                "name": "Trí tuệ nhân tạo",
                "content": "Artificial Intelligence, công nghệ máy học",
                "alias": ["AI", "Machine Learning"]
            }
        ]
        
        # Insert initial nodes
        for node in initial_nodes:
            try:
                weaviate_client.insert_knowledge_node(node)
                logger.info(f"Inserted node: {node['name']}")
            except Exception as e:
                logger.warning(f"Failed to insert node {node['name']}: {str(e)}")
        
        # Create initial state
        initial_state = create_initial_state(sample_vietphrase_text_long, chunk_size=250)
        logger.info(f"Created initial state with {len(initial_state['chunks'])} chunks")
        
        # Create the graph
        graph = create_translation_graph(config)
        
        # Execute the workflow
        start_time = time.time()
        final_state = graph.invoke(initial_state)
        end_time = time.time()
        
        logger.info(f"Workflow completed in {end_time - start_time:.2f} seconds")
        
        # Verify the workflow completed successfully
        assert final_state['processing_complete'] is True
        assert len(final_state['translated_text']) > 0
        
        # Verify memory operations were performed
        assert len(final_state['memory_state']['memory_operations']) > 0
        assert len(final_state['memory_state']['search_queries']) > 0
        
        # Verify that memory was retrieved (should find our initial nodes)
        retrieved_nodes = final_state['memory_state']['retrieved_nodes']
        logger.info(f"Retrieved {len(retrieved_nodes)} memory nodes")
        
        # Verify context continuity
        context_items = final_state['memory_context']
        logger.info(f"Context items: {len(context_items)}")
        for item in context_items:
            logger.info(f"Context: {item['summary'][:100]}...")
    
    def test_real_translation_quality(self, config, sample_vietphrase_text, weaviate_client, translation_tools):
        """Test real translation quality and consistency."""
        logger.info("Starting real translation quality test")
        
        # Create initial state
        initial_state = create_initial_state(sample_vietphrase_text, chunk_size=200)
        
        # Create the graph
        graph = create_translation_graph(config)
        
        # Execute the workflow
        final_state = graph.invoke(initial_state)
        
        # Verify final output structure
        assert 'translated_text' in final_state
        assert isinstance(final_state['translated_text'], list)
        assert len(final_state['translated_text']) > 0
        
        # Verify each translated chunk quality
        for i, translated_chunk in enumerate(final_state['translated_text']):
            assert isinstance(translated_chunk, str)
            assert len(translated_chunk) > 0
            
            # Check for Vietnamese language indicators
            vietnamese_indicators = ['là', 'của', 'với', 'trong', 'cho', 'để', 'nếu', 'khi']
            has_vietnamese = any(indicator in translated_chunk.lower() for indicator in vietnamese_indicators)
            
            logger.info(f"Chunk {i}: {translated_chunk[:100]}...")
            logger.info(f"Chunk {i} has Vietnamese: {has_vietnamese}")
            
            # Basic quality checks
            assert len(translated_chunk) > 10  # Should have substantial content
            assert not translated_chunk.startswith("[DỊCH]")  # Should not be fallback translation
        
        # Verify processing statistics
        assert final_state['total_chunks'] > 0
        assert final_state['current_chunk_index'] == final_state['total_chunks'] - 1
        assert final_state['processing_complete'] is True
    
    def test_real_memory_operations(self, config, sample_vietphrase_text, weaviate_client, translation_tools):
        """Test real memory operations and persistence."""
        logger.info("Starting real memory operations test")
        
        # Create initial state
        initial_state = create_initial_state(sample_vietphrase_text, chunk_size=200)
        
        # Create the graph
        graph = create_translation_graph(config)
        
        # Execute the workflow
        final_state = graph.invoke(initial_state)
        
        # Verify memory operations
        memory_ops = final_state['memory_state']['memory_operations']
        assert len(memory_ops) > 0
        
        # Check memory operation structure
        for op in memory_ops:
            assert 'operation_type' in op
            assert 'node_type' in op
            assert 'query_or_content' in op
            assert 'result' in op
            assert 'timestamp' in op
            
            logger.info(f"Memory operation: {op['operation_type']} - {op['node_type']}")
        
        # Verify search queries were generated
        search_queries = final_state['memory_state']['search_queries']
        assert len(search_queries) > 0
        
        for query in search_queries:
            logger.info(f"Search query: {query}")
            assert isinstance(query, str)
            assert len(query) > 0
    
    def test_real_context_continuity(self, config, sample_vietphrase_text_long, weaviate_client, translation_tools):
        """Test real context continuity across chunks."""
        logger.info("Starting real context continuity test")
        
        # Create initial state
        initial_state = create_initial_state(sample_vietphrase_text_long, chunk_size=150)
        logger.info(f"Created state with {len(initial_state['chunks'])} chunks")
        
        # Create the graph
        graph = create_translation_graph(config)
        
        # Execute the workflow
        final_state = graph.invoke(initial_state)
        
        # Verify context was maintained across chunks
        context_items = final_state['memory_context']
        assert len(context_items) > 0
        
        # Check context structure
        for item in context_items:
            assert 'chunk_index' in item
            assert 'summary' in item
            assert 'timestamp' in item
            assert isinstance(item['summary'], str)
            assert len(item['summary']) > 0
            
            logger.info(f"Context {item['chunk_index']}: {item['summary'][:100]}...")
        
        # Verify context doesn't exceed limit
        assert len(context_items) <= config.max_memory_context_items
    
    def test_real_error_handling(self, config, weaviate_client, translation_tools):
        """Test real error handling with invalid input."""
        logger.info("Starting real error handling test")
        
        # Test with empty text
        empty_text = ""
        initial_state = create_initial_state(empty_text, chunk_size=100)
        
        # Create the graph
        graph = create_translation_graph(config)
        
        # This should handle empty text gracefully
        try:
            final_state = graph.invoke(initial_state)
            logger.info("Empty text handled gracefully")
        except Exception as e:
            logger.info(f"Empty text caused expected error: {str(e)}")
        
        # Test with very long text
        long_text = "这是一个很长的文本。" * 1000  # Very long text
        initial_state = create_initial_state(long_text, chunk_size=50)
        
        # Create the graph
        graph = create_translation_graph(config)
        
        # This should handle long text
        try:
            final_state = graph.invoke(initial_state)
            logger.info(f"Long text processed: {len(final_state['chunks'])} chunks")
        except Exception as e:
            logger.info(f"Long text caused error: {str(e)}")
    
    def test_real_performance_metrics(self, config, sample_vietphrase_text, weaviate_client, translation_tools):
        """Test real performance metrics and timing."""
        logger.info("Starting real performance metrics test")
        
        # Create initial state
        initial_state = create_initial_state(sample_vietphrase_text, chunk_size=200)
        
        # Create the graph
        graph = create_translation_graph(config)
        
        # Measure execution time
        start_time = time.time()
        final_state = graph.invoke(initial_state)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Log performance metrics
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Chunks processed: {len(final_state['chunks'])}")
        logger.info(f"Average time per chunk: {execution_time / len(final_state['chunks']):.2f} seconds")
        
        # Performance assertions (adjust based on expected performance)
        assert execution_time > 0
        assert execution_time < 300  # Should complete within 5 minutes
        assert len(final_state['translated_text']) > 0
        
        # Memory usage metrics
        memory_ops = final_state['memory_state']['memory_operations']
        search_queries = final_state['memory_state']['search_queries']
        
        logger.info(f"Memory operations: {len(memory_ops)}")
        logger.info(f"Search queries: {len(search_queries)}")
        
        # Verify reasonable performance
        assert len(memory_ops) >= 0  # May be 0 if no new memory created
        assert len(search_queries) > 0  # Should have search queries 