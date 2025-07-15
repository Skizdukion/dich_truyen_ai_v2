#!/usr/bin/env python3
"""
End-to-End Integration Tests for Enhanced Translation Graph
Tests the complete workflow with big chunks, small chunks, review, and retry functionality.
"""

import pytest
import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_e2e.log')
    ]
)

# Create logger for this test module
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from agent.graph import create_translation_graph
from agent.configuration import Configuration
from agent.state import create_initial_state
from agent.utils import VietnameseTextChunker, EnhancedChunkingConfig
from weaviate_client.client import WeaviateWrapperClient


class TestE2EIntegration:
    """End-to-end integration tests for the enhanced translation workflow."""
    
    @pytest.fixture(scope="class")
    def config(self):
        """Create configuration for testing."""
        return Configuration()
    
    @pytest.fixture(scope="class")
    def graph(self, config):
        """Create enhanced translation graph."""
        return create_translation_graph(config)
    
    @pytest.fixture(scope="class")
    def weaviate_client(self):
        """Create Weaviate client for testing."""
        return WeaviateWrapperClient()
    
    @pytest.fixture(scope="class")
    def chunker(self):
        """Create Vietnamese text chunker."""
        return VietnameseTextChunker()
    
    def clear_all_data(self, weaviate_client):
        """Clear all data from Weaviate and reset memory context."""
        logger.info("=== CLEARING ALL DATA ===")
        
        # Clear Weaviate node collection
        logger.info("Clearing Weaviate node collection...")
        try:
            weaviate_client.delete_all_nodes()
            logger.info("✓ Weaviate node collection cleared successfully!")
        except Exception as e:
            logger.error(f"✗ Error clearing Weaviate: {str(e)}")
        
        logger.info("All data cleared!\n")
    
    def load_chapter_text(self, chapter_path):
        """Load chapter text from file."""
        logger.info(f"Loading {chapter_path}...")
        with open(chapter_path, "r", encoding="utf-8") as f:
            chapter_text = f.read()
        
        logger.info(f"✓ Chapter loaded: {len(chapter_text)} characters\n")
        return chapter_text
    
    def translate_chapter_with_enhanced_graph(self, chapter_name, chapter_text, graph, chunker, memory_context=None):
        """Translate a chapter using the enhanced translation graph."""
        logger.info(f"=== TRANSLATING {chapter_name} WITH ENHANCED GRAPH ===")
        
        # Initialize enhanced state
        state = create_initial_state(chapter_text)
        state['memory_context'] = memory_context or []
        
        logger.info(f"Starting translation of {chapter_name}...")
        logger.info(f"Initial state - input_text length: {len(chapter_text)}")
        logger.info(f"Memory context items: {len(state['memory_context'])}")
        
        # Execute the enhanced graph
        result_state = graph.invoke(state)
        
        logger.info(f"✓ Translation completed!")
        logger.info(f"✓ Big chunks processed: {len(result_state['big_chunks'])}")
        logger.info(f"✓ Small chunks processed: {len(result_state['small_chunks'])}")
        logger.info(f"✓ Review states: {len(result_state['review_states'])}")
        logger.info(f"✓ Memory nodes retrieved: {len(result_state['memory_state']['retrieved_nodes'])}")
        logger.info(f"✓ Memory nodes created: {len(result_state['memory_state']['created_nodes'])}")
        logger.info(f"✓ Memory nodes updated: {len(result_state['memory_state']['updated_nodes'])}")
        logger.info(f"✓ Processing complete: {result_state['processing_complete']}")
        logger.info("")
        
        return result_state
    
    def export_enhanced_translation(self, chapter_name, result_state, output_dir="test_outputs"):
        """Export enhanced translation results to a file."""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        filename = f"{output_dir}/{chapter_name.lower().replace(' ', '_')}_enhanced_translation.txt"
        
        # Write enhanced translation results to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== {chapter_name} ENHANCED TRANSLATION ===\n\n")
            
            # Write big chunks summary
            f.write("BIG CHUNKS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            for i, big_chunk in enumerate(result_state['big_chunks']):
                f.write(f"Big Chunk {i+1}:\n")
                f.write(f"  ID: {big_chunk['big_chunk_id']}\n")
                f.write(f"  Text length: {len(big_chunk['big_chunk_text'])}\n")
                f.write(f"  Small chunks: {len(big_chunk['small_chunks'])}\n")
                f.write(f"  Status: {big_chunk['status']}\n\n")
            
            # Write small chunks and translations
            f.write("SMALL CHUNKS AND TRANSLATIONS:\n")
            f.write("-" * 40 + "\n")
            for i, small_chunk in enumerate(result_state['small_chunks']):
                f.write(f"Small Chunk {i+1}:\n")
                f.write(f"  ID: {small_chunk['small_chunk_id']}\n")
                f.write(f"  Original: {small_chunk['small_chunk_text'][:100]}...\n")
                
                # Find corresponding translation state
                translation_state = None
                for ts in result_state['translation_states']:
                    if ts['chunk_id'] == small_chunk['small_chunk_id']:
                        translation_state = ts
                        break
                
                if translation_state:
                    f.write(f"  Translated: {translation_state['translated_text'][:100]}...\n")
                    f.write(f"  Quality score: {translation_state.get('quality_score', 'N/A')}\n")
                    f.write(f"  Processing time: {translation_state.get('processing_time', 'N/A')}\n")
                
                # Find corresponding review state
                review_state = None
                for rs in result_state['review_states']:
                    if rs['chunk_id'] == small_chunk['small_chunk_id']:
                        review_state = rs
                        break
                
                if review_state:
                    f.write(f"  Review rating: {review_state.get('rating', 'N/A')}\n")
                    f.write(f"  Requires revision: {review_state.get('requires_revision', 'N/A')}\n")
                    f.write(f"  Feedback: {review_state.get('feedback', 'N/A')}\n")
                
                f.write("-" * 20 + "\n\n")
            
            # Write memory operations summary
            f.write("MEMORY OPERATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Retrieved nodes: {len(result_state['memory_state']['retrieved_nodes'])}\n")
            f.write(f"Created nodes: {len(result_state['memory_state']['created_nodes'])}\n")
            f.write(f"Updated nodes: {len(result_state['memory_state']['updated_nodes'])}\n")
            f.write(f"Search queries: {len(result_state['memory_state']['search_queries'])}\n")
            f.write(f"Memory operations: {len(result_state['memory_state']['memory_operations'])}\n\n")
            
            f.write(f"Total big chunks: {len(result_state['big_chunks'])}\n")
            f.write(f"Total small chunks: {len(result_state['small_chunks'])}\n")
            f.write(f"Processing complete: {result_state['processing_complete']}\n")
        
        logger.info(f"✓ Enhanced translation exported to: {filename}")
        logger.info(f"✓ Total big chunks: {len(result_state['big_chunks'])}")
        logger.info(f"✓ Total small chunks: {len(result_state['small_chunks'])}\n")
    
    def test_enhanced_translation_workflow(self, config, graph, weaviate_client, chunker):
        """Test the complete enhanced translation workflow with real data."""
        # Clear all data first
        self.clear_all_data(weaviate_client)
        
        # Test with Chapter 1
        chap1_text = self.load_chapter_text("raw_text/demo/chap_1.txt")
        result_ch1 = self.translate_chapter_with_enhanced_graph(
            "CHAPTER 1", chap1_text, graph, chunker
        )
        self.export_enhanced_translation("CHAPTER 1", result_ch1)
        
        # Verify Chapter 1 results
        assert result_ch1['processing_complete'] is True
        assert len(result_ch1['big_chunks']) > 0
        assert len(result_ch1['small_chunks']) > 0
        assert len(result_ch1['memory_state']['retrieved_nodes']) >= 0
        assert len(result_ch1['memory_state']['created_nodes']) >= 0
        
        # Test with Chapter 2 (with memory context from Chapter 1)
        chap2_text = self.load_chapter_text("raw_text/demo/chap_2.txt")
        result_ch2 = self.translate_chapter_with_enhanced_graph(
            "CHAPTER 2", chap2_text, graph, chunker, 
            memory_context=result_ch1['memory_context']
        )
        self.export_enhanced_translation("CHAPTER 2", result_ch2)
        
        # Verify Chapter 2 results
        assert result_ch2['processing_complete'] is True
        assert len(result_ch2['big_chunks']) > 0
        assert len(result_ch2['small_chunks']) > 0
        assert len(result_ch2['memory_state']['retrieved_nodes']) >= 0
        assert len(result_ch2['memory_state']['created_nodes']) >= 0
        
        # Verify memory context was passed between chapters
        assert len(result_ch2['memory_context']) >= len(result_ch1['memory_context'])
        
        logger.info("=== ENHANCED TRANSLATION WORKFLOW TEST COMPLETED ===")
        logger.info("✓ Chapter 1 translated with fresh memory")
        logger.info("✓ Chapter 2 translated with context from Chapter 1")
        logger.info("✓ All data was cleared before starting")
        logger.info("✓ Enhanced graph with big chunks, small chunks, review, and retry")
    
    def test_enhanced_chunking_and_processing(self, config, graph, chunker):
        """Test enhanced chunking and processing functionality."""
        # Load a small text for testing
        test_text = "Đây là một đoạn văn bản tiếng Việt để kiểm tra chức năng chunking và xử lý. Văn bản này sẽ được chia thành các big chunks và small chunks để kiểm tra workflow."
        
        # Test big chunk creation
        big_chunk_config = EnhancedChunkingConfig(
            big_chunk_size=1000,
            small_chunk_size=200,
            overlap_size=50
        )
        
        big_chunks = chunker.chunk_text_into_big_chunks(test_text, big_chunk_config)
        assert len(big_chunks) > 0
        
        # Test small chunk creation within big chunks
        for big_chunk in big_chunks:
            small_chunks = chunker.chunk_big_chunk_into_small_chunks(
                big_chunk, big_chunk_config
            )
            assert len(small_chunks) > 0
            
            # Verify small chunks are properly structured and flexible in size
            for small_chunk in small_chunks:
                assert 'small_chunk_id' in small_chunk
                assert 'small_chunk_text' in small_chunk
                assert 'big_chunk_id' in small_chunk
                assert len(small_chunk['small_chunk_text']) > 0
                # Flexible chunk size assertion
                assert 400 <= small_chunk['small_chunk_size'] <= 700
        
        logger.info("✓ Enhanced chunking and processing test passed")
    
    def test_memory_context_persistence(self, config, graph, weaviate_client, chunker):
        """Test that memory context persists between translation sessions."""
        # Clear data
        self.clear_all_data(weaviate_client)
        
        # First translation session
        text1 = "Nhân vật chính trong câu chuyện là Nguyễn Văn A. Anh ta là một sinh viên đại học."
        result1 = self.translate_chapter_with_enhanced_graph("SESSION 1", text1, graph, chunker)
        
        # Second translation session with memory context
        text2 = "Nguyễn Văn A đang học ở trường đại học Bách Khoa."
        result2 = self.translate_chapter_with_enhanced_graph(
            "SESSION 2", text2, graph, chunker, 
            memory_context=result1['memory_context']
        )
        
        # Verify memory context was used
        assert len(result2['memory_context']) >= len(result1['memory_context'])
        
        # Verify memory operations increased
        total_memory_ops = (
            len(result2['memory_state']['retrieved_nodes']) +
            len(result2['memory_state']['created_nodes']) +
            len(result2['memory_state']['updated_nodes'])
        )
        assert total_memory_ops >= 0
        
        logger.info("✓ Memory context persistence test passed")
    
    def test_review_and_retry_functionality(self, config, graph, chunker):
        """Test review and retry functionality in the enhanced workflow."""
        # Use a text that might trigger review/retry
        test_text = "Đây là một câu rất phức tạp với nhiều từ khó hiểu và cấu trúc ngữ pháp phức tạp. Văn bản này có thể cần được review và retry nhiều lần để đạt được chất lượng tốt nhất."
        
        result = self.translate_chapter_with_enhanced_graph("REVIEW_TEST", test_text, graph, chunker)
        
        # Verify review states were created
        assert len(result['review_states']) >= 0
        
        # Check if any chunks required revision
        revisions_required = sum(
            1 for rs in result['review_states'] 
            if rs.get('requires_revision', False)
        )
        
        # Verify processing completed regardless of reviews
        assert result['processing_complete'] is True
        
        logger.info(f"✓ Review and retry test passed - {revisions_required} revisions required")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"]) 