import pytest
from typing import List
from src.agent.utils import (
    VietnameseTextChunker,
    ChunkingConfig,
    chunk_vietnamese_text,
    analyze_chunk_quality,
    validate_chunks
)
from src.agent.state import ChunkState


class TestChunkingConfig:
    """Test cases for ChunkingConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ChunkingConfig()
        assert config.max_chunk_size == 6000
        assert config.min_chunk_size == 2000
        assert config.overlap_size == 0
        assert config.preserve_sentences is True
        assert config.preserve_paragraphs is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ChunkingConfig(
            max_chunk_size=500,
            min_chunk_size=100,
            overlap_size=50,
            preserve_sentences=False,
            preserve_paragraphs=False
        )
        assert config.max_chunk_size == 500
        assert config.min_chunk_size == 100
        assert config.overlap_size == 50
        assert config.preserve_sentences is False
        assert config.preserve_paragraphs is False


class TestVietnameseTextChunker:
    """Test cases for VietnameseTextChunker"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.chunker = VietnameseTextChunker()
        self.small_config = ChunkingConfig(max_chunk_size=300, min_chunk_size=50)
        self.small_chunker = VietnameseTextChunker(self.small_config)
    
    def test_empty_text(self):
        """Test chunking empty text"""
        chunks = self.chunker.chunk_text("")
        assert chunks == []
        
        chunks = self.chunker.chunk_text("   ")
        assert chunks == []
    
    def test_single_sentence(self):
        """Test chunking a single sentence"""
        text = "Đây là một câu tiếng Việt đơn giản."
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0]['chunk_text'] == text
        assert chunks[0]['chunk_index'] == 0
        assert chunks[0]['is_processed'] is False
    
    def test_multiple_sentences(self):
        """Test chunking multiple sentences"""
        text = "Câu thứ nhất. Câu thứ hai. Câu thứ ba."
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) == 1  # Should be one chunk since it's small
        assert chunks[0]['chunk_text'] == text
    
    def test_large_text_chunking(self):
        """Test chunking large text that needs to be split"""
        # Use a small chunking config for this test
        small_config = ChunkingConfig(max_chunk_size=500, min_chunk_size=100)
        chunker = VietnameseTextChunker(small_config)
        sentences = [
            "Đây là câu đầu tiên của đoạn văn dài với nhiều từ ngữ để tăng kích thước.",
            "Câu thứ hai tiếp tục ý tưởng từ câu đầu và mở rộng thêm về chủ đề chính.",
            "Câu thứ ba mở rộng thêm về chủ đề này với nhiều chi tiết và ví dụ cụ thể.",
            "Câu thứ tư đưa ra ví dụ cụ thể và giải thích chi tiết về các khía cạnh khác nhau.",
            "Câu thứ năm kết luận đoạn văn này với tổng kết và nhận xét cuối cùng."
        ]
        text = " ".join(sentences * 100)  # Repeat many times to exceed 500 chars
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk['chunk_text']) <= chunker.config.max_chunk_size for chunk in chunks)
        assert all(len(chunk['chunk_text']) >= chunker.config.min_chunk_size for chunk in chunks)
    
    def test_paragraph_preservation(self):
        """Test that paragraphs are preserved when possible"""
        # Use a small chunking config for this test
        small_config = ChunkingConfig(max_chunk_size=500, min_chunk_size=100)
        chunker = VietnameseTextChunker(small_config)
        paragraph1 = "Đoạn văn thứ nhất với nhiều câu dài để đạt được kích thước tối thiểu. " * 20
        paragraph2 = "Đoạn văn thứ hai cũng có nhiều nội dung để đảm bảo kích thước phù hợp. " * 20
        paragraph3 = "Đoạn văn thứ ba hoàn thành bộ ba đoạn văn với đủ nội dung. " * 20
        
        text = paragraph1 + "\n\n" + paragraph2 + "\n\n" + paragraph3
        chunks = chunker.chunk_text(text)
        
        # Should have at least 3 chunks, and each paragraph should be present in at least one chunk
        assert len(chunks) >= 3
        assert any("Đoạn văn thứ nhất" in chunk['chunk_text'] for chunk in chunks)
        assert any("Đoạn văn thứ hai" in chunk['chunk_text'] for chunk in chunks)
        assert any("Đoạn văn thứ ba" in chunk['chunk_text'] for chunk in chunks)
    
    def test_sentence_boundary_detection(self):
        """Test Vietnamese sentence boundary detection"""
        # Create a longer text that will actually be split
        text = "Câu hỏi? " * 20 + "Câu cảm thán! " * 20 + "Câu bình thường. " * 20 + "Câu với dấu chấm. " * 20
        chunks = self.small_chunker.chunk_text(text)
        
        # Should break at sentence boundaries
        assert len(chunks) > 1
        for chunk in chunks:
            chunk_text = chunk['chunk_text']
            # Each chunk should end with proper sentence ending
            assert any(chunk_text.endswith(end) for end in ['.', '!', '?', '。', '！', '？'])
    
    def test_phrase_boundary_detection(self):
        """Test phrase boundary detection when sentences are too long"""
        # Create a longer text that will actually be split
        text = "Đây là một câu rất dài, có nhiều dấu phẩy; và dấu chấm phẩy: để tạo ra các ranh giới câu. " * 30
        chunks = self.small_chunker.chunk_text(text)
        
        assert len(chunks) > 1
        for chunk in chunks:
            chunk_text = chunk['chunk_text']
            # Should break at phrase boundaries
            assert any(punct in chunk_text for punct in [',', '，', ';', '；', ':', '：'])
    
    def test_text_normalization(self):
        """Test text normalization functionality"""
        text = "  Đây   là   văn   bản   có   nhiều   khoảng   trắng.  \n\n  Và   dòng   mới.  "
        chunks = self.chunker.chunk_text(text)
        
        # Should normalize whitespace
        assert len(chunks) > 0
        for chunk in chunks:
            chunk_text = chunk['chunk_text']
            # Should not have excessive whitespace
            assert '   ' not in chunk_text
            assert chunk_text.strip() == chunk_text
    
    def test_merge_small_chunks(self):
        """Test merging of small chunks"""
        # Create small chunks
        small_chunks = [
            ChunkState(chunk_index=0, chunk_text="Chunk nhỏ 1", chunk_size=50, is_processed=False, translation_attempts=0, max_attempts=3),
            ChunkState(chunk_index=1, chunk_text="Chunk nhỏ 2", chunk_size=50, is_processed=False, translation_attempts=0, max_attempts=3),
            ChunkState(chunk_index=2, chunk_text="Chunk lớn hơn", chunk_size=200, is_processed=False, translation_attempts=0, max_attempts=3),
        ]
        
        merged = self.chunker.merge_small_chunks(small_chunks)
        
        # Should merge small chunks together
        assert len(merged) < len(small_chunks)
        assert any("Chunk nhỏ 1" in chunk['chunk_text'] and "Chunk nhỏ 2" in chunk['chunk_text'] for chunk in merged)
    
    def test_vietnamese_punctuation(self):
        """Test handling of Vietnamese punctuation marks"""
        text = "Câu tiếng Việt với dấu câu: dấu chấm. dấu phẩy, dấu chấm phẩy; dấu hai chấm: và dấu chấm than!"
        chunks = self.small_chunker.chunk_text(text)
        
        assert len(chunks) > 0
        # Should handle Vietnamese punctuation correctly
        for chunk in chunks:
            assert len(chunk['chunk_text']) > 0
    
    def test_chunk_state_creation(self):
        """Test proper ChunkState creation"""
        text = "Test text"
        chunk = self.chunker._create_chunk_state(0, text, len(text))
        
        assert isinstance(chunk, dict)  # TypedDict behaves like dict
        assert chunk['chunk_index'] == 0
        assert chunk['chunk_text'] == text
        assert chunk['chunk_size'] == len(text)
        assert chunk['is_processed'] is False
        assert chunk['translation_attempts'] == 0
        assert chunk['max_attempts'] == 3


class TestChunkingFunctions:
    """Test cases for chunking utility functions"""
    
    def test_chunk_vietnamese_text(self):
        """Test convenience function for chunking"""
        text = "Câu thứ nhất. Câu thứ hai. Câu thứ ba."
        chunks = chunk_vietnamese_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all('chunk_text' in chunk for chunk in chunks)
    
    def test_chunk_vietnamese_text_with_config(self):
        """Test chunking with custom configuration"""
        text = "Câu dài " * 50  # Create long text
        config = ChunkingConfig(max_chunk_size=200, min_chunk_size=50)
        chunks = chunk_vietnamese_text(text, config)
        
        assert len(chunks) > 1
        assert all(len(chunk['chunk_text']) <= 200 for chunk in chunks)
    
    def test_analyze_chunk_quality_empty(self):
        """Test quality analysis with empty chunks"""
        analysis = analyze_chunk_quality([])
        
        assert analysis['total_chunks'] == 0
        assert analysis['average_chunk_size'] == 0
        assert analysis['min_chunk_size'] == 0
        assert analysis['max_chunk_size'] == 0
    
    def test_analyze_chunk_quality(self):
        """Test quality analysis with actual chunks"""
        chunks = [
            ChunkState(chunk_index=0, chunk_text="Short", chunk_size=5, is_processed=False, translation_attempts=0, max_attempts=3),
            ChunkState(chunk_index=1, chunk_text="Medium length text", chunk_size=18, is_processed=False, translation_attempts=0, max_attempts=3),
            ChunkState(chunk_index=2, chunk_text="Very long text that exceeds normal limits", chunk_size=40, is_processed=False, translation_attempts=0, max_attempts=3),
        ]
        
        analysis = analyze_chunk_quality(chunks)
        
        assert analysis['total_chunks'] == 3
        assert analysis['min_chunk_size'] == 5
        assert analysis['max_chunk_size'] == 41  # "Very long text that exceeds normal limits" is 41 chars
        assert 'size_distribution' in analysis
        assert analysis['size_distribution']['small (<500)'] == 3  # All chunks are small
    
    def test_validate_chunks_empty(self):
        """Test validation with empty chunks"""
        errors = validate_chunks([])
        assert len(errors) == 1
        assert "No chunks provided" in errors[0]
    
    def test_validate_chunks_valid(self):
        """Test validation with valid chunks"""
        chunks = [
            ChunkState(chunk_index=0, chunk_text="Valid chunk", chunk_size=11, is_processed=False, translation_attempts=0, max_attempts=3),
            ChunkState(chunk_index=1, chunk_text="Another valid chunk", chunk_size=19, is_processed=False, translation_attempts=0, max_attempts=3),
        ]
        
        errors = validate_chunks(chunks)
        assert len(errors) == 0
    
    def test_validate_chunks_invalid_index(self):
        """Test validation with invalid chunk indices"""
        chunks = [
            ChunkState(chunk_index=1, chunk_text="Wrong index", chunk_size=12, is_processed=False, translation_attempts=0, max_attempts=3),
        ]
        
        errors = validate_chunks(chunks)
        assert len(errors) == 1
        assert "incorrect index" in errors[0]
    
    def test_validate_chunks_empty_text(self):
        """Test validation with empty chunk text"""
        chunks = [
            ChunkState(chunk_index=0, chunk_text="", chunk_size=0, is_processed=False, translation_attempts=0, max_attempts=3),
        ]
        
        errors = validate_chunks(chunks)
        assert len(errors) == 2  # Both empty text and empty combined text
        assert any("is empty" in error for error in errors)


class TestChunkingEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_very_long_sentence(self):
        """Test chunking a very long sentence without natural breaks"""
        long_sentence = "Đây là một câu rất dài " * 100
        chunker = VietnameseTextChunker(ChunkingConfig(max_chunk_size=500))
        chunks = chunker.chunk_text(long_sentence)
        
        assert len(chunks) > 1
        # Should break at word boundaries when no sentence/phrase boundaries exist
        for chunk in chunks:
            assert len(chunk['chunk_text']) <= 500
    
    def test_mixed_language_text(self):
        """Test chunking text with mixed Vietnamese and English"""
        mixed_text = "This is English text. Đây là tiếng Việt. More English here. Và tiếng Việt nữa."
        chunks = chunk_vietnamese_text(mixed_text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk['chunk_text']) > 0
    
    def test_special_characters(self):
        """Test chunking text with special characters"""
        special_text = "Text với ký tự đặc biệt: @#$%^&*()_+{}|:<>?[]\\;'\",./"
        chunks = chunk_vietnamese_text(special_text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk['chunk_text']) > 0
    
    def test_numbers_and_symbols(self):
        """Test chunking text with numbers and mathematical symbols"""
        number_text = "Số 123 và 456. Phép tính: 2 + 2 = 4. Đơn vị: 100%."
        chunks = chunk_vietnamese_text(number_text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk['chunk_text']) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 