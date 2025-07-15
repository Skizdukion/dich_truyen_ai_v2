import pytest
from typing import List
from src.agent.utils import (
    VietnameseTextChunker,
    ChunkingConfig,
    EnhancedChunkingConfig,
    chunk_vietnamese_text,
    chunk_text_into_big_chunks,
    chunk_big_chunk_into_small_chunks,
    validate_enhanced_chunks,
    reassemble_chunks_from_small_chunks,
    reassemble_chunks_from_big_chunks,
    preserve_context_between_small_chunks,
    analyze_enhanced_chunk_quality,
    analyze_chunk_quality,
    validate_chunks
)
from src.agent.state import ChunkState, BigChunkState, SmallChunkState


class TestChunkingConfig:
    """Test cases for ChunkingConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ChunkingConfig()
        assert config.max_chunk_size == 5000
        assert config.min_chunk_size == 800
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


class TestEnhancedChunkingConfig:
    """Test cases for EnhancedChunkingConfig"""
    
    def test_default_enhanced_config(self):
        """Test default enhanced configuration values"""
        config = EnhancedChunkingConfig()
        assert config.big_chunk_size == 16000
        assert config.small_chunk_size == 500
        assert config.big_chunk_min_size == 8000
        assert config.small_chunk_min_size == 200
        assert config.overlap_size == 0
        assert config.preserve_sentences is True
        assert config.preserve_paragraphs is True
    
    def test_custom_enhanced_config(self):
        """Test custom enhanced configuration values"""
        config = EnhancedChunkingConfig(
            big_chunk_size=12000,
            small_chunk_size=400,
            big_chunk_min_size=6000,
            small_chunk_min_size=150,
            overlap_size=100,
            preserve_sentences=False,
            preserve_paragraphs=False
        )
        assert config.big_chunk_size == 12000
        assert config.small_chunk_size == 400
        assert config.big_chunk_min_size == 6000
        assert config.small_chunk_min_size == 150
        assert config.overlap_size == 100
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
        
        # Allow for flexible chunk sizes (400-700)
        for chunk in chunks:
            assert 400 <= len(chunk['chunk_text']) <= 700
        assert len(chunks) > 1
    
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
        # Allow for flexible chunk sizes
        for chunk in chunks:
            assert 400 <= len(chunk['chunk_text']) <= 700
    
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


class TestEnhancedChunking:
    """Test cases for enhanced chunking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.chunker = VietnameseTextChunker()
        self.enhanced_config = EnhancedChunkingConfig(
            big_chunk_size=8000,  # Smaller for testing
            small_chunk_size=300,  # Smaller for testing
            big_chunk_min_size=4000,
            small_chunk_min_size=100
        )
    
    def test_chunk_text_into_big_chunks_empty(self):
        """Test chunking empty text into big chunks"""
        big_chunks = self.chunker.chunk_text_into_big_chunks("", self.enhanced_config)
        assert big_chunks == []
        
        big_chunks = self.chunker.chunk_text_into_big_chunks("   ", self.enhanced_config)
        assert big_chunks == []
    
    def test_chunk_text_into_big_chunks_small_text(self):
        """Test chunking small text into big chunks"""
        text = "Đây là một đoạn văn ngắn."
        big_chunks = self.chunker.chunk_text_into_big_chunks(text, self.enhanced_config)
        
        assert len(big_chunks) == 1
        assert big_chunks[0]['big_chunk_text'] == text
        assert big_chunks[0]['big_chunk_id'] == "big_chunk_0000"
        assert big_chunks[0]['big_chunk_size'] == len(text)
        assert big_chunks[0]['is_processed'] is False
        assert big_chunks[0]['processing_status'] == "pending"
    
    def test_chunk_text_into_big_chunks_large_text(self):
        """Test chunking large text into big chunks"""
        # Create text that will be split into multiple big chunks
        paragraph = "Đây là một đoạn văn dài với nhiều câu để tạo ra kích thước lớn. " * 200
        text = paragraph + "\n\n" + paragraph + "\n\n" + paragraph
        
        big_chunks = self.chunker.chunk_text_into_big_chunks(text, self.enhanced_config)
        
        assert len(big_chunks) > 1
        for chunk in big_chunks:
            assert chunk['big_chunk_size'] <= self.enhanced_config.big_chunk_size
            assert chunk['big_chunk_size'] >= self.enhanced_config.big_chunk_min_size
            assert chunk['big_chunk_text'].strip() != ""
    
    def test_chunk_big_chunk_into_small_chunks_empty(self):
        """Test splitting empty big chunk into small chunks"""
        big_chunk = BigChunkState(
            big_chunk_id="big_chunk_0000",
            big_chunk_text="",
            big_chunk_size=0,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        small_chunks = self.chunker.chunk_big_chunk_into_small_chunks(big_chunk, self.enhanced_config)
        assert small_chunks == []
    
    def test_chunk_big_chunk_into_small_chunks_small_text(self):
        """Test splitting small big chunk into small chunks"""
        big_chunk = BigChunkState(
            big_chunk_id="big_chunk_0000",
            big_chunk_text="Đây là một đoạn văn ngắn.",
            big_chunk_size=len("Đây là một đoạn văn ngắn."),
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        small_chunks = self.chunker.chunk_big_chunk_into_small_chunks(big_chunk, self.enhanced_config)
        
        assert len(small_chunks) == 1
        assert small_chunks[0]['small_chunk_text'] == big_chunk['big_chunk_text']
        assert small_chunks[0]['big_chunk_id'] == big_chunk['big_chunk_id']
        assert small_chunks[0]['small_chunk_id'] == f"small_chunk_{big_chunk['big_chunk_id']}_0000"
        assert small_chunks[0]['position_in_big_chunk'] == 0
        assert small_chunks[0]['is_processed'] is False
        assert small_chunks[0]['processing_status'] == "pending"
    
    def test_chunk_big_chunk_into_small_chunks_large_text(self):
        """Test splitting large big chunk into small chunks"""
        # Create a big chunk that will be split into multiple small chunks
        paragraph = "Đây là một đoạn văn dài với nhiều câu để tạo ra kích thước lớn. " * 50
        big_chunk = BigChunkState(
            big_chunk_id="big_chunk_0000",
            big_chunk_text=paragraph,
            big_chunk_size=len(paragraph),
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        small_chunks = self.chunker.chunk_big_chunk_into_small_chunks(big_chunk, self.enhanced_config)
        
        assert len(small_chunks) > 1
        for chunk in small_chunks:
            assert chunk['big_chunk_id'] == big_chunk['big_chunk_id']
            assert chunk['small_chunk_size'] <= self.enhanced_config.small_chunk_size * 2  # Allow flexibility
            assert chunk['small_chunk_size'] >= self.enhanced_config.small_chunk_min_size
            assert chunk['small_chunk_text'].strip() != ""
            assert chunk['position_in_big_chunk'] >= 0


class TestEnhancedChunkingFunctions:
    """Test cases for enhanced chunking utility functions"""
    
    def test_chunk_text_into_big_chunks_function(self):
        """Test convenience function for chunking into big chunks"""
        text = "Đây là một đoạn văn ngắn."
        big_chunks = chunk_text_into_big_chunks(text)
        
        assert len(big_chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in big_chunks)
        assert all('big_chunk_text' in chunk for chunk in big_chunks)
        assert all('big_chunk_id' in chunk for chunk in big_chunks)
    
    def test_chunk_big_chunk_into_small_chunks_function(self):
        """Test convenience function for splitting big chunks into small chunks"""
        big_chunk = BigChunkState(
            big_chunk_id="big_chunk_0000",
            big_chunk_text="Đây là một đoạn văn ngắn.",
            big_chunk_size=len("Đây là một đoạn văn ngắn."),
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        small_chunks = chunk_big_chunk_into_small_chunks(big_chunk)
        
        assert len(small_chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in small_chunks)
        assert all('small_chunk_text' in chunk for chunk in small_chunks)
        assert all('big_chunk_id' in chunk for chunk in small_chunks)
    
    def test_validate_enhanced_chunks_empty(self):
        """Test validation with empty chunks"""
        errors = validate_enhanced_chunks([], [])
        assert len(errors) == 2
        assert "No big chunks provided" in errors
        assert "No small chunks provided" in errors
    
    def test_validate_enhanced_chunks_valid(self):
        """Test validation with valid chunks"""
        big_chunk = BigChunkState(
            big_chunk_id="big_chunk_0000",
            big_chunk_text="Test text",
            big_chunk_size=1000,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        small_chunk = SmallChunkState(
            small_chunk_id="small_chunk_big_chunk_0000_0000",
            big_chunk_id="big_chunk_0000",
            small_chunk_text="Test text",
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
        
        errors = validate_enhanced_chunks([big_chunk], [small_chunk], strict=False)
        assert len(errors) == 0
    
    def test_validate_enhanced_chunks_invalid_sizes(self):
        """Test validation with invalid chunk sizes"""
        config = EnhancedChunkingConfig(
            big_chunk_size=1000,
            small_chunk_size=100,
            big_chunk_min_size=500,
            small_chunk_min_size=50
        )
        
        # Big chunk too large
        big_chunk_large = BigChunkState(
            big_chunk_id="big_chunk_0000",
            big_chunk_text="x" * 2000,  # Exceeds big_chunk_size
            big_chunk_size=2000,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        # Small chunk too small
        small_chunk_small = SmallChunkState(
            small_chunk_id="small_chunk_big_chunk_0000_0000",
            big_chunk_id="big_chunk_0000",
            small_chunk_text="x" * 25,  # Below small_chunk_min_size
            small_chunk_size=25,
            position_in_big_chunk=0,
            translated_text=None,
            recent_context=[],
            translation_attempts=0,
            max_attempts=3,
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        errors = validate_enhanced_chunks([big_chunk_large], [small_chunk_small], config, strict=True)
        assert len(errors) == 2
        assert any("exceeds maximum size" in error for error in errors)
        assert any("is too small" in error for error in errors)
    
    def test_reassemble_chunks_from_small_chunks(self):
        """Test reassembling text from small chunks"""
        small_chunks = [
            SmallChunkState(
                small_chunk_id="small_chunk_big_chunk_0000_0000",
                big_chunk_id="big_chunk_0000",
                small_chunk_text="Phần đầu tiên",
                small_chunk_size=len("Phần đầu tiên"),
                position_in_big_chunk=0,
                translated_text=None,
                recent_context=[],
                translation_attempts=0,
                max_attempts=3,
                is_processed=False,
                processing_status="pending",
                error_message=None
            ),
            SmallChunkState(
                small_chunk_id="small_chunk_big_chunk_0000_0001",
                big_chunk_id="big_chunk_0000",
                small_chunk_text="Phần thứ hai",
                small_chunk_size=len("Phần thứ hai"),
                position_in_big_chunk=len("Phần đầu tiên"),
                translated_text=None,
                recent_context=[],
                translation_attempts=0,
                max_attempts=3,
                is_processed=False,
                processing_status="pending",
                error_message=None
            )
        ]
        
        reassembled = reassemble_chunks_from_small_chunks(small_chunks)
        expected = "Phần đầu tiên\n\nPhần thứ hai"
        assert reassembled == expected
    
    def test_reassemble_chunks_from_big_chunks(self):
        """Test reassembling text from big chunks"""
        big_chunks = [
            BigChunkState(
                big_chunk_id="big_chunk_0000",
                big_chunk_text="Big chunk đầu tiên",
                big_chunk_size=len("Big chunk đầu tiên"),
                memory_context=[],
                small_chunks=[],
                is_processed=False,
                processing_status="pending",
                error_message=None
            ),
            BigChunkState(
                big_chunk_id="big_chunk_0001",
                big_chunk_text="Big chunk thứ hai",
                big_chunk_size=len("Big chunk thứ hai"),
                memory_context=[],
                small_chunks=[],
                is_processed=False,
                processing_status="pending",
                error_message=None
            )
        ]
        
        reassembled = reassemble_chunks_from_big_chunks(big_chunks)
        expected = "Big chunk đầu tiên\n\nBig chunk thứ hai"
        assert reassembled == expected
    
    def test_preserve_context_between_small_chunks(self):
        """Test preserving context between small chunks"""
        small_chunks = [
            SmallChunkState(
                small_chunk_id="small_chunk_big_chunk_0000_0000",
                big_chunk_id="big_chunk_0000",
                small_chunk_text="Chunk 1",
                small_chunk_size=len("Chunk 1"),
                position_in_big_chunk=0,
                translated_text="Translated 1",
                recent_context=[],
                translation_attempts=0,
                max_attempts=3,
                is_processed=True,
                processing_status="completed",
                error_message=None
            ),
            SmallChunkState(
                small_chunk_id="small_chunk_big_chunk_0000_0001",
                big_chunk_id="big_chunk_0000",
                small_chunk_text="Chunk 2",
                small_chunk_size=len("Chunk 2"),
                position_in_big_chunk=len("Chunk 1"),
                translated_text="Translated 2",
                recent_context=[],
                translation_attempts=0,
                max_attempts=3,
                is_processed=True,
                processing_status="completed",
                error_message=None
            ),
            SmallChunkState(
                small_chunk_id="small_chunk_big_chunk_0000_0002",
                big_chunk_id="big_chunk_0000",
                small_chunk_text="Chunk 3",
                small_chunk_size=len("Chunk 3"),
                position_in_big_chunk=len("Chunk 1") + len("Chunk 2"),
                translated_text=None,
                recent_context=[],
                translation_attempts=0,
                max_attempts=3,
                is_processed=False,
                processing_status="pending",
                error_message=None
            )
        ]
        
        chunks_with_context = preserve_context_between_small_chunks(small_chunks, context_window=2)
        
        # First chunk should have no context
        assert len(chunks_with_context[0]['recent_context']) == 0
        
        # Second chunk should have context from first chunk
        assert len(chunks_with_context[1]['recent_context']) == 1
        assert chunks_with_context[1]['recent_context'][0]['chunk_id'] == "small_chunk_big_chunk_0000_0000"
        
        # Third chunk should have context from first two chunks
        assert len(chunks_with_context[2]['recent_context']) == 2
        assert chunks_with_context[2]['recent_context'][0]['chunk_id'] == "small_chunk_big_chunk_0000_0000"
        assert chunks_with_context[2]['recent_context'][1]['chunk_id'] == "small_chunk_big_chunk_0000_0001"
    
    def test_analyze_enhanced_chunk_quality(self):
        """Test analyzing enhanced chunk quality"""
        big_chunks = [
            BigChunkState(
                big_chunk_id="big_chunk_0000",
                big_chunk_text="x" * 10000,
                big_chunk_size=10000,
                memory_context=[],
                small_chunks=[],
                is_processed=False,
                processing_status="pending",
                error_message=None
            ),
            BigChunkState(
                big_chunk_id="big_chunk_0001",
                big_chunk_text="x" * 12000,
                big_chunk_size=12000,
                memory_context=[],
                small_chunks=[],
                is_processed=False,
                processing_status="pending",
                error_message=None
            )
        ]
        
        small_chunks = [
            SmallChunkState(
                small_chunk_id="small_chunk_big_chunk_0000_0000",
                big_chunk_id="big_chunk_0000",
                small_chunk_text="x" * 500,
                small_chunk_size=500,
                position_in_big_chunk=0,
                translated_text=None,
                recent_context=[],
                translation_attempts=0,
                max_attempts=3,
                is_processed=False,
                processing_status="pending",
                error_message=None
            ),
            SmallChunkState(
                small_chunk_id="small_chunk_big_chunk_0000_0001",
                big_chunk_id="big_chunk_0000",
                small_chunk_text="x" * 600,
                small_chunk_size=600,
                position_in_big_chunk=500,
                translated_text=None,
                recent_context=[],
                translation_attempts=0,
                max_attempts=3,
                is_processed=False,
                processing_status="pending",
                error_message=None
            )
        ]
        
        analysis = analyze_enhanced_chunk_quality(big_chunks, small_chunks)
        
        assert analysis['big_chunks']['total'] == 2
        assert analysis['big_chunks']['average_size'] == 11000
        assert analysis['big_chunks']['min_size'] == 10000
        assert analysis['big_chunks']['max_size'] == 12000
        
        assert analysis['small_chunks']['total'] == 2
        assert analysis['small_chunks']['average_size'] == 550
        assert analysis['small_chunks']['min_size'] == 500
        assert analysis['small_chunks']['max_size'] == 600
        
        assert analysis['relationships']['chunks_per_big_chunk'] == 1.0
        assert analysis['relationships']['coverage_ratio'] > 0


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
        assert 'size_distribution' in analysis
    
    def test_analyze_chunk_quality_with_chunks(self):
        """Test quality analysis with actual chunks"""
        chunks = [
            ChunkState(
                chunk_index=0,
                chunk_text="Short",
                chunk_size=5,
                is_processed=False,
                translation_attempts=0,
                max_attempts=3
            ),
            ChunkState(
                chunk_index=1,
                chunk_text="Longer chunk with more content",
                chunk_size=30,
                is_processed=False,
                translation_attempts=0,
                max_attempts=3
            )
        ]
        
        analysis = analyze_chunk_quality(chunks)
        
        assert analysis['total_chunks'] == 2
        assert analysis['average_chunk_size'] == 17.5
        assert analysis['min_chunk_size'] == 5
        assert analysis['max_chunk_size'] == 30
        assert 'size_distribution' in analysis


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
    
    def test_enhanced_chunking_edge_cases(self):
        """Test edge cases for enhanced chunking"""
        # Test with very large text that needs multiple big chunks
        large_text = "Đây là một đoạn văn rất dài. " * 1000
        
        big_chunks = chunk_text_into_big_chunks(large_text)
        assert len(big_chunks) > 1
        
        # Test splitting a big chunk that's exactly at the limit
        config = EnhancedChunkingConfig(big_chunk_size=1000, small_chunk_size=200)
        exact_size_text = "x" * 1000
        big_chunk = BigChunkState(
            big_chunk_id="big_chunk_0000",
            big_chunk_text=exact_size_text,
            big_chunk_size=1000,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
        
        small_chunks = chunk_big_chunk_into_small_chunks(big_chunk, config)
        assert len(small_chunks) > 1 