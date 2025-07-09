"""
Unit tests for Vietnamese prompts.
Tests prompt formatting and content validation.
"""

import pytest
from agent.prompts import (
    translation_instructions,
    memory_search_instructions,
    memory_update_instructions,
    context_summary_instructions,
    get_current_date
)


class TestPrompts:
    """Test cases for Vietnamese prompts."""
    
    def test_get_current_date(self):
        """Test get_current_date function."""
        date = get_current_date()
        assert isinstance(date, str)
        assert len(date) > 0
        # Should contain month and day
        assert any(month in date for month in [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
    
    def test_translation_instructions_formatting(self):
        """Test translation instructions prompt formatting."""
        memory_context = "character: Nhân vật chính (Nguyễn Văn A) - Nhân vật chính trong truyện"
        recent_context = "Ngữ cảnh gần đây: Nhân vật bắt đầu hành trình"
        original_text = "这是原文"
        
        formatted_prompt = translation_instructions.format(
            memory_context=memory_context,
            recent_context=recent_context,
            original_text=original_text
        )
        
        # Verify all placeholders are replaced
        assert "{memory_context}" not in formatted_prompt
        assert "{recent_context}" not in formatted_prompt
        assert "{original_text}" not in formatted_prompt
        
        # Verify content is included
        assert memory_context in formatted_prompt
        assert recent_context in formatted_prompt
        assert original_text in formatted_prompt
        
        # Verify Vietnamese instructions are present
        assert "chuyên gia dịch thuật tiếng Việt" in formatted_prompt
        assert "VietPhrase" in formatted_prompt
        assert "tiếng Việt trôi chảy" in formatted_prompt
    
    def test_memory_search_instructions_formatting(self):
        """Test memory search instructions prompt formatting."""
        chunk_text = "Nhân vật chính bắt đầu hành trình với công nghệ AI."
        
        formatted_prompt = memory_search_instructions.format(chunk_text=chunk_text)
        
        # Verify placeholder is replaced
        assert "{chunk_text}" not in formatted_prompt
        assert chunk_text in formatted_prompt
        
        # Verify Vietnamese instructions are present
        assert "trợ lý AI phân tích văn bản tiếng Việt" in formatted_prompt
        assert "Tên nhân vật" in formatted_prompt
        assert "thuật ngữ kỹ thuật" in formatted_prompt
        assert "mảng JSON" in formatted_prompt
    
    def test_memory_update_instructions_formatting(self):
        """Test memory update instructions prompt formatting."""
        original_text = "原文"
        translated_text = "Văn bản đã dịch"
        memory_context = "character: Nhân vật chính (Nguyễn Văn A) - Nhân vật chính trong truyện"
        
        formatted_prompt = memory_update_instructions.format(
            original_text=original_text,
            translated_text=translated_text,
            memory_context=memory_context
        )
        
        # Verify all placeholders are replaced
        assert "{original_text}" not in formatted_prompt
        assert "{translated_text}" not in formatted_prompt
        assert "{memory_context}" not in formatted_prompt
        
        # Verify content is included
        assert original_text in formatted_prompt
        assert translated_text in formatted_prompt
        assert memory_context in formatted_prompt
        
        # Verify Vietnamese instructions are present
        assert "trợ lý AI quyết định" in formatted_prompt
        assert "nút kiến thức mới" in formatted_prompt
        assert "character, term, item, location, event" in formatted_prompt
        assert "create_nodes" in formatted_prompt
        assert "update_nodes" in formatted_prompt
    
    def test_context_summary_instructions_formatting(self):
        """Test context summary instructions prompt formatting."""
        recent_context = "Ngữ cảnh gần đây: Nhân vật bắt đầu hành trình"
        current_translation = "Đây là bản dịch hiện tại của đoạn văn."
        
        formatted_prompt = context_summary_instructions.format(
            recent_context=recent_context,
            current_translation=current_translation
        )
        
        # Verify all placeholders are replaced
        assert "{recent_context}" not in formatted_prompt
        assert "{current_translation}" not in formatted_prompt
        
        # Verify content is included
        assert recent_context in formatted_prompt
        assert current_translation in formatted_prompt
        
        # Verify Vietnamese instructions are present
        assert "trợ lý AI tóm tắt ngữ cảnh" in formatted_prompt
        assert "thuật ngữ chính" in formatted_prompt
        assert "tên nhân vật" in formatted_prompt
        assert "ngữ cảnh tương lai" in formatted_prompt
    
    def test_translation_instructions_content(self):
        """Test translation instructions content validation."""
        # Check for required Vietnamese translation guidelines
        required_phrases = [
            "chuyên gia dịch thuật tiếng Việt",
            "VietPhrase",
            "tiếng Việt trôi chảy",
            "Duy trì ý nghĩa",
            "ngữ pháp tiếng Việt",
            "thuật ngữ nhất quán",
            "tiếng Việt trang trọng",
            "tiếng Việt thân mật"
        ]
        
        for phrase in required_phrases:
            assert phrase in translation_instructions, f"Missing phrase: {phrase}"
    
    def test_memory_search_instructions_content(self):
        """Test memory search instructions content validation."""
        # Check for required search guidelines
        required_phrases = [
            "trợ lý AI phân tích văn bản",
            "Tên nhân vật",
            "thuật ngữ kỹ thuật",
            "Cụm từ hoặc cách diễn đạt lặp lại",
            "Sự kiện hoặc khái niệm quan trọng",
            "Tham chiếu văn hóa hoặc thành ngữ",
            "danh từ riêng",
            "mảng JSON"
        ]
        
        for phrase in required_phrases:
            assert phrase in memory_search_instructions, f"Missing phrase: {phrase}"
    
    def test_memory_update_instructions_content(self):
        """Test memory update instructions content validation."""
        # Check for required memory operation guidelines
        required_phrases = [
            "trợ lý AI quyết định",
            "nút kiến thức mới",
            "Cập nhật nút hiện có",
            "thông tin quan trọng",
            "character, term, item, location, event",
            "create_nodes",
            "update_nodes"
        ]
        
        for phrase in required_phrases:
            assert phrase in memory_update_instructions, f"Missing phrase: {phrase}"
    
    def test_context_summary_instructions_content(self):
        """Test context summary instructions content validation."""
        # Check for required summary guidelines
        required_phrases = [
            "trợ lý AI tóm tắt",
            "thuật ngữ chính",
            "tên nhân vật",
            "lựa chọn phong cách",
            "ngắn gọn",
            "ngữ cảnh tương lai"
        ]
        
        for phrase in required_phrases:
            assert phrase in context_summary_instructions, f"Missing phrase: {phrase}"
    
    def test_prompt_length_reasonable(self):
        """Test that prompts are not excessively long."""
        max_length = 5000  # Reasonable max length for prompts
        
        assert len(translation_instructions) < max_length
        assert len(memory_search_instructions) < max_length
        assert len(memory_update_instructions) < max_length
        assert len(context_summary_instructions) < max_length
    
    def test_prompt_placeholders_consistent(self):
        """Test that all prompts have consistent placeholder usage."""
        # Check that all prompts use the same placeholder format
        all_prompts = [
            translation_instructions,
            memory_search_instructions,
            memory_update_instructions,
            context_summary_instructions
        ]
        
        for prompt in all_prompts:
            # Should use {variable_name} format
            assert "{" in prompt
            assert "}" in prompt
            # Should not have any unmatched braces
            assert prompt.count("{") == prompt.count("}")
    
    def test_vietnamese_language_usage(self):
        """Test that prompts use proper Vietnamese language."""
        # Check for common Vietnamese words and phrases
        vietnamese_indicators = [
            "là", "của", "với", "trong", "cho", "để", "nếu", "khi",
            "tiếng Việt", "văn bản", "ngữ cảnh", "thuật ngữ"
        ]
        
        all_prompts = [
            translation_instructions,
            memory_search_instructions,
            memory_update_instructions,
            context_summary_instructions
        ]
        
        for prompt in all_prompts:
            # At least some Vietnamese words should be present
            vietnamese_count = sum(1 for word in vietnamese_indicators if word in prompt)
            assert vietnamese_count > 0, f"Prompt lacks Vietnamese language indicators: {prompt[:100]}..." 