"""
Translation tools for VietPhrase Reader Assistant.
Contains tools for LLM integration and translation operations.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import (
    translation_instructions,
    memory_search_instructions,
    memory_update_instructions,
    context_summary_instructions
)
from .configuration import Configuration

# Configure logging
logger = logging.getLogger(__name__)


class TranslationTools:
    """Tools for translation operations with LLM integration."""
    
    def __init__(self, config: Configuration):
        """
        Initialize translation tools with configuration.
        
        Args:
            config: Configuration object with model settings
        """
        self.config = config
        self.translation_llm = None
        self.memory_search_llm = None
        self.memory_update_llm = None
        self.context_summary_llm = None
        
        # Initialize LLM clients
        self._initialize_llms()
    
    def _initialize_llms(self):
        """Initialize LLM clients for different operations."""
        try:
            # Initialize translation LLM
            self.translation_llm = ChatGoogleGenerativeAI(
                model=self.config.translation_model,
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=8192  # Increased for longer Vietnamese translations
            )
            
            # Initialize memory search LLM
            self.memory_search_llm = ChatGoogleGenerativeAI(
                model=self.config.memory_search_model,
                temperature=0.1,  # Very low temperature for consistent search queries
                max_tokens=8192
            )
            
            # Initialize memory update LLM
            self.memory_update_llm = ChatGoogleGenerativeAI(
                model=self.config.memory_update_model,
                temperature=0.2,  # Low temperature for consistent decisions
                max_tokens=8192
            )
            
            # Initialize context summary LLM
            self.context_summary_llm = ChatGoogleGenerativeAI(
                model=self.config.context_summary_model,
                temperature=0.3,  # Moderate temperature for natural summaries
                max_tokens=8192
            )
            
            logger.info("LLM clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {str(e)}")
            raise
    
    def translate_chunk(
        self, 
        original_text: str, 
        memory_context: List[Dict[str, Any]], 
        recent_context: List[Dict[str, Any]]
    ) -> str:
        """
        Translate a chunk of Vietnamese text using LLM with memory context.
        
        Args:
            original_text: Original Vietnamese text to translate
            memory_context: Retrieved memory nodes from Weaviate
            recent_context: Recent translation context
            
        Returns:
            str: Translated text
        """
        try:
            # Format context for translation
            formatted_memory_context = self._format_memory_context(memory_context)
            formatted_recent_context = self._format_recent_context(recent_context)
            
            # Create translation prompt
            prompt = translation_instructions.format(
                memory_context=formatted_memory_context,
                recent_context=formatted_recent_context,
                original_text=original_text
            )
            
            # Call LLM for translation
            if self.translation_llm:
                messages = [
                    SystemMessage(content="Bạn là chuyên gia dịch thuật tiếng Việt chuyên nghiệp. Chỉ trả về bản dịch tiếng Việt, không thêm bất kỳ lời bình luận hay giải thích nào."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.translation_llm.invoke(messages)
                translated_text = str(response.content).strip()
                
                logger.info(f"Translation completed: {len(translated_text)} characters")
                return translated_text
            else:
                # Fallback to placeholder translation
                logger.warning("LLM not available, using placeholder translation")
                return f"[DỊCH] {original_text}"
                
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise
    
    def generate_search_queries(self, chunk_text: str) -> List[str]:
        """
        Generate search queries from chunk text using LLM.
        
        Args:
            chunk_text: Text to analyze for search queries
            
        Returns:
            List[str]: List of search queries
        """
        try:
            # Create search prompt
            prompt = memory_search_instructions.format(chunk_text=chunk_text)
            
            # Call LLM for search queries
            if self.memory_search_llm:
                messages = [
                    SystemMessage(content="Bạn là trợ lý AI chuyên phân tích văn bản để tạo truy vấn tìm kiếm."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.memory_search_llm.invoke(messages)
                
                # Parse JSON response - handle markdown code blocks
                try:
                    content = str(response.content).strip()
                    
                    # Remove markdown code blocks if present
                    if content.startswith('```json'):
                        content = content[7:]  # Remove ```json
                    if content.startswith('```'):
                        content = content[3:]  # Remove ```
                    if content.endswith('```'):
                        content = content[:-3]  # Remove ```
                    
                    content = content.strip()
                    queries = json.loads(content)
                    
                    if isinstance(queries, list):
                        return queries[:8]  # Limit to 3 queries
                    else:
                        logger.warning("Invalid JSON response format in search queries")
                        return self._fallback_search_queries(chunk_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response in search queries: {e}")
                    logger.warning(f"Raw response: {str(response.content)}")
                    return self._fallback_search_queries(chunk_text)
            else:
                return self._fallback_search_queries(chunk_text)
                
        except Exception as e:
            logger.error(f"Search query generation failed: {str(e)}")
            return self._fallback_search_queries(chunk_text)
    
    def generate_memory_operations(
        self, 
        original_text: str, 
        translated_text: str, 
        memory_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate memory operations based on translation results.
        
        Args:
            original_text: Original Vietnamese text
            translated_text: Translated text
            memory_context: Retrieved memory context
            
        Returns:
            Dict[str, Any]: Memory operations to perform
        """
        try:
            # Create memory update prompt
            prompt = memory_update_instructions.format(
                original_text=original_text,
                translated_text=translated_text,
                memory_context=self._format_memory_context(memory_context)
            )
            
            # Call LLM for memory operations
            if self.memory_update_llm:
                messages = [
                    SystemMessage(content="Bạn là trợ lý AI chuyên quản lý cơ sở kiến thức."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.memory_update_llm.invoke(messages)
                
                # Parse JSON response - handle markdown code blocks
                try:
                    content = str(response.content).strip()
                    
                    # Remove markdown code blocks if present
                    if content.startswith('```json'):
                        content = content[7:]  # Remove ```json
                    if content.startswith('```'):
                        content = content[3:]  # Remove ```
                    if content.endswith('```'):
                        content = content[:-3]  # Remove ```
                    
                    content = content.strip()
                    operations = json.loads(content)
                    return operations
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse memory operations JSON: {e}")
                    logger.warning(f"Raw response: {str(response.content)}")
                    return {"create_nodes": [], "update_nodes": []}
            else:
                return {"create_nodes": [], "update_nodes": []}
                
        except Exception as e:
            logger.error(f"Memory operations generation failed: {str(e)}")
            return {"create_nodes": [], "update_nodes": []}
    
    def generate_context_summary(self, recent_context: str, current_translation: str) -> str:
        """
        Generate context summary for future chunks.
        
        Args:
            recent_context: Recent translation context
            current_translation: Current chunk translation
            
        Returns:
            str: Context summary
        """
        try:
            # Create context summary prompt
            prompt = context_summary_instructions.format(
                recent_context=recent_context,
                current_translation=current_translation
            )
            
            # Call LLM for context summary
            if self.context_summary_llm:
                messages = [
                    SystemMessage(content="Bạn là trợ lý AI chuyên tóm tắt ngữ cảnh."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.context_summary_llm.invoke(messages)
                summary = str(response.content).strip()
                
                return summary
            else:
                # Fallback summary
                return f"Ngữ cảnh gần đây: {len(current_translation)} ký tự đã dịch"
                
        except Exception as e:
            logger.error(f"Context summary generation failed: {str(e)}")
            return f"Ngữ cảnh gần đây: {len(current_translation)} ký tự đã dịch"
    
    def _format_memory_context(self, memory_context: List[Dict[str, Any]]) -> str:
        """
        Format memory context for prompts.
        
        Args:
            memory_context: List of retrieved memory nodes
            
        Returns:
            str: Formatted memory context
        """
        if not memory_context:
            return "Không tìm thấy ngữ cảnh bộ nhớ liên quan."
        
        formatted_context = []
        for node in memory_context:
            node_type = node.get('type', 'Unknown')
            label = node.get('label', 'Unknown')
            name = node.get('name', '')
            content = node.get('content', '')
            formatted_context.append(f"{node_type}: {label} ({name}) - {content}")
        
        return "\n".join(formatted_context)
    
    def _format_recent_context(self, recent_context: List[Dict[str, Any]]) -> str:
        """
        Format recent context for prompts.
        
        Args:
            recent_context: List of recent context items
            
        Returns:
            str: Formatted recent context
        """
        if not recent_context:
            return "Không có ngữ cảnh dịch thuật gần đây."
        
        formatted_context = []
        for item in recent_context[-3:]:  # Last 3 items
            summary = item.get('summary', '')
            formatted_context.append(summary)
        
        return "\n".join(formatted_context)
    
    def _fallback_search_queries(self, chunk_text: str) -> List[str]:
        """
        Generate fallback search queries using simple heuristics.
        
        Args:
            chunk_text: Text to analyze
            
        Returns:
            List[str]: Basic search queries
        """
        queries = []
        
        # Simple heuristic: look for proper nouns and technical terms
        words = chunk_text.split()
        for word in words:
            if len(word) > 3 and word[0].isupper():
                queries.append(word)
        
        return queries[:3]  # Limit to 3 queries 