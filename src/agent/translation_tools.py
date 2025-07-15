"""
Enhanced Translation tools for VietPhrase Reader Assistant.
Contains tools for LLM integration and translation operations with review and feedback capabilities.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import (
    memory_search_instructions,
    memory_update_instructions,
    context_summary_instructions,
    small_chunk_translation_instructions,
    feedback_incorporation_instructions,
    quality_assessment_instructions
)
from .configuration import Configuration
from .state import ReviewRating

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TranslationQuality:
    """Translation quality metrics and assessment."""
    accuracy_score: float  # 0.0-1.0
    fluency_score: float  # 0.0-1.0
    style_score: float  # 0.0-1.0
    consistency_score: float  # 0.0-1.0
    overall_score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    issues: List[str]
    suggestions: List[str]


@dataclass
class TranslationMetrics:
    """Translation performance metrics."""
    translation_time: float  # seconds
    character_count: int
    word_count: int
    memory_context_items: int
    search_queries_generated: int
    quality_score: float  # 0.0-1.0
    retry_count: int


class TranslationTools:
    """Enhanced tools for translation operations with LLM integration."""
    
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
        self.quality_assessment_llm = None
        
        # Translation quality tracking
        self.quality_history: List[TranslationQuality] = []
        self.metrics_history: List[TranslationMetrics] = []
        
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
            
            # Initialize quality assessment LLM
            self.quality_assessment_llm = ChatGoogleGenerativeAI(
                model=self.config.translation_model,  # Use same model for consistency
                temperature=0.2,  # Low temperature for consistent assessment
                max_tokens=4096
            )
            
            logger.info("Enhanced LLM clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {str(e)}")
            raise
    
    def translate_small_chunk(
        self, 
        original_text: str, 
        memory_context: List[Dict[str, Any]], 
        recent_context: List[Dict[str, Any]],
        position_in_big_chunk: int,
        total_small_chunks: int,
        feedback: Optional[str] = None
    ) -> Tuple[str, TranslationQuality, TranslationMetrics]:
        """
        Translate a small chunk (~500 words) with enhanced context and feedback incorporation.
        
        Args:
            original_text: Original Vietnamese text to translate
            memory_context: Retrieved memory nodes from Weaviate
            recent_context: Recent translation context
            position_in_big_chunk: Position of this chunk within the big chunk
            total_small_chunks: Total number of small chunks in the big chunk
            feedback: Optional feedback from previous translation attempts
            
        Returns:
            Tuple[str, TranslationQuality, TranslationMetrics]: Translated text, quality assessment, and metrics
        """
        start_time = datetime.now()
        
        try:
            # Format context for translation
            formatted_memory_context = self._format_memory_context(memory_context)
            formatted_recent_context = self._format_recent_context(recent_context)
            
            # Create enhanced translation prompt for small chunks
            prompt = small_chunk_translation_instructions.format(
                memory_context=formatted_memory_context,
                recent_context=formatted_recent_context,
                original_text=original_text,
                position=position_in_big_chunk,
                total_chunks=total_small_chunks,
                feedback=feedback or "No previous feedback available"
            )
            
            # Call LLM for translation
            if self.translation_llm:
                messages = [
                    SystemMessage(content="Bạn là chuyên gia dịch thuật tiếng Việt chuyên nghiệp, chuyên về dịch các đoạn văn bản nhỏ với độ chính xác cao."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.translation_llm.invoke(messages)
                translated_text = str(response.content).strip()
                
                # Assess translation quality
                quality = self._assess_translation_quality(original_text, translated_text, memory_context)
                
                # Calculate metrics
                end_time = datetime.now()
                translation_time = (end_time - start_time).total_seconds()
                metrics = TranslationMetrics(
                    translation_time=translation_time,
                    character_count=len(translated_text),
                    word_count=len(translated_text.split()),
                    memory_context_items=len(memory_context),
                    search_queries_generated=0,  # Will be updated by caller
                    quality_score=quality.overall_score,
                    retry_count=0  # Will be updated by caller
                )
                
                # Store quality and metrics
                self.quality_history.append(quality)
                self.metrics_history.append(metrics)
                
                logger.info(f"Small chunk translation completed: {len(translated_text)} characters, quality: {quality.overall_score:.2f}")
                return translated_text, quality, metrics
            else:
                # Fallback to placeholder translation
                logger.warning("LLM not available, using placeholder translation")
                fallback_text = f"[DỊCH] {original_text}"
                fallback_quality = TranslationQuality(
                    accuracy_score=0.0,
                    fluency_score=0.0,
                    style_score=0.0,
                    consistency_score=0.0,
                    overall_score=0.0,
                    confidence=0.0,
                    issues=["LLM not available"],
                    suggestions=["Check LLM configuration"]
                )
                fallback_metrics = TranslationMetrics(
                    translation_time=0.0,
                    character_count=len(fallback_text),
                    word_count=len(fallback_text.split()),
                    memory_context_items=len(memory_context),
                    search_queries_generated=0,
                    quality_score=0.0,
                    retry_count=0
                )
                return fallback_text, fallback_quality, fallback_metrics
                
        except Exception as e:
            logger.error(f"Small chunk translation failed: {str(e)}")
            raise
    
    def retranslate_with_feedback(
        self,
        original_text: str,
        previous_translation: str,
        feedback: str,
        memory_context: List[Dict[str, Any]],
        recent_context: List[Dict[str, Any]]
    ) -> Tuple[str, TranslationQuality, TranslationMetrics]:
        """
        Retranslate text incorporating feedback from review.
        
        Args:
            original_text: Original Vietnamese text
            previous_translation: Previous translation attempt
            feedback: Feedback from review agent
            memory_context: Retrieved memory nodes from Weaviate
            recent_context: Recent translation context
            
        Returns:
            Tuple[str, TranslationQuality, TranslationMetrics]: New translation, quality assessment, and metrics
        """
        start_time = datetime.now()
        
        try:
            # Format context for translation
            formatted_memory_context = self._format_memory_context(memory_context)
            formatted_recent_context = self._format_recent_context(recent_context)
            
            # Create feedback incorporation prompt
            prompt = feedback_incorporation_instructions.format(
                memory_context=formatted_memory_context,
                recent_context=formatted_recent_context,
                original_text=original_text,
                previous_translation=previous_translation,
                feedback=feedback
            )
            
            # Call LLM for retranslation
            if self.translation_llm:
                messages = [
                    SystemMessage(content="Bạn là chuyên gia dịch thuật tiếng Việt chuyên nghiệp, chuyên về cải thiện bản dịch dựa trên phản hồi."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.translation_llm.invoke(messages)
                translated_text = str(response.content).strip()
                
                # Assess translation quality
                quality = self._assess_translation_quality(original_text, translated_text, memory_context)
                
                # Calculate metrics
                end_time = datetime.now()
                translation_time = (end_time - start_time).total_seconds()
                metrics = TranslationMetrics(
                    translation_time=translation_time,
                    character_count=len(translated_text),
                    word_count=len(translated_text.split()),
                    memory_context_items=len(memory_context),
                    search_queries_generated=0,
                    quality_score=quality.overall_score,
                    retry_count=1  # This is a retry
                )
                
                # Store quality and metrics
                self.quality_history.append(quality)
                self.metrics_history.append(metrics)
                
                logger.info(f"Retranslation with feedback completed: quality improved to {quality.overall_score:.2f}")
                return translated_text, quality, metrics
            else:
                logger.warning("LLM not available for retranslation")
                raise RuntimeError("LLM not available for retranslation")
                
        except Exception as e:
            logger.error(f"Retranslation with feedback failed: {str(e)}")
            raise
    
    def _assess_translation_quality(
        self,
        original_text: str,
        translated_text: str,
        memory_context: List[Dict[str, Any]]
    ) -> TranslationQuality:
        """
        Assess the quality of a translation using LLM.
        """
        assert isinstance(original_text, str) and original_text.strip(), "Original text must be a non-empty string"
        assert isinstance(translated_text, str) and translated_text.strip(), "Translated text must be a non-empty string"
        try:
            # Create quality assessment prompt
            prompt = quality_assessment_instructions.format(
                original_text=original_text,
                translated_text=translated_text,
                memory_context=self._format_memory_context(memory_context)
            )
            # Call LLM for quality assessment
            if self.quality_assessment_llm:
                messages = [
                    SystemMessage(content="Bạn là chuyên gia đánh giá chất lượng dịch thuật tiếng Việt."),
                    HumanMessage(content=prompt)
                ]
                response = self.quality_assessment_llm.invoke(messages)
                # Parse JSON response
                try:
                    content = str(response.content).strip()
                    logger.debug(f"Quality assessment raw response: {content[:200]}...")
                    # Remove markdown code blocks if present
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.startswith('```'):
                        content = content[3:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    if not content:
                        logger.error("Empty response from quality assessment LLM")
                        raise ValueError("Empty response from quality assessment LLM")
                    try:
                        quality_data = json.loads(content)
                    except json.JSONDecodeError as e:
                        # Try to recover from truncated JSON by truncating at the last '}':
                        last_brace = content.rfind('}')
                        if last_brace != -1:
                            try:
                                quality_data = json.loads(content[:last_brace+1])
                                logger.warning("Recovered from truncated JSON in quality assessment response.")
                            except Exception as e2:
                                logger.error(f"Failed to recover from truncated JSON: {e2}\nRaw content: {content}")
                                raise ValueError(f"Failed to parse quality assessment JSON: {e}\nRaw content: {content}")
                        else:
                            logger.error(f"Failed to parse quality assessment JSON: {e}\nRaw content: {content}")
                            raise ValueError(f"Failed to parse quality assessment JSON: {e}\nRaw content: {content}")
                    return TranslationQuality(
                        accuracy_score=float(quality_data.get("accuracy_score", 0.0)),
                        fluency_score=float(quality_data.get("fluency_score", 0.0)),
                        style_score=float(quality_data.get("style_score", 0.0)),
                        consistency_score=float(quality_data.get("consistency_score", 0.0)),
                        overall_score=float(quality_data.get("overall_score", 0.0)),
                        confidence=float(quality_data.get("confidence", 0.0)),
                        issues=quality_data.get("issues", []),
                        suggestions=quality_data.get("suggestions", [])
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse quality assessment JSON: {e}\nRaw content: {content}")
                    raise ValueError(f"Failed to parse quality assessment JSON: {e}\nRaw content: {content}")
            else:
                return self._fallback_quality_assessment(original_text, translated_text)
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            raise
    
    def _fallback_quality_assessment(self, original_text: str, translated_text: str) -> TranslationQuality:
        """Fallback quality assessment using simple heuristics."""
        # Simple heuristic: if translation is longer than original, it might be more detailed
        length_ratio = len(translated_text) / max(len(original_text), 1)
        
        # Basic scoring based on length and content
        if length_ratio > 0.5 and length_ratio < 2.0:
            base_score = 0.7
        else:
            base_score = 0.5
        
        return TranslationQuality(
            accuracy_score=base_score,
            fluency_score=base_score,
            style_score=base_score,
            consistency_score=base_score,
            overall_score=base_score,
            confidence=0.5,
            issues=["Fallback assessment used"],
            suggestions=["Enable LLM for better quality assessment"]
        )
    
    def update_recent_context(
        self,
        current_translation: str,
        memory_context: List[Dict[str, Any]],
        position_in_big_chunk: int
    ) -> Dict[str, Any]:
        """
        Update recent context with current translation for continuity.
        
        Args:
            current_translation: Current chunk translation
            memory_context: Memory context used
            position_in_big_chunk: Position within big chunk
            
        Returns:
            Dict[str, Any]: Updated context item
        """
        try:
            # Generate context summary
            summary = self.generate_context_summary(
                f"Position {position_in_big_chunk} in big chunk",
                current_translation
            )
            
            # Create context item
            context_item = {
                "timestamp": datetime.now().isoformat(),
                "position": position_in_big_chunk,
                "translation": current_translation,
                "summary": summary,
                "memory_context_count": len(memory_context),
                "character_count": len(current_translation)
            }
            
            return context_item
            
        except Exception as e:
            logger.error(f"Recent context update failed: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "position": position_in_big_chunk,
                "translation": current_translation,
                "summary": f"Translation at position {position_in_big_chunk}",
                "memory_context_count": len(memory_context),
                "character_count": len(current_translation)
            }
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about translation quality over time.
        
        Returns:
            Dict[str, Any]: Quality statistics
        """
        if not self.quality_history:
            return {
                "total_translations": 0,
                "average_quality": 0.0,
                "quality_trend": "no_data",
                "common_issues": [],
                "improvement_suggestions": []
            }
        
        total_translations = len(self.quality_history)
        average_quality = sum(q.overall_score for q in self.quality_history) / total_translations
        
        # Calculate quality trend
        if total_translations >= 2:
            recent_avg = sum(q.overall_score for q in self.quality_history[-5:]) / min(5, total_translations)
            early_avg = sum(q.overall_score for q in self.quality_history[:5]) / min(5, total_translations)
            
            if recent_avg > early_avg + 0.1:
                quality_trend = "improving"
            elif recent_avg < early_avg - 0.1:
                quality_trend = "declining"
            else:
                quality_trend = "stable"
        else:
            quality_trend = "insufficient_data"
        
        # Collect common issues and suggestions
        from collections import Counter
        all_issues = []
        all_suggestions = []
        for quality in self.quality_history:
            all_issues.extend(quality.issues)
            all_suggestions.extend(quality.suggestions)
        
        # Get most common issues and suggestions
        common_issues = [issue for issue, count in Counter(all_issues).most_common(5)]
        improvement_suggestions = [suggestion for suggestion, count in Counter(all_suggestions).most_common(5)]
        
        return {
            "total_translations": total_translations,
            "average_quality": average_quality,
            "quality_trend": quality_trend,
            "common_issues": common_issues,
            "improvement_suggestions": improvement_suggestions,
            "recent_quality_scores": [q.overall_score for q in self.quality_history[-10:]]
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for translation operations.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if not self.metrics_history:
            return {
                "total_translations": 0,
                "average_translation_time": 0.0,
                "total_characters_translated": 0,
                "average_quality_score": 0.0,
                "total_retries": 0
            }
        
        total_translations = len(self.metrics_history)
        average_translation_time = sum(m.translation_time for m in self.metrics_history) / total_translations
        total_characters_translated = sum(m.character_count for m in self.metrics_history)
        average_quality_score = sum(m.quality_score for m in self.metrics_history) / total_translations
        total_retries = sum(m.retry_count for m in self.metrics_history)
        
        return {
            "total_translations": total_translations,
            "average_translation_time": average_translation_time,
            "total_characters_translated": total_characters_translated,
            "average_quality_score": average_quality_score,
            "total_retries": total_retries,
            "characters_per_second": total_characters_translated / max(sum(m.translation_time for m in self.metrics_history), 1)
        }
    
    def enhance_memory_context_for_big_chunk(
        self,
        big_chunk_text: str,
        existing_memory_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance memory context retrieval for big chunks with broader search.
        
        Args:
            big_chunk_text: Text of the big chunk
            existing_memory_context: Already retrieved memory context
            
        Returns:
            List[Dict[str, Any]]: Enhanced memory context
        """
        try:
            # Generate additional search queries for the big chunk
            additional_queries = self.generate_search_queries(big_chunk_text)
            
            # For big chunks, we want more comprehensive context
            enhanced_context = existing_memory_context.copy()
            
            # Add broader context searches
            for query in additional_queries[:5]:  # Limit to 5 additional queries
                # This would typically call Weaviate client
                # For now, we'll simulate enhanced context
                enhanced_context.append({
                    "type": "enhanced_search",
                    "label": f"Enhanced search: {query}",
                    "name": query,
                    "content": f"Additional context for: {query}",
                    "alias": [query.lower(), query.replace(" ", "_")]
                })
            
            # Remove duplicates based on name
            unique_context = {}
            for item in enhanced_context:
                name = item.get("name", "")
                if name not in unique_context:
                    unique_context[name] = item
            
            return list(unique_context.values())
            
        except Exception as e:
            logger.error(f"Memory context enhancement failed: {str(e)}")
            return existing_memory_context
    
    # def translate_chunk(
    #     self, 
    #     original_text: str, 
    #     memory_context: List[Dict[str, Any]], 
    #     recent_context: List[Dict[str, Any]]
    # ) -> str:
    #     """
    #     Translate a chunk of Vietnamese text using LLM with memory context.
        
    #     Args:
    #         original_text: Original Vietnamese text to translate
    #         memory_context: Retrieved memory nodes from Weaviate
    #         recent_context: Recent translation context
            
    #     Returns:
    #         str: Translated text
    #     """
    #     try:
    #         # Format context for translation
    #         formatted_memory_context = self._format_memory_context(memory_context)
    #         formatted_recent_context = self._format_recent_context(recent_context)
            
    #         # Create translation prompt
    #         prompt = translation_instructions.format(
    #             memory_context=formatted_memory_context,
    #             recent_context=formatted_recent_context,
    #             original_text=original_text
    #         )
            
    #         # Call LLM for translation
    #         if self.translation_llm:
    #             messages = [
    #                 SystemMessage(content="Bạn là chuyên gia dịch thuật tiếng Việt chuyên nghiệp."),
    #                 HumanMessage(content=prompt)
    #             ]
                
    #             response = self.translation_llm.invoke(messages)
    #             translated_text = str(response.content).strip()
                
    #             logger.info(f"Translation completed: {len(translated_text)} characters")
    #             return translated_text
    #         else:
    #             # Fallback to placeholder translation
    #             logger.warning("LLM not available, using placeholder translation")
    #             return f"[DỊCH] {original_text}"
                
    #     except Exception as e:
    #         logger.error(f"Translation failed: {str(e)}")
    #         raise
    
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