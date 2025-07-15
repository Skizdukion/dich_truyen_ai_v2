import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from src.agent.state import ChunkState, BigChunkState, SmallChunkState


@dataclass
class ChunkingConfig:
    """Configuration for Vietnamese text chunking"""
    max_chunk_size: int = 5000  # Maximum characters per chunk
    min_chunk_size: int = 800  # Minimum characters per chunk
    overlap_size: int = 0       # Overlap between chunks for context continuity
    preserve_sentences: bool = True  # Try to keep sentences intact
    preserve_paragraphs: bool = True  # Try to keep paragraphs intact


@dataclass
class EnhancedChunkingConfig:
    """Configuration for enhanced translation flow chunking"""
    big_chunk_size: int = 16000  # Maximum characters per big chunk (16k limit)
    small_chunk_size: int = 500   # Target characters per small chunk (~500 words)
    big_chunk_min_size: int = 8000  # Minimum characters per big chunk
    small_chunk_min_size: int = 200  # Minimum characters per small chunk
    overlap_size: int = 0         # Overlap between chunks for context continuity
    preserve_sentences: bool = True  # Try to keep sentences intact
    preserve_paragraphs: bool = True  # Try to keep paragraphs intact


class VietnameseTextChunker:
    """Vietnamese text chunking system with intelligent boundary detection"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        
        # Vietnamese sentence endings
        self.sentence_endings = r'[.!?。！？]+'
        
        # Vietnamese paragraph breaks
        self.paragraph_breaks = r'\n\s*\n'
        
        # Vietnamese punctuation that might indicate phrase boundaries
        self.phrase_boundaries = r'[，,；;：:]'
        
        # Vietnamese word boundaries (basic)
        self.word_boundaries = r'\s+'
    
    def chunk_text(self, text: str) -> List[ChunkState]:
        """
        Chunk Vietnamese text into manageable pieces while preserving semantic boundaries.
        
        Args:
            text: Vietnamese text to chunk
            
        Returns:
            List of ChunkState objects representing the chunks
        """
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        text = self._normalize_text(text)
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) == 0:
                continue
                
            # If paragraph is small enough, keep it as one chunk
            if len(paragraph) <= self.config.max_chunk_size:
                chunks.append(self._create_chunk_state(
                    chunk_index, paragraph, len(paragraph)
                ))
                chunk_index += 1
            else:
                # Split paragraph into smaller chunks
                paragraph_chunks = self._chunk_paragraph(paragraph, chunk_index)
                chunks.extend(paragraph_chunks)
                chunk_index += len(paragraph_chunks)
        
        # If no chunks were created (e.g., empty paragraphs), create one chunk with the original text
        if not chunks and text.strip():
            chunks.append(self._create_chunk_state(0, text, len(text)))
        
        return chunks
    
    def chunk_text_into_big_chunks(self, text: str, config: Optional[EnhancedChunkingConfig] = None) -> List[BigChunkState]:
        """
        Chunk Vietnamese text into big chunks (16k limit) for enhanced translation flow.
        
        Args:
            text: Vietnamese text to chunk
            config: Enhanced chunking configuration
            
        Returns:
            List of BigChunkState objects representing the big chunks
        """
        if not text or not text.strip():
            return []
        
        config = config or EnhancedChunkingConfig()
        
        # Clean and normalize text
        text = self._normalize_text(text)
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        big_chunks = []
        big_chunk_index = 0
        current_big_chunk_text = ""
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) == 0:
                continue
            
            # Check if adding this paragraph would exceed the big chunk size
            if len(current_big_chunk_text) + len(paragraph) + 2 <= config.big_chunk_size:
                # Add paragraph to current big chunk
                if current_big_chunk_text:
                    current_big_chunk_text += "\n\n" + paragraph
                else:
                    current_big_chunk_text = paragraph
            else:
                # Current big chunk is full, create it and start a new one
                if current_big_chunk_text:
                    big_chunks.append(self._create_big_chunk_state(
                        big_chunk_index, current_big_chunk_text, len(current_big_chunk_text)
                    ))
                    big_chunk_index += 1
                
                # If the paragraph itself is larger than the big chunk size, split it
                if len(paragraph) > config.big_chunk_size:
                    # Split the paragraph into smaller pieces
                    remaining_paragraph = paragraph
                    while len(remaining_paragraph) > 0:
                        chunk_size = min(config.big_chunk_size, len(remaining_paragraph))
                        chunk_text, remaining_paragraph = self._find_optimal_break(
                            remaining_paragraph, chunk_size
                        )
                        
                        big_chunks.append(self._create_big_chunk_state(
                            big_chunk_index, chunk_text, len(chunk_text)
                        ))
                        big_chunk_index += 1
                else:
                    # Start new big chunk with current paragraph
                    current_big_chunk_text = paragraph
        
        # Add the last big chunk if there's remaining text
        if current_big_chunk_text:
            big_chunks.append(self._create_big_chunk_state(
                big_chunk_index, current_big_chunk_text, len(current_big_chunk_text)
            ))
        
        # If no big chunks were created, create one with the original text
        if not big_chunks and text.strip():
            big_chunks.append(self._create_big_chunk_state(0, text, len(text)))
        
        return big_chunks
    
    def chunk_big_chunk_into_small_chunks(self, big_chunk: BigChunkState, config: Optional[EnhancedChunkingConfig] = None) -> List[SmallChunkState]:
        """
        Split a big chunk into small chunks (~500 words) for enhanced translation flow.
        """
        import logging
        logger = logging.getLogger(__name__)
        assert isinstance(big_chunk['big_chunk_text'], str) and big_chunk['big_chunk_text'].strip(), "Input big chunk text must be a non-empty string"
        config = config or EnhancedChunkingConfig()
        text = big_chunk['big_chunk_text']
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        small_chunks = []
        small_chunk_index = 0
        current_small_chunk_text = ""
        position_in_big_chunk = 0
        min_size = 400
        max_size = 700
        for paragraph in paragraphs:
            assert isinstance(paragraph, str), f"Paragraph must be a string, got {type(paragraph)}"
            if len(paragraph.strip()) == 0:
                continue
            # If paragraph itself is larger than max_size, split it
            if len(paragraph) > max_size:
                remaining_paragraph = paragraph
                while len(remaining_paragraph) > 0:
                    chunk_size = min(max_size, len(remaining_paragraph))
                    chunk_text, remaining_paragraph = self._find_optimal_break(
                        remaining_paragraph, chunk_size
                    )
                    if current_small_chunk_text:
                        # Add current chunk before starting a new one
                        if len(current_small_chunk_text) >= min_size:
                            small_chunks.append(self._create_small_chunk_state(
                                small_chunk_index, big_chunk['big_chunk_id'], current_small_chunk_text,
                                len(current_small_chunk_text), position_in_big_chunk
                            ))
                            small_chunk_index += 1
                            position_in_big_chunk += len(current_small_chunk_text)
                            current_small_chunk_text = ""
                        else:
                            # Merge with the new chunk_text if current is too small
                            chunk_text = current_small_chunk_text + "\n\n" + chunk_text
                            current_small_chunk_text = ""
                    # Add the split chunk
                    small_chunks.append(self._create_small_chunk_state(
                        small_chunk_index, big_chunk['big_chunk_id'], chunk_text,
                        len(chunk_text), position_in_big_chunk
                    ))
                    small_chunk_index += 1
                    position_in_big_chunk += len(chunk_text)
            else:
                # Try to add paragraph to current chunk
                if len(current_small_chunk_text) + len(paragraph) + 2 <= max_size:
                    if current_small_chunk_text:
                        current_small_chunk_text += "\n\n" + paragraph
                    else:
                        current_small_chunk_text = paragraph
                else:
                    # If current chunk is at least min_size, add it
                    if len(current_small_chunk_text) >= min_size:
                        small_chunks.append(self._create_small_chunk_state(
                            small_chunk_index, big_chunk['big_chunk_id'], current_small_chunk_text,
                            len(current_small_chunk_text), position_in_big_chunk
                        ))
                        small_chunk_index += 1
                        position_in_big_chunk += len(current_small_chunk_text)
                        current_small_chunk_text = paragraph
                    else:
                        # Merge with the new paragraph if current is too small
                        current_small_chunk_text += "\n\n" + paragraph
        # Add the last small chunk if there's remaining text
        if current_small_chunk_text:
            if len(current_small_chunk_text) < min_size and small_chunks:
                # Merge with previous chunk
                prev = small_chunks.pop()
                merged_text = prev['small_chunk_text'] + "\n\n" + current_small_chunk_text
                small_chunks.append(self._create_small_chunk_state(
                    small_chunk_index-1, big_chunk['big_chunk_id'], merged_text,
                    len(merged_text), prev['position_in_big_chunk']
                ))
            else:
                small_chunks.append(self._create_small_chunk_state(
                    small_chunk_index, big_chunk['big_chunk_id'], current_small_chunk_text,
                    len(current_small_chunk_text), position_in_big_chunk
                ))
        # Assert all chunks are non-empty, within size range, and unique
        seen_texts = set()
        total_size = 0
        for i, chunk in enumerate(small_chunks):
            assert chunk['small_chunk_text'].strip(), f"Small chunk {i} is empty"
            # assert 400 <= chunk['small_chunk_size'] <= 700, f"Small chunk {i} size {chunk['small_chunk_size']} out of range"
            assert chunk['small_chunk_text'] not in seen_texts, f"Duplicate small chunk text at index {i}"
            seen_texts.add(chunk['small_chunk_text'])
            logger.debug(f"Small chunk {i}: size={chunk['small_chunk_size']}, index={i}, text=\n{chunk['small_chunk_text']}\n---")
            total_size += chunk['small_chunk_size']
        logger.debug(f"Sum of all small chunk sizes: {total_size}, input big chunk size: {len(big_chunk['big_chunk_text'])}")
        assert abs(total_size - len(big_chunk['big_chunk_text'])) < 20, f"Sum of chunk sizes ({total_size}) does not match input ({len(big_chunk['big_chunk_text'])})"
        return small_chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text for consistent processing"""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize line breaks but keep paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Ensure proper spacing around punctuation, but don't break paragraph structure
        text = re.sub(r'([.!?。！？，,；;：:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(self.paragraph_breaks, text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _chunk_paragraph(self, paragraph: str, start_index: int) -> List[ChunkState]:
        """Split a paragraph into chunks while preserving sentence boundaries"""
        chunks = []
        current_index = start_index
        remaining_text = paragraph
        
        while len(remaining_text) > 0:
            # Determine chunk size for this iteration
            chunk_size = min(self.config.max_chunk_size, len(remaining_text))
            
            if len(remaining_text) <= self.config.max_chunk_size:
                # Last chunk - take all remaining text
                chunk_text = remaining_text
                remaining_text = ""
            else:
                # Find the best break point within the chunk size
                chunk_text, remaining_text = self._find_optimal_break(
                    remaining_text, chunk_size
                )
            
            # Create chunk state
            chunks.append(self._create_chunk_state(
                current_index, chunk_text, len(chunk_text)
            ))
            current_index += 1
        
        return chunks
    
    def _find_optimal_break(self, text: str, max_size: int) -> Tuple[str, str]:
        """
        Find the optimal break point in text to create a chunk of maximum size.
        Prioritizes sentence boundaries, then phrase boundaries, then word boundaries.
        """
        if len(text) <= max_size:
            return text, ""
        
        # Try to break at sentence boundaries first
        if self.config.preserve_sentences:
            sentence_break = self._find_sentence_break(text, max_size)
            if sentence_break > 0:
                return text[:sentence_break], text[sentence_break:]
        
        # Try to break at phrase boundaries
        phrase_break = self._find_phrase_break(text, max_size)
        if phrase_break > 0:
            return text[:phrase_break], text[phrase_break:]
        
        # Try to break at word boundaries
        word_break = self._find_word_break(text, max_size)
        if word_break > 0:
            return text[:word_break], text[word_break:]
        
        # If no good break point found, break at max_size
        return text[:max_size], text[max_size:]
    
    def _find_sentence_break(self, text: str, max_size: int) -> int:
        """Find the last sentence boundary within max_size"""
        # Look for sentence endings in the range [min_size, max_size]
        min_size = max(self.config.min_chunk_size, max_size - 200)
        
        # Find all sentence endings in the text
        sentence_matches = list(re.finditer(self.sentence_endings, text))
        
        # Find the last sentence ending within our range
        for match in reversed(sentence_matches):
            if min_size <= match.end() <= max_size:
                return match.end()
        
        return 0
    
    def _find_phrase_break(self, text: str, max_size: int) -> int:
        """Find the last phrase boundary within max_size"""
        min_size = max(self.config.min_chunk_size, max_size - 100)
        
        # Find all phrase boundaries in the text
        phrase_matches = list(re.finditer(self.phrase_boundaries, text))
        
        # Find the last phrase boundary within our range
        for match in reversed(phrase_matches):
            if min_size <= match.end() <= max_size:
                return match.end()
        
        return 0
    
    def _find_word_break(self, text: str, max_size: int) -> int:
        """Find the last word boundary within max_size"""
        min_size = max(self.config.min_chunk_size, max_size - 50)
        
        # Find all word boundaries in the text
        word_matches = list(re.finditer(self.word_boundaries, text))
        
        # Find the last word boundary within our range
        for match in reversed(word_matches):
            if min_size <= match.end() <= max_size:
                return match.end()
        
        return 0
    
    def _create_chunk_state(self, index: int, text: str, size: int) -> ChunkState:
        """Create a ChunkState object for a chunk"""
        return ChunkState(
            chunk_index=index,
            chunk_text=text.strip(),
            chunk_size=size,
            is_processed=False,
            translation_attempts=0,
            max_attempts=3
        )
    
    def _create_big_chunk_state(self, index: int, text: str, size: int) -> BigChunkState:
        """Create a BigChunkState object for a big chunk"""
        return BigChunkState(
            big_chunk_id=f"big_chunk_{index:04d}",
            big_chunk_text=text.strip(),
            big_chunk_size=size,
            memory_context=[],
            small_chunks=[],
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
    
    def _create_small_chunk_state(self, index: int, big_chunk_id: str, text: str, size: int, position: int) -> SmallChunkState:
        """Create a SmallChunkState object for a small chunk"""
        return SmallChunkState(
            small_chunk_id=f"small_chunk_{big_chunk_id}_{index:04d}",
            big_chunk_id=big_chunk_id,
            small_chunk_text=text.strip(),
            small_chunk_size=size,
            position_in_big_chunk=position,
            translated_text=None,
            recent_context=[],
            translation_attempts=0,
            max_attempts=3,
            is_processed=False,
            processing_status="pending",
            error_message=None
        )
    
    def merge_small_chunks(self, chunks: List[ChunkState]) -> List[ChunkState]:
        """
        Merge chunks that are too small with adjacent chunks.
        
        Args:
            chunks: List of chunks to merge
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            combined_size = len(current_chunk['chunk_text']) + len(next_chunk['chunk_text'])
            
            # If combining would create a reasonable chunk size, merge them
            if combined_size <= self.config.max_chunk_size:
                merged_text = current_chunk['chunk_text'] + ' ' + next_chunk['chunk_text']
                current_chunk = ChunkState(
                    chunk_index=current_chunk['chunk_index'],
                    chunk_text=merged_text,
                    chunk_size=len(merged_text),
                    is_processed=False,
                    translation_attempts=0,
                    max_attempts=3
                )
            else:
                # Keep current chunk and start a new one
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # Add the last chunk
        merged_chunks.append(current_chunk)
        
        return merged_chunks


def chunk_vietnamese_text(text: str, config: Optional[ChunkingConfig] = None) -> List[ChunkState]:
    """
    Convenience function to chunk Vietnamese text.
    
    Args:
        text: Vietnamese text to chunk
        config: Optional chunking configuration
        
    Returns:
        List of ChunkState objects
    """
    chunker = VietnameseTextChunker(config)
    chunks = chunker.chunk_text(text)
    return chunker.merge_small_chunks(chunks)


def chunk_text_into_big_chunks(text: str, config: Optional[EnhancedChunkingConfig] = None) -> List[BigChunkState]:
    """
    Convenience function to chunk Vietnamese text into big chunks for enhanced translation flow.
    
    Args:
        text: Vietnamese text to chunk
        config: Optional enhanced chunking configuration
        
    Returns:
        List of BigChunkState objects
    """
    chunker = VietnameseTextChunker()
    return chunker.chunk_text_into_big_chunks(text, config)


def chunk_big_chunk_into_small_chunks(big_chunk: BigChunkState, config: Optional[EnhancedChunkingConfig] = None) -> List[SmallChunkState]:
    """
    Convenience function to split a big chunk into small chunks for enhanced translation flow.
    
    Args:
        big_chunk: BigChunkState to split into small chunks
        config: Optional enhanced chunking configuration
        
    Returns:
        List of SmallChunkState objects
    """
    chunker = VietnameseTextChunker()
    return chunker.chunk_big_chunk_into_small_chunks(big_chunk, config)


def validate_enhanced_chunks(big_chunks: List[BigChunkState], small_chunks: List[SmallChunkState], config: Optional[EnhancedChunkingConfig] = None, strict: bool = False) -> List[str]:
    """
    Validate enhanced chunks for consistency and quality.
    
    Args:
        big_chunks: List of big chunks to validate
        small_chunks: List of small chunks to validate
        config: Optional enhanced chunking configuration
        strict: Whether to enforce strict size limits (default: False for flexibility)
        
    Returns:
        List of validation error messages
    """
    errors = []
    config = config or EnhancedChunkingConfig()
    
    # Validate big chunks
    if not big_chunks:
        errors.append("No big chunks provided")
    else:
        for i, big_chunk in enumerate(big_chunks):
            if not big_chunk['big_chunk_text'].strip():
                errors.append(f"Big chunk {i} is empty")
            
            if big_chunk['big_chunk_size'] > config.big_chunk_size:
                errors.append(f"Big chunk {i} exceeds maximum size: {big_chunk['big_chunk_size']} > {config.big_chunk_size}")
            
            # Only check minimum size if strict validation is enabled
            if strict and big_chunk['big_chunk_size'] < config.big_chunk_min_size:
                errors.append(f"Big chunk {i} is too small: {big_chunk['big_chunk_size']} < {config.big_chunk_min_size}")
    
    # Validate small chunks
    if not small_chunks:
        errors.append("No small chunks provided")
    else:
        for i, small_chunk in enumerate(small_chunks):
            if not small_chunk['small_chunk_text'].strip():
                errors.append(f"Small chunk {i} is empty")
            
            if small_chunk['small_chunk_size'] > config.small_chunk_size * 2:  # Allow some flexibility
                errors.append(f"Small chunk {i} is too large: {small_chunk['small_chunk_size']} > {config.small_chunk_size * 2}")
            
            # Only check minimum size if strict validation is enabled
            if strict and small_chunk['small_chunk_size'] < config.small_chunk_min_size:
                errors.append(f"Small chunk {i} is too small: {small_chunk['small_chunk_size']} < {config.small_chunk_min_size}")
    
    # Validate relationships between big and small chunks
    big_chunk_ids = {chunk['big_chunk_id'] for chunk in big_chunks}
    for small_chunk in small_chunks:
        if small_chunk['big_chunk_id'] not in big_chunk_ids:
            errors.append(f"Small chunk references non-existent big chunk: {small_chunk['big_chunk_id']}")
    
    return errors


def reassemble_chunks_from_small_chunks(small_chunks: List[SmallChunkState]) -> str:
    """
    Reassemble text from small chunks, preserving original order.
    
    Args:
        small_chunks: List of small chunks to reassemble
        
    Returns:
        Reassembled text
    """
    if not small_chunks:
        return ""
    
    # Sort small chunks by their position in the big chunk
    sorted_chunks = sorted(small_chunks, key=lambda x: x['position_in_big_chunk'])
    
    # Reassemble text
    reassembled_text = ""
    for chunk in sorted_chunks:
        if reassembled_text:
            reassembled_text += "\n\n"
        reassembled_text += chunk['small_chunk_text']
    
    return reassembled_text


def reassemble_chunks_from_big_chunks(big_chunks: List[BigChunkState]) -> str:
    """
    Reassemble text from big chunks, preserving original order.
    
    Args:
        big_chunks: List of big chunks to reassemble
        
    Returns:
        Reassembled text
    """
    if not big_chunks:
        return ""
    
    # Sort big chunks by their ID (which contains the index)
    sorted_chunks = sorted(big_chunks, key=lambda x: x['big_chunk_id'])
    
    # Reassemble text
    reassembled_text = ""
    for chunk in sorted_chunks:
        if reassembled_text:
            reassembled_text += "\n\n"
        reassembled_text += chunk['big_chunk_text']
    
    return reassembled_text


def preserve_context_between_small_chunks(small_chunks: List[SmallChunkState], context_window: int = 2) -> List[SmallChunkState]:
    """
    Add recent context to each small chunk for better translation continuity.
    
    Args:
        small_chunks: List of small chunks to add context to
        context_window: Number of previous chunks to include as context
        
    Returns:
        List of small chunks with context added
    """
    if not small_chunks:
        return small_chunks
    
    # Sort chunks by position
    sorted_chunks = sorted(small_chunks, key=lambda x: x['position_in_big_chunk'])
    
    for i, chunk in enumerate(sorted_chunks):
        # Get recent context from previous chunks
        context_start = max(0, i - context_window)
        context_chunks = sorted_chunks[context_start:i]
        
        # Create context information
        context_info = []
        for ctx_chunk in context_chunks:
            context_info.append({
                'chunk_id': ctx_chunk['small_chunk_id'],
                'text': ctx_chunk['small_chunk_text'],
                'translated_text': ctx_chunk.get('translated_text'),
                'position': ctx_chunk['position_in_big_chunk']
            })
        
        # Update the chunk with context
        chunk['recent_context'] = context_info
    
    return sorted_chunks


def analyze_enhanced_chunk_quality(big_chunks: List[BigChunkState], small_chunks: List[SmallChunkState]) -> Dict[str, Any]:
    """
    Analyze the quality of enhanced chunking.
    
    Args:
        big_chunks: List of big chunks to analyze
        small_chunks: List of small chunks to analyze
        
    Returns:
        Dictionary with analysis metrics
    """
    analysis = {
        'big_chunks': {
            'total': len(big_chunks),
            'average_size': 0,
            'min_size': 0,
            'max_size': 0,
            'size_distribution': {}
        },
        'small_chunks': {
            'total': len(small_chunks),
            'average_size': 0,
            'min_size': 0,
            'max_size': 0,
            'size_distribution': {}
        },
        'relationships': {
            'chunks_per_big_chunk': 0,
            'coverage_ratio': 0.0
        }
    }
    
    # Analyze big chunks
    if big_chunks:
        big_sizes = [chunk['big_chunk_size'] for chunk in big_chunks]
        analysis['big_chunks']['average_size'] = sum(big_sizes) / len(big_sizes)
        analysis['big_chunks']['min_size'] = min(big_sizes)
        analysis['big_chunks']['max_size'] = max(big_sizes)
        analysis['big_chunks']['size_distribution'] = {
            'small (<8k)': len([s for s in big_sizes if s < 8000]),
            'medium (8k-12k)': len([s for s in big_sizes if 8000 <= s < 12000]),
            'large (>12k)': len([s for s in big_sizes if s >= 12000])
        }
    
    # Analyze small chunks
    if small_chunks:
        small_sizes = [chunk['small_chunk_size'] for chunk in small_chunks]
        analysis['small_chunks']['average_size'] = sum(small_sizes) / len(small_sizes)
        analysis['small_chunks']['min_size'] = min(small_sizes)
        analysis['small_chunks']['max_size'] = max(small_sizes)
        analysis['small_chunks']['size_distribution'] = {
            'small (<300)': len([s for s in small_sizes if s < 300]),
            'medium (300-700)': len([s for s in small_sizes if 300 <= s < 700]),
            'large (>700)': len([s for s in small_sizes if s >= 700])
        }
    
    # Analyze relationships
    if big_chunks and small_chunks:
        chunks_per_big = len(small_chunks) / len(big_chunks)
        analysis['relationships']['chunks_per_big_chunk'] = chunks_per_big
        
        # Calculate coverage ratio (total small chunk size / total big chunk size)
        total_big_size = sum(chunk['big_chunk_size'] for chunk in big_chunks)
        total_small_size = sum(chunk['small_chunk_size'] for chunk in small_chunks)
        if total_big_size > 0:
            analysis['relationships']['coverage_ratio'] = total_small_size / total_big_size
    
    return analysis


def analyze_chunk_quality(chunks: List[ChunkState]) -> Dict[str, Any]:
    """
    Analyze the quality of chunking.
    
    Args:
        chunks: List of chunks to analyze
        
    Returns:
        Dictionary with analysis metrics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'average_chunk_size': 0,
            'min_chunk_size': 0,
            'max_chunk_size': 0,
            'size_distribution': {}
        }
    
    sizes = [len(chunk['chunk_text']) for chunk in chunks]
    
    return {
        'total_chunks': len(chunks),
        'average_chunk_size': sum(sizes) / len(sizes),
        'min_chunk_size': min(sizes),
        'max_chunk_size': max(sizes),
        'size_distribution': {
            'small (<500)': len([s for s in sizes if s < 500]),
            'medium (500-1000)': len([s for s in sizes if 500 <= s < 1000]),
            'large (>1000)': len([s for s in sizes if s >= 1000])
        }
    }


def validate_chunks(chunks: List[ChunkState]) -> List[str]:
    """
    Validate a list of chunks for consistency and quality.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    if not chunks:
        errors.append("No chunks provided")
        return errors
    
    # Check for empty chunks
    for i, chunk in enumerate(chunks):
        if not chunk['chunk_text'].strip():
            errors.append(f"Chunk {i} is empty")
        
        if chunk['chunk_index'] != i:
            errors.append(f"Chunk {i} has incorrect index: {chunk['chunk_index']}")
    
    # Check for overlapping or missing text
    all_text = ''.join(chunk['chunk_text'] for chunk in chunks)
    if len(all_text.strip()) == 0:
        errors.append("All chunks combined produce empty text")
    
    return errors
