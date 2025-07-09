import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from src.agent.state import ChunkState


@dataclass
class ChunkingConfig:
    """Configuration for Vietnamese text chunking"""
    max_chunk_size: int = 6000  # Maximum characters per chunk
    min_chunk_size: int = 2000  # Minimum characters per chunk
    overlap_size: int = 0       # Overlap between chunks for context continuity
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
