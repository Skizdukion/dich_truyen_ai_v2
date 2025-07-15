from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, List, Optional, Dict, Any

from langgraph.graph import add_messages
from typing_extensions import Annotated

import operator
from datetime import datetime


class TranslationState(TypedDict):
    """State for processing individual chunks during translation"""
    chunk_id: str
    original_text: str
    translated_text: Optional[str]
    memory_context: List[Dict[str, Any]]  # Retrieved nodes from Weaviate
    translation_quality: Optional[str]  # LLM feedback on translation quality
    processing_status: str  # "pending", "processing", "completed", "failed"
    error_message: Optional[str]


class MemoryState(TypedDict):
    """State for Weaviate memory operations"""
    retrieved_nodes: List[Dict[str, Any]]  # Nodes found during search
    created_nodes: List[Dict[str, Any]]  # New nodes created
    updated_nodes: List[Dict[str, Any]]  # Nodes that were updated
    search_queries: List[str]  # Queries sent to Weaviate
    memory_operations: List[Dict[str, Any]]  # Log of all memory operations


class ChunkState(TypedDict):
    """State for individual chunk processing"""
    chunk_index: int
    chunk_text: str
    chunk_size: int
    is_processed: bool
    translation_attempts: int
    max_attempts: int


class OverallState(TypedDict):
    """Main state for the translation workflow - simplified and maintainable"""
    messages: Annotated[list, add_messages]
    
    # Core translation data
    input_text: str  # Original Vietnamese text input
    chunks: List[ChunkState]  # List of chunks to process
    translated_chunks: List[str]  # Accumulated translated chunks (no operator.add)
    memory_context: List[Dict[str, Any]]  # Recent context for continuity (no operator.add)
    
    # Processing state
    current_chunk_index: int  # Index of current chunk being processed
    total_chunks: int  # Total number of chunks
    processing_complete: bool  # Whether all chunks have been processed
    
    # Current operation state
    translation_state: TranslationState
    memory_state: MemoryState
    
    # Configuration
    chunk_size: int  # Size of chunks to process
    
    # Error handling
    failed_chunks: List[int]  # Indices of failed chunks (no operator.add)
    retry_count: int  # Number of retry attempts for current chunk


class KnowledgeNode(TypedDict):
    """Schema for Weaviate knowledge nodes"""
    id: Optional[str]
    type: str  # "GlossaryTerm", "Character", "Event"
    label: str
    content: str
    aliases: List[str]
    metadata: Dict[str, Any]


class MemoryOperation(TypedDict):
    """Schema for memory operations"""
    operation_type: str  # "search", "create", "update"
    node_type: str
    query_or_content: str
    result: Dict[str, Any]
    timestamp: str


@dataclass(kw_only=True)
class TranslationOutput:
    """Output structure for translation results"""
    final_translation: str = field(default="")
    memory_summary: Dict[str, Any] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)


def create_initial_state(input_text: str, chunk_size: int = 1000) -> OverallState:
    """Create a properly initialized OverallState for translation workflow"""
    return OverallState(
        messages=[],
        input_text=input_text,
        chunks=[],  # Will be populated by chunking logic
        translated_chunks=[],  # Simple list, no operator.add
        memory_context=[],  # Simple list, no operator.add
        memory_state=MemoryState(
            retrieved_nodes=[],
            created_nodes=[],
            updated_nodes=[],
            search_queries=[],
            memory_operations=[]
        ),
        translation_state=TranslationState(
            chunk_id="initial",
            original_text=input_text,
            translated_text=None,
            memory_context=[],
            translation_quality=None,
            processing_status="pending",
            error_message=None
        ),
        chunk_size=chunk_size,
        current_chunk_index=0,
        total_chunks=0,
        processing_complete=False,
        failed_chunks=[],  # Simple list, no operator.add
        retry_count=0
    )
