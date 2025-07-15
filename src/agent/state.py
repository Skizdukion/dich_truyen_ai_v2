from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum

from langgraph.graph import add_messages
from typing_extensions import Annotated

import operator
from datetime import datetime


class ReviewRating(Enum):
    """Review rating levels for translation quality assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


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


class BigChunkState(TypedDict):
    """State for big chunk processing (16k limit)"""
    big_chunk_id: str
    big_chunk_text: str
    big_chunk_size: int
    memory_context: List[Dict[str, Any]]  # Retrieved nodes for this big chunk
    small_chunks: List[SmallChunkState]  # List of small chunks within this big chunk
    is_processed: bool
    processing_status: str  # "pending", "processing", "completed", "failed"
    error_message: Optional[str]


class SmallChunkState(TypedDict):
    """State for small chunk processing (~500 words)"""
    small_chunk_id: str
    big_chunk_id: str  # Reference to parent big chunk
    small_chunk_text: str
    small_chunk_size: int
    position_in_big_chunk: int  # Position within the big chunk
    translated_text: Optional[str]
    recent_context: List[Dict[str, Any]]  # Context from previous small chunks
    translation_attempts: int
    max_attempts: int
    is_processed: bool
    processing_status: str  # "pending", "processing", "completed", "failed"
    error_message: Optional[str]


class ReviewState(TypedDict):
    """State for review and feedback processing"""
    chunk_id: str  # ID of the chunk being reviewed
    original_text: str  # Original text for comparison
    translated_text: str  # The translation to be reviewed
    context: Optional[str]  # Additional context for review
    rating: Optional[ReviewRating]  # Rating from ReviewRating enum
    feedback: Optional[str]  # Detailed feedback for the translation
    confidence: float  # Confidence score from 0.0 to 1.0
    requires_revision: bool  # Whether the translation needs revision
    review_timestamp: Optional[datetime]  # When the review was performed
    reviewer_id: Optional[str]  # ID of the reviewer (model or human)


class RetryState(TypedDict):
    """State for retry mechanism with feedback incorporation"""
    retry_id: str
    small_chunk_id: str  # Reference to the small chunk being retried
    original_translation: str  # The failed translation
    feedback: str  # Feedback from review
    retry_attempt: int
    max_retries: int
    new_translation: Optional[str]
    retry_status: str  # "pending", "retrying", "completed", "failed"
    error_message: Optional[str]


class OverallState(TypedDict):
    """Main state for the enhanced translation workflow"""
    messages: Annotated[list, add_messages]
    
    # Core translation data
    input_text: str  # Original Vietnamese text input
    big_chunks: List[BigChunkState]  # List of big chunks (16k limit)
    small_chunks: List[SmallChunkState]  # List of all small chunks (~500 words)
    translated_chunks: List[str]  # Accumulated translated chunks (no operator.add)
    memory_context: List[Dict[str, Any]]  # Recent context for continuity (no operator.add)
    big_chunk_memory_context: List[Dict[str, Any]]  # Memory context for current big chunk
    
    # Review and retry tracking
    review_states: List[ReviewState]  # List of review states
    retry_states: List[RetryState]  # List of retry states
    failed_translations: List[str]  # List of failed translation IDs (no operator.add)
    
    # Processing state
    current_big_chunk_index: int  # Index of current big chunk being processed
    current_small_chunk_index: int  # Index of current small chunk being processed
    total_big_chunks: int  # Total number of big chunks
    total_small_chunks: int  # Total number of small chunks
    processing_complete: bool  # Whether all chunks have been processed
    
    # Current operation state
    translation_state: TranslationState
    memory_state: MemoryState
    
    # Configuration
    big_chunk_size: int  # Size of big chunks (16k limit)
    small_chunk_size: int  # Size of small chunks (~500 words)
    
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


# Validation functions
def validate_translation_state(state: TranslationState) -> List[str]:
    """Validate TranslationState and return list of error messages"""
    errors = []
    
    if not state.get("chunk_id"):
        errors.append("chunk_id is required")
    
    if not state.get("original_text"):
        errors.append("original_text is required")
    
    if not isinstance(state.get("memory_context", []), list):
        errors.append("memory_context must be a list")
    
    valid_statuses = ["pending", "processing", "completed", "failed"]
    if state.get("processing_status") not in valid_statuses:
        errors.append(f"processing_status must be one of {valid_statuses}")
    
    return errors


def validate_memory_state(state: MemoryState) -> List[str]:
    """Validate MemoryState and return list of error messages"""
    errors = []
    
    if not isinstance(state.get("retrieved_nodes", []), list):
        errors.append("retrieved_nodes must be a list")
    
    if not isinstance(state.get("created_nodes", []), list):
        errors.append("created_nodes must be a list")
    
    if not isinstance(state.get("updated_nodes", []), list):
        errors.append("updated_nodes must be a list")
    
    if not isinstance(state.get("search_queries", []), list):
        errors.append("search_queries must be a list")
    
    memory_operations = state.get("memory_operations", [])
    if not isinstance(memory_operations, list):
        errors.append("memory_operations must be a list")
    else:
        for op in memory_operations:
            if not isinstance(op, dict):
                errors.append("memory_operations must contain dictionaries")
                break
            required_fields = ["operation_type", "node_type", "query_or_content", "result", "timestamp"]
            for field in required_fields:
                if field not in op:
                    errors.append(f"memory_operations must contain {field} field")
    
    return errors


def validate_chunk_state(state: ChunkState) -> List[str]:
    """Validate ChunkState and return list of error messages"""
    errors = []
    
    if not isinstance(state.get("chunk_index"), int):
        errors.append("chunk_index must be an integer")
    
    if not state.get("chunk_text"):
        errors.append("chunk_text is required")
    
    chunk_size = state.get("chunk_size")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        errors.append("chunk_size must be a positive integer")
    
    translation_attempts = state.get("translation_attempts", 0)
    max_attempts = state.get("max_attempts", 3)
    
    if not isinstance(translation_attempts, int) or translation_attempts < 0:
        errors.append("translation_attempts must be a non-negative integer")
    
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        errors.append("max_attempts must be a positive integer")
    
    if isinstance(translation_attempts, int) and isinstance(max_attempts, int) and translation_attempts > max_attempts:
        errors.append("translation_attempts cannot exceed max_attempts")
    
    if not isinstance(state.get("is_processed"), bool):
        errors.append("is_processed must be a boolean")
    
    return errors


def validate_big_chunk_state(state: BigChunkState) -> List[str]:
    """Validate BigChunkState and return list of error messages"""
    errors = []
    
    if not state.get("big_chunk_id"):
        errors.append("big_chunk_id is required")
    
    if not state.get("big_chunk_text"):
        errors.append("big_chunk_text is required")
    
    big_chunk_size = state.get("big_chunk_size")
    if not isinstance(big_chunk_size, int) or big_chunk_size <= 0:
        errors.append("big_chunk_size must be a positive integer")
    
    if not isinstance(state.get("memory_context", []), list):
        errors.append("memory_context must be a list")
    
    small_chunks = state.get("small_chunks", [])
    if not isinstance(small_chunks, list):
        errors.append("small_chunks must be a list")
    else:
        for chunk in small_chunks:
            chunk_errors = validate_small_chunk_state(chunk)
            errors.extend(chunk_errors)
    
    if not isinstance(state.get("is_processed"), bool):
        errors.append("is_processed must be a boolean")
    
    valid_statuses = ["pending", "processing", "completed", "failed"]
    if state.get("processing_status") not in valid_statuses:
        errors.append(f"processing_status must be one of {valid_statuses}")
    
    return errors


def validate_small_chunk_state(state: SmallChunkState) -> List[str]:
    """Validate SmallChunkState and return list of error messages"""
    errors = []
    
    if not state.get("small_chunk_id"):
        errors.append("small_chunk_id is required")
    
    if not state.get("big_chunk_id"):
        errors.append("big_chunk_id is required")
    
    if not state.get("small_chunk_text"):
        errors.append("small_chunk_text is required")
    
    small_chunk_size = state.get("small_chunk_size")
    if not isinstance(small_chunk_size, int) or small_chunk_size <= 0:
        errors.append("small_chunk_size must be a positive integer")
    
    position = state.get("position_in_big_chunk")
    if not isinstance(position, int) or position < 0:
        errors.append("position_in_big_chunk must be a non-negative integer")
    
    if not isinstance(state.get("recent_context", []), list):
        errors.append("recent_context must be a list")
    
    translation_attempts = state.get("translation_attempts", 0)
    max_attempts = state.get("max_attempts", 3)
    
    if not isinstance(translation_attempts, int) or translation_attempts < 0:
        errors.append("translation_attempts must be a non-negative integer")
    
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        errors.append("max_attempts must be a positive integer")
    
    if isinstance(translation_attempts, int) and isinstance(max_attempts, int) and translation_attempts > max_attempts:
        errors.append("translation_attempts cannot exceed max_attempts")
    
    if not isinstance(state.get("is_processed"), bool):
        errors.append("is_processed must be a boolean")
    
    valid_statuses = ["pending", "processing", "completed", "failed"]
    if state.get("processing_status") not in valid_statuses:
        errors.append(f"processing_status must be one of {valid_statuses}")
    
    return errors


def validate_review_state(state: ReviewState) -> List[str]:
    """Validate ReviewState and return list of error messages"""
    errors = []
    
    if not state.get("chunk_id"):
        errors.append("chunk_id is required")
    
    if not state.get("original_text"):
        errors.append("original_text is required")
    
    if not state.get("translated_text"):
        errors.append("translated_text is required")
    
    rating = state.get("rating")
    if rating is not None:
        if not isinstance(rating, ReviewRating):
            errors.append("rating must be a ReviewRating enum value")
    
    confidence = state.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
        errors.append("confidence must be a number between 0.0 and 1.0")
    
    if not isinstance(state.get("requires_revision"), bool):
        errors.append("requires_revision must be a boolean")
    
    return errors


def validate_retry_state(state: RetryState) -> List[str]:
    """Validate RetryState and return list of error messages"""
    errors = []
    
    if not state.get("retry_id"):
        errors.append("retry_id is required")
    
    if not state.get("small_chunk_id"):
        errors.append("small_chunk_id is required")
    
    if not state.get("original_translation"):
        errors.append("original_translation is required")
    
    if not state.get("feedback"):
        errors.append("feedback is required")
    
    retry_attempt = state.get("retry_attempt", 0)
    max_retries = state.get("max_retries", 3)
    
    if not isinstance(retry_attempt, int) or retry_attempt < 0:
        errors.append("retry_attempt must be a non-negative integer")
    
    if not isinstance(max_retries, int) or max_retries <= 0:
        errors.append("max_retries must be a positive integer")
    
    if retry_attempt > max_retries:
        errors.append("retry_attempt cannot exceed max_retries")
    
    valid_statuses = ["pending", "retrying", "completed", "failed"]
    if state.get("retry_status") not in valid_statuses:
        errors.append(f"retry_status must be one of {valid_statuses}")
    
    return errors


def validate_overall_state(state: OverallState) -> List[str]:
    """Validate OverallState and return list of error messages"""
    errors = []
    
    if not state.get("input_text"):
        errors.append("input_text is required")
    
    big_chunks = state.get("big_chunks", [])
    if not isinstance(big_chunks, list):
        errors.append("big_chunks must be a list")
    else:
        for chunk in big_chunks:
            chunk_errors = validate_big_chunk_state(chunk)
            errors.extend(chunk_errors)
    
    small_chunks = state.get("small_chunks", [])
    if not isinstance(small_chunks, list):
        errors.append("small_chunks must be a list")
    else:
        for chunk in small_chunks:
            chunk_errors = validate_small_chunk_state(chunk)
            errors.extend(chunk_errors)
    
    if not isinstance(state.get("translated_chunks", []), list):
        errors.append("translated_chunks must be a list")
    
    if not isinstance(state.get("memory_context", []), list):
        errors.append("memory_context must be a list")
    
    review_states = state.get("review_states", [])
    if not isinstance(review_states, list):
        errors.append("review_states must be a list")
    else:
        for review in review_states:
            review_errors = validate_review_state(review)
            errors.extend(review_errors)
    
    retry_states = state.get("retry_states", [])
    if not isinstance(retry_states, list):
        errors.append("retry_states must be a list")
    else:
        for retry in retry_states:
            retry_errors = validate_retry_state(retry)
            errors.extend(retry_errors)
    
    if not isinstance(state.get("failed_translations", []), list):
        errors.append("failed_translations must be a list")
    
    current_big_chunk_index = state.get("current_big_chunk_index", 0)
    if not isinstance(current_big_chunk_index, int) or current_big_chunk_index < 0:
        errors.append("current_big_chunk_index must be a non-negative integer")
    
    current_small_chunk_index = state.get("current_small_chunk_index", 0)
    if not isinstance(current_small_chunk_index, int) or current_small_chunk_index < 0:
        errors.append("current_small_chunk_index must be a non-negative integer")
    
    total_big_chunks = state.get("total_big_chunks", 0)
    if not isinstance(total_big_chunks, int) or total_big_chunks < 0:
        errors.append("total_big_chunks must be a non-negative integer")
    
    total_small_chunks = state.get("total_small_chunks", 0)
    if not isinstance(total_small_chunks, int) or total_small_chunks < 0:
        errors.append("total_small_chunks must be a non-negative integer")
    
    if not isinstance(state.get("processing_complete"), bool):
        errors.append("processing_complete must be a boolean")
    
    # Validate nested states
    translation_errors = validate_translation_state(state.get("translation_state", {}))
    # Only add original_text error if input_text is present
    if state.get("input_text"):
        errors.extend(translation_errors)
    else:
        errors.extend([e for e in translation_errors if "original_text is required" not in e])
    
    memory_errors = validate_memory_state(state.get("memory_state", {}))
    errors.extend(memory_errors)
    
    big_chunk_size = state.get("big_chunk_size")
    if not isinstance(big_chunk_size, int) or big_chunk_size <= 0:
        errors.append("big_chunk_size must be a positive integer")
    
    small_chunk_size = state.get("small_chunk_size")
    if not isinstance(small_chunk_size, int) or small_chunk_size <= 0:
        errors.append("small_chunk_size must be a positive integer")
    
    if not isinstance(state.get("failed_chunks", []), list):
        errors.append("failed_chunks must be a list")
    
    retry_count = state.get("retry_count", 0)
    if not isinstance(retry_count, int) or retry_count < 0:
        errors.append("retry_count must be a non-negative integer")
    
    return errors


def validate_knowledge_node(node: KnowledgeNode) -> List[str]:
    """Validate KnowledgeNode and return list of error messages"""
    errors = []
    
    if not node.get("type"):
        errors.append("type is required")
    else:
        valid_types = ["GlossaryTerm", "Character", "Event"]
        if node["type"] not in valid_types:
            errors.append(f"type must be one of {valid_types}")
    
    if not node.get("label"):
        errors.append("label is required")
    
    if not node.get("content"):
        errors.append("content is required")
    
    if not isinstance(node.get("aliases", []), list):
        errors.append("aliases must be a list")
    
    if not isinstance(node.get("metadata", {}), dict):
        errors.append("metadata must be a dictionary")
    
    return errors


def is_state_valid(state: OverallState) -> bool:
    """Check if OverallState is valid"""
    return len(validate_overall_state(state)) == 0


def get_state_validation_errors(state: OverallState) -> List[str]:
    """Get all validation errors for OverallState"""
    return validate_overall_state(state)


def create_initial_state(input_text: str, big_chunk_size: int = 16000, small_chunk_size: int = 500) -> OverallState:
    """Create a properly initialized OverallState for enhanced translation workflow"""
    return OverallState(
        messages=[],
        input_text=input_text,
        big_chunks=[],  # Will be populated by chunking logic
        small_chunks=[],  # Will be populated by chunking logic
        translated_chunks=[],  # Simple list, no operator.add
        memory_context=[],  # Simple list, no operator.add
        big_chunk_memory_context=[], # Simple list, no operator.add
        review_states=[],  # Will be populated during review process
        retry_states=[],  # Will be populated during retry process
        failed_translations=[],  # Simple list, no operator.add
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
        big_chunk_size=big_chunk_size,
        small_chunk_size=small_chunk_size,
        current_big_chunk_index=0,
        current_small_chunk_index=0,
        total_big_chunks=0,
        total_small_chunks=0,
        processing_complete=False,
        failed_chunks=[],  # Simple list, no operator.add
        retry_count=0
    )
