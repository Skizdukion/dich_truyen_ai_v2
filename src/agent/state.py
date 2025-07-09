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
    retrieved_nodes: Annotated[List[Dict[str, Any]], operator.add]  # Nodes found during search
    created_nodes: Annotated[List[Dict[str, Any]], operator.add]  # New nodes created
    updated_nodes: Annotated[List[Dict[str, Any]], operator.add]  # Nodes that were updated
    search_queries: Annotated[List[str], operator.add]  # Queries sent to Weaviate
    memory_operations: Annotated[List[Dict[str, Any]], operator.add]  # Log of all memory operations


class ChunkState(TypedDict):
    """State for individual chunk processing"""
    chunk_index: int
    chunk_text: str
    chunk_size: int
    is_processed: bool
    translation_attempts: int
    max_attempts: int


class OverallState(TypedDict):
    """Main state for the translation workflow"""
    messages: Annotated[list, add_messages]
    # Translation-specific fields
    input_text: str  # Original Vietnamese text input
    chunks: List[ChunkState]  # List of chunks to process
    translated_text: Annotated[List[str], operator.add]  # Accumulated translated text
    memory_context: Annotated[List[Dict[str, Any]], operator.add]  # Recent context for continuity
    # Memory and processing state
    memory_state: MemoryState
    translation_state: TranslationState
    # Configuration and tracking
    chunk_size: int  # Size of chunks to process
    current_chunk_index: int  # Index of current chunk being processed
    total_chunks: int  # Total number of chunks
    processing_complete: bool  # Whether all chunks have been processed
    # Error handling and retry logic
    failed_chunks: Annotated[List[int], operator.add]  # Indices of failed chunks
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


# State validation functions
def validate_translation_state(state: TranslationState) -> List[str]:
    """Validate TranslationState and return list of validation errors"""
    errors = []
    
    # Required fields validation
    if not state.get('chunk_id'):
        errors.append("TranslationState: chunk_id is required")
    
    if not state.get('original_text'):
        errors.append("TranslationState: original_text is required")
    
    # Processing status validation
    valid_statuses = ["pending", "processing", "completed", "failed"]
    if state.get('processing_status') not in valid_statuses:
        errors.append(f"TranslationState: processing_status must be one of {valid_statuses}")
    
    # Translation quality validation (if present)
    if state.get('translation_quality') and not isinstance(state['translation_quality'], str):
        errors.append("TranslationState: translation_quality must be a string")
    
    # Memory context validation
    if not isinstance(state.get('memory_context', []), list):
        errors.append("TranslationState: memory_context must be a list")
    
    return errors


def validate_memory_state(state: MemoryState) -> List[str]:
    """Validate MemoryState and return list of validation errors"""
    errors = []
    
    # Validate all list fields are actually lists
    list_fields = ['retrieved_nodes', 'created_nodes', 'updated_nodes', 'search_queries', 'memory_operations']
    for field in list_fields:
        if not isinstance(state.get(field, []), list):
            errors.append(f"MemoryState: {field} must be a list")
    
    # Validate memory operations structure
    for operation in state.get('memory_operations', []):
        if not isinstance(operation, dict):
            errors.append("MemoryState: memory_operations must contain dictionaries")
            continue
        
        required_fields = ['operation_type', 'node_type', 'query_or_content', 'result', 'timestamp']
        for field in required_fields:
            if field not in operation:
                errors.append(f"MemoryState: memory_operations must contain '{field}' field")
    
    return errors


def validate_chunk_state(state: ChunkState) -> List[str]:
    """Validate ChunkState and return list of validation errors"""
    errors = []
    
    # Required fields validation
    if not isinstance(state.get('chunk_index'), int):
        errors.append("ChunkState: chunk_index must be an integer")
    
    if not state.get('chunk_text'):
        errors.append("ChunkState: chunk_text is required")
    
    if not isinstance(state.get('chunk_size'), int) or state.get('chunk_size', 0) <= 0:
        errors.append("ChunkState: chunk_size must be a positive integer")
    
    # Boolean validation
    if not isinstance(state.get('is_processed'), bool):
        errors.append("ChunkState: is_processed must be a boolean")
    
    # Attempts validation
    if not isinstance(state.get('translation_attempts'), int) or state.get('translation_attempts', 0) < 0:
        errors.append("ChunkState: translation_attempts must be a non-negative integer")
    
    if not isinstance(state.get('max_attempts'), int) or state.get('max_attempts', 0) <= 0:
        errors.append("ChunkState: max_attempts must be a positive integer")
    
    # Logical validation
    if state.get('translation_attempts', 0) > state.get('max_attempts', 0):
        errors.append("ChunkState: translation_attempts cannot exceed max_attempts")
    
    return errors


def validate_overall_state(state: OverallState) -> List[str]:
    """Validate OverallState and return list of validation errors"""
    errors = []
    
    # Required fields validation
    if not state.get('input_text'):
        errors.append("OverallState: input_text is required")
    
    if not isinstance(state.get('chunks'), list):
        errors.append("OverallState: chunks must be a list")
    
    # Validate chunks list
    chunks = state.get('chunks', [])
    if not isinstance(chunks, list):
        errors.append("OverallState: chunks must be a list")
    else:
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                errors.append(f"OverallState: chunk[{i}] must be a dictionary")
                continue
            chunk_errors = validate_chunk_state(chunk)
            for error in chunk_errors:
                errors.append(f"OverallState: chunk[{i}]: {error}")
    
    # Configuration validation
    if not isinstance(state.get('chunk_size'), int) or state.get('chunk_size', 0) <= 0:
        errors.append("OverallState: chunk_size must be a positive integer")
    
    if not isinstance(state.get('current_chunk_index'), int) or state.get('current_chunk_index', 0) < 0:
        errors.append("OverallState: current_chunk_index must be a non-negative integer")
    
    if not isinstance(state.get('total_chunks'), int) or state.get('total_chunks', 0) < 0:
        errors.append("OverallState: total_chunks must be a non-negative integer")
    
    # Logical validation
    current_index = state.get('current_chunk_index', 0)
    total_chunks = state.get('total_chunks', 0)
    if total_chunks > 0 and current_index >= total_chunks:
        errors.append("OverallState: current_chunk_index cannot exceed total_chunks")
    
    chunks_list = state.get('chunks', [])
    total_chunks = state.get('total_chunks', 0)
    if total_chunks > 0 and len(chunks_list) != total_chunks:
        errors.append("OverallState: chunks list length must match total_chunks")
    
    # Boolean validation
    if not isinstance(state.get('processing_complete'), bool):
        errors.append("OverallState: processing_complete must be a boolean")
    
    # Retry count validation
    if not isinstance(state.get('retry_count'), int) or state.get('retry_count', 0) < 0:
        errors.append("OverallState: retry_count must be a non-negative integer")
    
    # Validate nested states
    if 'memory_state' in state:
        memory_errors = validate_memory_state(state['memory_state'])
        for error in memory_errors:
            errors.append(f"OverallState: memory_state: {error}")
    
    if 'translation_state' in state:
        translation_errors = validate_translation_state(state['translation_state'])
        for error in translation_errors:
            errors.append(f"OverallState: translation_state: {error}")
    
    return errors


def validate_knowledge_node(node: KnowledgeNode) -> List[str]:
    """Validate KnowledgeNode and return list of validation errors"""
    errors = []
    
    # Required fields validation
    if not node.get('type'):
        errors.append("KnowledgeNode: type is required")
    elif node['type'] not in ['GlossaryTerm', 'Character', 'Event']:
        errors.append("KnowledgeNode: type must be one of ['GlossaryTerm', 'Character', 'Event']")
    
    if not node.get('label'):
        errors.append("KnowledgeNode: label is required")
    
    if not node.get('content'):
        errors.append("KnowledgeNode: content is required")
    
    # Aliases validation
    if not isinstance(node.get('aliases', []), list):
        errors.append("KnowledgeNode: aliases must be a list")
    
    # Metadata validation
    if not isinstance(node.get('metadata', {}), dict):
        errors.append("KnowledgeNode: metadata must be a dictionary")
    
    return errors


def create_initial_state(input_text: str, chunk_size: int = 1000) -> OverallState:
    """Create a properly initialized OverallState for translation workflow"""
    return OverallState(
        messages=[],
        input_text=input_text,
        chunks=[],  # Will be populated by chunking logic
        translated_text=[],
        memory_context=[],
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
        failed_chunks=[],
        retry_count=0
    )


def is_state_valid(state: OverallState) -> bool:
    """Check if the overall state is valid"""
    return len(validate_overall_state(state)) == 0


def get_state_validation_errors(state: OverallState) -> List[str]:
    """Get all validation errors for the state"""
    return validate_overall_state(state)
