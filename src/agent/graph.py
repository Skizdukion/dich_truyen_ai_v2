"""
Translation workflow graph for VietPhrase Reader Assistant.
Implements a memory-aware translation pipeline using LangGraph.
"""

from typing import Dict, Any, List, Tuple
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
import json
import logging

# Suppress httpx and httpcore logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from .state import OverallState, TranslationState, MemoryState, ChunkState
# from .prompts import (
#     translation_instructions,
#     memory_search_instructions,
#     memory_update_instructions,
#     context_summary_instructions
# )
from .configuration import Configuration
# from .utils import chunk_vietnamese_text
from weaviate_client.client import WeaviateWrapperClient
from .translation_tools import TranslationTools
from .utils import VietnameseTextChunker
from .review_agent import ReviewAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_translation_graph(config: Configuration):
    """
    Create the enhanced translation workflow graph with big chunk, small chunk, review, and retry.
    """
    translation_tools = TranslationTools(config)
    chunker = VietnameseTextChunker()
    review_agent = ReviewAgent(translation_tools.translation_llm)

    workflow = StateGraph(OverallState)

    # Add enhanced nodes
    workflow.add_node("big_chunk_input", lambda state: big_chunk_input_node(state, chunker, config))
    workflow.add_node("search_memory", lambda state: search_memory_node(state, translation_tools))
    workflow.add_node("small_chunk_input", lambda state: small_chunk_input_node(state, chunker, config))
    workflow.add_node("translate_small_chunk", lambda state: translate_small_chunk_node(state, translation_tools))
    workflow.add_node("review_chunk", lambda state: review_chunk_node(state, review_agent))
    workflow.add_node("retry_translation", lambda state: retry_translation_node(state, translation_tools))
    workflow.add_node("memory_update", lambda state: memory_update_node(state, translation_tools))
    workflow.add_node("recent_context_update", lambda state: recent_context_update_node(state, translation_tools))

    # Entry point
    workflow.set_entry_point("big_chunk_input")

    # Wrapper functions for conditional edges that extract routing decision and update state
    def retry_or_continue_wrapper(state: OverallState) -> str:
        routing_decision, updated_state = should_retry_or_continue(state)
        # Update the state with the changes from should_retry_or_continue
        state.update(updated_state)
        return routing_decision
    
    def memory_update_or_continue_wrapper(state: OverallState) -> str:
        routing_decision, updated_state = should_memory_update_or_continue(state)
        # Update the state with the changes from should_memory_update_or_continue
        state.update(updated_state)
        return routing_decision

    # Enhanced workflow edges
    workflow.add_edge("big_chunk_input", "search_memory")
    workflow.add_edge("search_memory", "small_chunk_input")
    workflow.add_edge("small_chunk_input", "translate_small_chunk")
    workflow.add_edge("translate_small_chunk", "review_chunk")
    workflow.add_conditional_edges(
        "review_chunk",
        retry_or_continue_wrapper,
        {
            "retry": "retry_translation",
            "continue": "recent_context_update"
        }
    )
    workflow.add_edge("retry_translation", "review_chunk")
    workflow.add_conditional_edges(
        "recent_context_update",
        memory_update_or_continue_wrapper,
        {
            "memory_update": "memory_update",
            "continue": "small_chunk_input",
            "next_big_chunk": "big_chunk_input",
            "end": END
        }
    )
    workflow.add_edge("memory_update", "recent_context_update")

    return workflow.compile()


def search_memory_node(state: OverallState, translation_tools: TranslationTools) -> OverallState:
    """
    Search Weaviate for relevant memory context for the current big chunk (not each small chunk).
    Store results in state['big_chunk_memory_context'] for use by all small chunks in this big chunk.
    """
    try:
        weaviate_client = WeaviateWrapperClient()
        big_chunk_idx = state.get('current_big_chunk_index', 0)
        if big_chunk_idx >= len(state['big_chunks']):
            logger.info("All big chunks processed.")
            return state
        current_big_chunk = state['big_chunks'][big_chunk_idx]
        # Generate search queries from the big chunk text using translation tools
        search_queries = translation_tools.generate_search_queries(current_big_chunk['big_chunk_text'])
        # Search for relevant nodes in Weaviate
        retrieved_nodes = []
        for query in search_queries:
            nodes = weaviate_client.search_nodes_by_text(query, limit=2)
            retrieved_nodes.extend(nodes)
        # Remove duplicates based on node UUID
        unique_nodes = {}
        for node in retrieved_nodes:
            if node.get('uuid') not in unique_nodes:
                unique_nodes[node.get('uuid')] = node
        retrieved_nodes = list(unique_nodes.values())
        # Store in state for all small chunks in this big chunk
        state['big_chunk_memory_context'] = retrieved_nodes
        state['memory_state']['retrieved_nodes'] = state['memory_state']['retrieved_nodes'] + retrieved_nodes
        state['memory_state']['search_queries'] = state['memory_state']['search_queries'] + search_queries
        logger.info(f"Memory search for big chunk {big_chunk_idx} completed: {len(retrieved_nodes)} nodes retrieved")
    except Exception as e:
        logger.error(f"Error in memory search: {str(e)}")
        state['big_chunk_memory_context'] = []
    return state


# def translate_chunk_node(state: OverallState, translation_tools: TranslationTools) -> OverallState:
#     """
#     Translate the current chunk using LLM with memory context.
    
#     Args:
#         state: Current state with chunk and memory context
        
#     Returns:
#         OverallState: Updated state with translated text
#     """
#     try:
#         # Use translation tools to translate the chunk
#         original_text = state['translation_state']['original_text']
#         memory_context = state['translation_state']['memory_context']
#         recent_context = state['memory_context']
        
#         translated_text = translation_tools.translate_chunk(
#             original_text, memory_context, recent_context
#         )
        
#         # Update translation state
#         state['translation_state']['translated_text'] = translated_text
#         state['translation_state']['processing_status'] = 'completed'
        
#         # Add the translated text to the accumulated list
#         logger.info(f"Before adding translated chunk: len={len(state['translated_chunks'])}")
#         state['translated_chunks'].append(translated_text)
#         logger.info(f"Translation completed, added chunk: {translated_text[:100]}...")
        
#     except Exception as e:
#         logger.error(f"Error in translation: {str(e)}")
#         state['translation_state']['error_message'] = f"Translation failed: {str(e)}"
#         state['translation_state']['processing_status'] = 'failed'
    
#     logger.info(f"translate_chunk_node final: len(translated_chunks)={len(state['translated_chunks'])}")
    
#     return state


def memory_update_node(state: OverallState, translation_tools: TranslationTools) -> OverallState:
    """
    Update Weaviate memory based on translation results.
    
    Args:
        state: Current state with translation results
        
    Returns:
        OverallState: Updated state with memory operations
    """
    try:
        logger.info(f"memory_update_node: len(translated_chunks)={len(state['translated_chunks'])}")
        weaviate_client = WeaviateWrapperClient()
        
        # Prepare context for memory update decision
        original_text = state['translation_state']['original_text']
        translated_text = state['translation_state']['translated_text']
        memory_context = state['translation_state']['memory_context']
        
        # Generate memory update decisions using translation tools
        if translated_text:
            memory_operations = translation_tools.generate_memory_operations(
                original_text, translated_text, memory_context
            )
        else:
            memory_operations = {"create_nodes": [], "update_nodes": []}
        
        # Execute memory operations
        created_nodes = []
        updated_nodes = []
        
        # Create new nodes
        for node_data in memory_operations.get('create_nodes', []):
            try:
                from weaviate_client.schema import KnowledgeNode
                node = KnowledgeNode(**node_data)
                node_id = weaviate_client.insert_knowledge_node(node)
                created_nodes.append({'id': str(node_id), 'data': node_data})
            except Exception as e:
                logger.error(f"Failed to create node: {str(e)}")
        
        # Update existing nodes
        for update_data in memory_operations.get('update_nodes', []):
            try:
                name = update_data['name']
                new_content = update_data['new_content']
                
                # Use the new update_node method
                success = weaviate_client.update_node(name, new_content)
                if success:
                    updated_nodes.append({'id': node_id, 'new_content': new_content})
                else:
                    logger.error(f"Failed to update node {node_id}")
            except Exception as e:
                logger.error(f"Failed to update node: {str(e)}")
        
        # Update memory state
        state['memory_state']['created_nodes'] = state['memory_state']['created_nodes'] + created_nodes
        state['memory_state']['updated_nodes'] = state['memory_state']['updated_nodes'] + updated_nodes
        
        # Log memory operation
        # Use translated_chunks length to determine which chunk was processed
        chunks_processed = len(state['translated_chunks'])
        memory_operation = {
            'operation_type': 'memory_update',
            'node_type': 'mixed',
            'query_or_content': f"Translation of chunk {chunks_processed - 1}",
            'result': {
                'created_count': len(created_nodes),
                'updated_count': len(updated_nodes)
            },
            'timestamp': str(datetime.now())
        }
        state['memory_state']['memory_operations'].append(memory_operation)
        
        logger.info(f"Memory update completed: {len(created_nodes)} created, {len(updated_nodes)} updated")
        
        # Log total objects in Weaviate after memory update
        try:
            total_objects = weaviate_client.count_objects()
            logger.info(f"Total objects in Weaviate collection: {total_objects}")
        except Exception as e:
            logger.warning(f"Could not count Weaviate objects: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in memory update: {str(e)}")
        # Don't fail the entire workflow for memory update errors
    
    return state


def recent_context_update_node(state: OverallState, translation_tools: TranslationTools) -> OverallState:
    """
    Update recent context for continuity across chunks.
    
    Args:
        state: Current state with translation results
        
    Returns:
        OverallState: Updated state with refreshed recent context
    """
    logger.debug(f"[recent_context_update_node] current_small_chunk_index={state.get('current_small_chunk_index')}, current_big_chunk_index={state.get('current_big_chunk_index')}, processing_complete={state.get('processing_complete')}")
    try:
        logger.info(f"recent_context_update_node: len(translated_chunks)={len(state['translated_chunks'])}")
        # Prepare context for summary
        recent_context = translation_tools._format_recent_context(state['memory_context'])
        current_translation = state['translation_state']['translated_text']
        
        # Generate context summary using translation tools
        if current_translation:
            context_summary = translation_tools.generate_context_summary(recent_context, current_translation)
        else:
            context_summary = "Không có bản dịch để tóm tắt"
        
        # Update recent context (keep last 5 items to avoid context overflow)
        # Use translated_chunks length to determine which chunk was processed
        chunks_processed = len(state['translated_chunks'])
        new_context_item = {
            'chunk_index': chunks_processed - 1,
            'summary': context_summary,
            'timestamp': str(datetime.now())
        }
        state['memory_context'] = state['memory_context'] + [new_context_item]
        
        # Trim context if too long
        if len(state['memory_context']) > 5:
            state['memory_context'] = state['memory_context'][-5:]
        
        logger.info("Recent context updated")
        
    except Exception as e:
        logger.error(f"Error in context update: {str(e)}")
        # Don't fail the entire workflow for context update errors
    
    return state


def big_chunk_input_node(state: OverallState, chunker: VietnameseTextChunker, config: Configuration) -> OverallState:
    """
    Split input text into big chunks and initialize their state if not already done.
    """
    if not state.get('big_chunks') or len(state['big_chunks']) == 0:
        from .utils import EnhancedChunkingConfig
        chunking_config = EnhancedChunkingConfig(
            big_chunk_size=state.get('big_chunk_size', 16000),
            small_chunk_size=state.get('small_chunk_size', 500)
        )
        big_chunks = chunker.chunk_text_into_big_chunks(state['input_text'], chunking_config)
        state['big_chunks'] = big_chunks
        state['total_big_chunks'] = len(big_chunks)
        state['current_big_chunk_index'] = 0
        logger.info(f"Created {len(big_chunks)} big chunks.")
    return state


def small_chunk_input_node(state: OverallState, chunker: VietnameseTextChunker, config: Configuration) -> OverallState:
    """
    Split the current big chunk into small chunks and initialize their state if not already done.
    """
    big_chunk_idx = state.get('current_big_chunk_index', 0)
    if big_chunk_idx >= len(state['big_chunks']):
        logger.info("All big chunks processed.")
        return state
    big_chunk = state['big_chunks'][big_chunk_idx]
    if not big_chunk.get('small_chunks') or len(big_chunk['small_chunks']) == 0:
        small_chunks = chunker.chunk_big_chunk_into_small_chunks(big_chunk)
        big_chunk['small_chunks'] = small_chunks
        state['small_chunks'] = state['small_chunks'] + small_chunks
        state['total_small_chunks'] = len(state['small_chunks'])
        state['current_small_chunk_index'] = 0
        logger.info(f"Created {len(small_chunks)} small chunks for big chunk {big_chunk_idx}.")
    return state


def translate_small_chunk_node(state: OverallState, translation_tools: TranslationTools) -> OverallState:
    """
    Translate the current small chunk using LLM with memory context from the current big chunk.
    """
    logger.debug(f"[translate_small_chunk_node] current_small_chunk_index={state.get('current_small_chunk_index')}, current_big_chunk_index={state.get('current_big_chunk_index')}, processing_complete={state.get('processing_complete')}")
    try:
        small_chunk_idx = state.get('current_small_chunk_index', 0)
        if small_chunk_idx >= len(state['small_chunks']):
            logger.info("All small chunks processed.")
            return state
        current_small_chunk = state['small_chunks'][small_chunk_idx]
        # Use memory context from big chunk memory search
        memory_context = state.get('big_chunk_memory_context', [])
        recent_context = state['memory_context']
        original_text = current_small_chunk['small_chunk_text']
        translated_text, quality, metrics = translation_tools.translate_small_chunk(
            original_text, memory_context, recent_context,
            current_small_chunk['position_in_big_chunk'], len(state['small_chunks'])
        )
        # Update translation state
        state['translation_state']['original_text'] = original_text
        state['translation_state']['translated_text'] = translated_text
        state['translation_state']['memory_context'] = memory_context
        state['translation_state']['translation_quality'] = str(quality.overall_score)
        state['translation_state']['processing_status'] = 'completed'
        # Update the small chunk with the translation
        current_small_chunk['translated_text'] = translated_text
        # Add the translated text to the accumulated list
        logger.info(f"Before adding translated chunk: len={len(state['translated_chunks'])}")
        state['translated_chunks'] = state['translated_chunks'] + [translated_text]
        logger.info(f"Small chunk translation completed: {len(translated_text)} characters, quality: {quality.overall_score:.2f}")
    except Exception as e:
        logger.error(f"Error in small chunk translation: {str(e)}")
        state['translation_state']['error_message'] = f"Translation failed: {str(e)}"
        state['translation_state']['processing_status'] = 'failed'
    return state


def review_chunk_node(state: OverallState, review_agent: ReviewAgent) -> OverallState:
    """
    Review the current small chunk translation and update review state.
    """
    logger.debug(f"[review_chunk_node] current_small_chunk_index={state.get('current_small_chunk_index')}, current_big_chunk_index={state.get('current_big_chunk_index')}, processing_complete={state.get('processing_complete')}")
    from .state import ReviewState
    small_chunk_idx = state.get('current_small_chunk_index', 0)
    if small_chunk_idx >= len(state['small_chunks']):
        logger.info("All small chunks processed.")
        return state
    small_chunk = state['small_chunks'][small_chunk_idx]
    review_state: ReviewState = {
        'chunk_id': small_chunk['small_chunk_id'],
        'original_text': small_chunk['small_chunk_text'],
        'translated_text': small_chunk['translated_text'] or "",
        'context': None,
        'rating': None,
        'feedback': None,
        'confidence': 0.0,
        'requires_revision': False,
        'review_timestamp': None,
        'reviewer_id': None
    }
    updated_review = review_agent.review_chunk(review_state)
    current_review_states = state.get('review_states', [])
    state['review_states'] = current_review_states + [updated_review]
    logger.info(f"Reviewed small chunk {small_chunk_idx}.")
    return state


def retry_translation_node(state: OverallState, translation_tools: TranslationTools) -> OverallState:
    """
    Retry translation for the current small chunk using feedback from review.
    """
    small_chunk_idx = state.get('current_small_chunk_index', 0)
    if small_chunk_idx >= len(state['small_chunks']):
        logger.info("All small chunks processed.")
        return state
    small_chunk = state['small_chunks'][small_chunk_idx]
    # Find the latest review for this chunk
    review = None
    for r in reversed(state.get('review_states', [])):
        if r['chunk_id'] == small_chunk['small_chunk_id']:
            review = r
            break
    feedback = (review['feedback'] if review and review['feedback'] else "")
    previous_translation = small_chunk['translated_text'] or ""
    translated_text, quality, metrics = translation_tools.retranslate_with_feedback(
        small_chunk['small_chunk_text'],
        previous_translation,
        feedback,
        small_chunk.get('memory_context', []),
        small_chunk.get('recent_context', [])
    )
    small_chunk['translated_text'] = translated_text
    small_chunk['is_processed'] = True
    small_chunk['processing_status'] = 'completed'
    logger.info(f"Retried translation for small chunk {small_chunk_idx}.")
    return state


def should_retry_or_continue(state: OverallState) -> Tuple[str, OverallState]:
    """
    Decide whether to retry translation or continue based on review.
    """
    logger.debug(f"[should_retry_or_continue] current_small_chunk_index={state.get('current_small_chunk_index')}, current_big_chunk_index={state.get('current_big_chunk_index')}, processing_complete={state.get('processing_complete')}")
    small_chunk_idx = state.get('current_small_chunk_index', 0)
    if small_chunk_idx >= len(state['small_chunks']):
        return "continue", state
    small_chunk = state['small_chunks'][small_chunk_idx]
    review = None
    for r in reversed(state.get('review_states', [])):
        if r['chunk_id'] == small_chunk['small_chunk_id']:
            review = r
            break
    if review and review.get('requires_revision', False):
        logger.info(f"Retry required for small chunk {small_chunk_idx}.")
        return "retry", state
    logger.info(f"Continue after review for small chunk {small_chunk_idx}.")
    return "continue", state


def should_memory_update_or_continue(state: OverallState) -> Tuple[str, OverallState]:
    """
    Decide whether to run memory_update (if last small chunk in big chunk), continue to next small chunk, next big chunk, or end.
    """
    logger.debug(f"[should_memory_update_or_continue] current_small_chunk_index={state.get('current_small_chunk_index')}, current_big_chunk_index={state.get('current_big_chunk_index')}, processing_complete={state.get('processing_complete')}")
    small_chunk_idx = state.get('current_small_chunk_index', 0)
    big_chunk_idx = state.get('current_big_chunk_index', 0)
    big_chunk = state['big_chunks'][big_chunk_idx] if big_chunk_idx < len(state['big_chunks']) else None
    is_last_small_chunk = False
    if big_chunk and big_chunk.get('small_chunks'):
        is_last_small_chunk = (small_chunk_idx + 1 == len(big_chunk['small_chunks']))
    if is_last_small_chunk:
        return "memory_update", state
    elif small_chunk_idx + 1 < state.get('total_small_chunks', 0):
        state['current_small_chunk_index'] = state['current_small_chunk_index'] + 1
        logger.info(f"Continuing to next small chunk: {state['current_small_chunk_index']}")
        return "continue", state
    elif state.get('current_big_chunk_index', 0) + 1 < state.get('total_big_chunks', 0):
        state['current_big_chunk_index'] = state['current_big_chunk_index'] + 1
        state['current_small_chunk_index'] = 0
        logger.info(f"Moving to next big chunk: {state['current_big_chunk_index']}")
        return "next_big_chunk", state
    else:
        state['processing_complete'] = True
        logger.info("All big and small chunks processed, ending workflow.")
        return "end", state

# Import datetime for timestamps
from datetime import datetime
