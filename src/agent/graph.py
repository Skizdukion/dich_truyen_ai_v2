"""
Translation workflow graph for VietPhrase Reader Assistant.
Implements a memory-aware translation pipeline using LangGraph.
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
import json
import os
import logging

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_token_usage(usage_meta: Any) -> Dict[str, int]:
    """Best-effort extraction of input/output token counts from provider metadata."""
    def get_int(d: Any, *keys: str) -> int:
        cur = d
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return 0
        return int(cur) if isinstance(cur, (int, float)) else 0

    if usage_meta is None:
        return {"input_tokens": 0, "output_tokens": 0}

    # Common patterns
    # Gemini via langchain_google_genai might expose usage_metadata with fields
    if isinstance(usage_meta, dict):
        est_in = int(usage_meta.get('estimated_input_tokens', 0)) if isinstance(usage_meta.get('estimated_input_tokens'), (int, float)) else 0
        est_out = int(usage_meta.get('estimated_output_tokens', 0)) if isinstance(usage_meta.get('estimated_output_tokens'), (int, float)) else 0
    else:
        est_in = 0
        est_out = 0

    input_tokens = (
        get_int(usage_meta, "token_count", "prompt_tokens")
        or get_int(usage_meta, "token_count", "input_tokens")
        or get_int(usage_meta, "usage", "prompt_tokens")
        or get_int(usage_meta, "usage", "input_tokens")
        or get_int(usage_meta, "prompt_tokens")
        or get_int(usage_meta, "input_tokens")
        or est_in
    )
    output_tokens = (
        get_int(usage_meta, "token_count", "candidates_tokens")
        or get_int(usage_meta, "token_count", "output_tokens")
        or get_int(usage_meta, "usage", "completion_tokens")
        or get_int(usage_meta, "usage", "output_tokens")
        or get_int(usage_meta, "completion_tokens")
        or get_int(usage_meta, "output_tokens")
        or est_out
    )

    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _update_summary_tokens(summary_path: str, step_name: str, usage_meta: Any):
    try:
        usage = _extract_token_usage(usage_meta)
        summary = {}
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        # Per-step
        summary.setdefault('token_usage', {})[step_name] = usage
        # Totals
        totals = summary.setdefault('token_usage', {}).setdefault('total', {"input_tokens": 0, "output_tokens": 0})
        totals['input_tokens'] = int(totals.get('input_tokens', 0)) + usage['input_tokens']
        totals['output_tokens'] = int(totals.get('output_tokens', 0)) + usage['output_tokens']
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def create_translation_graph(config: Configuration):
    """
    Create the translation workflow graph with memory context.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        StateGraph: The translation workflow graph
    """
    
    # Initialize Weaviate client and translation tools
    # weaviate_client = WeaviateWrapperClient()
    translation_tools = TranslationTools(config)
    
    # Create the graph
    workflow = StateGraph(OverallState)
    
    # Add nodes to the graph with translation tools
    workflow.add_node("chunk_input", lambda state: chunk_input_node(state))
    workflow.add_node("search_memory", lambda state: search_memory_node(state, translation_tools))
    workflow.add_node("translate_chunk", lambda state: translate_chunk_node(state, translation_tools))
    workflow.add_node("memory_update", lambda state: memory_update_node(state, translation_tools))
    workflow.add_node("recent_context_update", lambda state: recent_context_update_node(state, translation_tools))
    
    # Define the workflow edges
    workflow.set_entry_point("chunk_input")
    
    # Main translation flow
    workflow.add_edge("chunk_input", "search_memory")
    workflow.add_edge("search_memory", "translate_chunk")
    workflow.add_edge("translate_chunk", "memory_update")
    workflow.add_edge("memory_update", "recent_context_update")
    
    # Conditional edge: continue to next chunk or end
    workflow.add_conditional_edges(
        "recent_context_update",
        should_continue_to_next_chunk,
        {
            "continue": "chunk_input",
            "end": END
        }
    )
    
    return workflow.compile()


def chunk_input_node(state: OverallState) -> OverallState:
    """
    Process the next chunk in the translation pipeline.
    
    Args:
        state: Current state containing chunks and processing info
        
    Returns:
        OverallState: Updated state with current chunk information
    """
    # Use translated_chunks length to determine which chunk to process
    chunks_processed = len(state['translated_chunks'])
    current_chunk_index = chunks_processed
    
    logger.info(f"Processing chunk {current_chunk_index + 1}/{state['total_chunks']} (chunks_processed={chunks_processed})")
    
    # Safety check: ensure we don't exceed the chunks array bounds
    if current_chunk_index >= len(state['chunks']):
        logger.error(f"Chunk index {current_chunk_index} exceeds available chunks {len(state['chunks'])}")
        state['processing_complete'] = True
        return state
    
    # Get the current chunk
    current_chunk = state['chunks'][current_chunk_index]
    
    # Update translation state
    state['translation_state'] = {
        'chunk_id': f"chunk_{current_chunk_index}",
        'original_text': current_chunk['chunk_text'],
        'translated_text': None,
        'memory_context': [],
        'translation_quality': None,
        'processing_status': 'processing',
        'error_message': None
    }
    
    # Mark chunk as being processed
    state['chunks'][current_chunk_index]['is_processed'] = True
    state['chunks'][current_chunk_index]['translation_attempts'] += 1
    
    logger.info(f"Chunk input processed: {len(current_chunk['chunk_text'])} characters")
    return state


def search_memory_node(state: OverallState, translation_tools: TranslationTools) -> OverallState:
    """
    Search Weaviate for relevant memory context for the current chunk.
    
    Args:
        state: Current state with chunk information
        
    Returns:
        OverallState: Updated state with retrieved memory context
    """
    try:
        weaviate_client = WeaviateWrapperClient()
        # Use translated_chunks length to determine which chunk to process
        chunks_processed = len(state['translated_chunks'])
        current_chunk_index = chunks_processed
        
        # Safety check: ensure we don't exceed the chunks array bounds
        if current_chunk_index >= len(state['chunks']):
            logger.error(f"Chunk index {current_chunk_index} exceeds available chunks {len(state['chunks'])}")
            state['translation_state']['processing_status'] = 'failed'
            return state
            
        current_chunk = state['chunks'][current_chunk_index]
        
        # Generate search queries from the chunk text using translation tools
        search_queries = translation_tools.generate_search_queries(current_chunk['chunk_text'])
        
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
        
        # Update memory state
        state['memory_state']['retrieved_nodes'].extend(retrieved_nodes)
        state['memory_state']['search_queries'].extend(search_queries)
        
        # Update translation state with memory context
        state['translation_state']['memory_context'] = retrieved_nodes
        
        logger.info(f"Memory search completed: {len(retrieved_nodes)} nodes retrieved")

        # Trace to file (size-limited)
        try:
            trace_dir = state.get('trace_dir')
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                # Limit nodes for trace to avoid huge JSON
                max_nodes = 20
                small_nodes = retrieved_nodes[:max_nodes]
                for n in small_nodes:
                    if isinstance(n.get('content'), str) and len(n['content']) > 800:
                        n['content'] = n['content'][:800] + "..."
                trace = {
                    'request': {
                        'chunk_index': current_chunk_index,
                        'chunk_text_preview': current_chunk['chunk_text'][:500],
                        'search_queries': search_queries,
                    },
                    'response': {
                        'retrieved_count': len(retrieved_nodes),
                        'retrieved_nodes_preview': small_nodes,
                    },
                }
                with open(os.path.join(trace_dir, 'search_memory.json'), 'w', encoding='utf-8') as f:
                    json.dump(trace, f, ensure_ascii=False, indent=2)
                # Update summary.json with usage and counts
                summary_path = os.path.join(trace_dir, 'summary.json')
                summary = {}
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                # We cannot directly access usage here; leave placeholder counters
                summary['search_memory'] = {
                    'retrieved_count': len(retrieved_nodes),
                }
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                # Token usage from translation_tools
                try:
                    usage_meta = getattr(translation_tools, 'last_search_usage', None)
                    _update_summary_tokens(summary_path, 'search_memory', usage_meta)
                except Exception:
                    pass
        except Exception as _:
            pass
        
    except Exception as e:
        logger.error(f"Error in memory search: {str(e)}")
        state['translation_state']['error_message'] = f"Memory search failed: {str(e)}"
        state['translation_state']['processing_status'] = 'failed'
    
    return state


def translate_chunk_node(state: OverallState, translation_tools: TranslationTools) -> OverallState:
    """
    Translate the current chunk using LLM with memory context.
    
    Args:
        state: Current state with chunk and memory context
        
    Returns:
        OverallState: Updated state with translated text
    """
    try:
        # Use translation tools to translate the chunk
        original_text = state['translation_state']['original_text']
        memory_context = state['translation_state']['memory_context']
        recent_context = state['memory_context']
        
        translated_text = translation_tools.translate_chunk(
            original_text, memory_context, recent_context
        )
        
        # Update translation state
        state['translation_state']['translated_text'] = translated_text
        state['translation_state']['processing_status'] = 'completed'
        
        # Add the translated text to the accumulated list
        logger.info(f"Before adding translated chunk: len={len(state['translated_chunks'])}")
        state['translated_chunks'].append(translated_text)
        logger.info(f"Translation completed, added chunk: {translated_text[:100]}...")

        # Trace to file (size-limited)
        try:
            trace_dir = state.get('trace_dir')
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                trace = {
                    'request': {
                        'original_text_preview': original_text[:800],
                        'memory_context_preview': memory_context[:10],
                        'recent_context_preview': recent_context[-3:],
                    },
                    'response': {
                        'translated_text_preview': translated_text[:2000],
                    },
                }
                with open(os.path.join(trace_dir, 'translate_chunk.json'), 'w', encoding='utf-8') as f:
                    json.dump(trace, f, ensure_ascii=False, indent=2)
                # Update summary.json - we cannot read usage metadata here reliably
                summary_path = os.path.join(trace_dir, 'summary.json')
                summary = {}
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                summary['translate_chunk'] = {
                    'translated_chars': len(translated_text) if translated_text else 0,
                }
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                try:
                    usage_meta = getattr(translation_tools, 'last_translation_usage', None)
                    _update_summary_tokens(summary_path, 'translate_chunk', usage_meta)
                except Exception:
                    pass
        except Exception as _:
            pass
        
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        state['translation_state']['error_message'] = f"Translation failed: {str(e)}"
        state['translation_state']['processing_status'] = 'failed'
    
    logger.info(f"translate_chunk_node final: len(translated_chunks)={len(state['translated_chunks'])}")
    
    return state


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
        update_failures = []
        
        # Create new nodes
        for node_data in memory_operations.get('create_nodes', []):
            try:
                from weaviate_client.schema import KnowledgeNode
                node = KnowledgeNode(**node_data)
                node_id = weaviate_client.insert_knowledge_node(node)
                created_nodes.append({'id': str(node_id), 'name': node_data.get('name'), 'data': node_data})
            except Exception as e:
                logger.error(f"Failed to create node: {str(e)}")
        
        # Update existing nodes
        requested_updates = memory_operations.get('update_nodes', [])
        for update_data in requested_updates:
            try:
                name = update_data.get('name')
                # Normalize payload
                payload = update_data.get('new_content', update_data)
                if not name:
                    update_failures.append({'reason': 'missing_name', 'data': update_data})
                    continue
                # Try by name
                success, updated_props = weaviate_client.update_node(name, payload)
                if success:
                    updated_nodes.append({'name': name, 'updated_properties': updated_props})
                else:
                    # Fallback: if a node with same name was just created, try by its id
                    fallback_id = None
                    for cn in created_nodes:
                        if cn.get('name') == name:
                            fallback_id = cn.get('id')
                            break
                    if fallback_id:
                        success_by_id, updated_props_by_id = weaviate_client.update_node_by_id(fallback_id, payload)
                        if success_by_id:
                            updated_nodes.append({'name': name, 'updated_properties': updated_props_by_id, 'via': 'by_id'})
                        else:
                            update_failures.append({'reason': 'not_found_or_update_failed', 'name': name})
                    else:
                        update_failures.append({'reason': 'not_found_or_update_failed', 'name': name})

            except Exception as e:
                update_failures.append({'reason': str(e), 'data': update_data})
        
        # Update memory state
        state['memory_state']['created_nodes'].extend(created_nodes)
        state['memory_state']['updated_nodes'].extend(updated_nodes)
        
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

        # Trace to file (size-limited)
        try:
            trace_dir = state.get('trace_dir')
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                trace = {
                    'request': {
                        'original_text_preview': original_text[:800] if original_text else None,
                        'translated_text_preview': translated_text[:800] if translated_text else None,
                        'memory_context_preview': memory_context[:10],
                        'requested_updates_preview': requested_updates[:10],
                    },
                    'response': {
                        'created_nodes_preview': created_nodes[:10],
                        'updated_nodes_preview': updated_nodes[:10],
                        'update_failures_preview': update_failures[:10],
                    },
                }
                with open(os.path.join(trace_dir, 'memory_update.json'), 'w', encoding='utf-8') as f:
                    json.dump(trace, f, ensure_ascii=False, indent=2)
                # Update summary.json
                summary_path = os.path.join(trace_dir, 'summary.json')
                summary = {}
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                summary['memory_update'] = {
                    'created_count': len(created_nodes),
                    'updated_count': len(updated_nodes),
                    'requested_updates_count': len(requested_updates),
                    'update_failures_count': len(update_failures),
                }
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                try:
                    usage_meta = getattr(translation_tools, 'last_memory_update_usage', None)
                    _update_summary_tokens(summary_path, 'memory_update', usage_meta)
                except Exception:
                    pass
        except Exception as _:
            pass
        
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
        state['memory_context'].append({
            'chunk_index': chunks_processed - 1,
            'summary': context_summary,
            'timestamp': str(datetime.now())
        })
        
        # Trim context if too long
        if len(state['memory_context']) > 5:
            state['memory_context'] = state['memory_context'][-5:]
        
        logger.info("Recent context updated")

        # Trace to file (size-limited)
        try:
            trace_dir = state.get('trace_dir')
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                trace = {
                    'request': {
                        'recent_context_before_preview': recent_context[-3:],
                        'current_translation_preview': current_translation[:800] if current_translation else None,
                    },
                    'response': {
                        'context_summary': context_summary,
                        'recent_context_after_preview': state['memory_context'][-3:],
                    },
                }
                with open(os.path.join(trace_dir, 'recent_context_update.json'), 'w', encoding='utf-8') as f:
                    json.dump(trace, f, ensure_ascii=False, indent=2)
        except Exception as _:
            pass

        # Update per-input summary.json
        try:
            if trace_dir:
                summary_path = os.path.join(trace_dir, 'summary.json')
                summary = {}
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                summary['recent_context_update'] = {
                    'has_translation': bool(current_translation),
                    'recent_context_size': len(state['memory_context']),
                }
                try:
                    usage_meta = getattr(translation_tools, 'last_context_summary_usage', None)
                    _update_summary_tokens(summary_path, 'recent_context_update', usage_meta)
                except Exception:
                    pass
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        
    except Exception as e:
        logger.error(f"Error in context update: {str(e)}")
        # Don't fail the entire workflow for context update errors
    
    return state


def should_continue_to_next_chunk(state: OverallState) -> str:
    """
    Determine if the workflow should continue to the next chunk or end.
    
    Args:
        state: Current state with processing information
        
    Returns:
        str: "continue" or "end"
    """
    # Use translated_chunks length to determine progress instead of current_chunk_index
    chunks_processed = len(state['translated_chunks'])
    total_chunks = state['total_chunks']
    
    logger.info(f"Checking continuation: chunks_processed={chunks_processed}, total_chunks={total_chunks}")
    
    # Add safety check to prevent infinite loops
    if chunks_processed >= total_chunks:
        # All chunks processed
        state['processing_complete'] = True
        logger.info("All chunks processed, ending workflow")
        return "end"
    elif chunks_processed < 0:
        # Safety check for negative values
        logger.warning(f"Invalid chunks_processed value: {chunks_processed}, ending workflow")
        state['processing_complete'] = True
        return "end"
    else:
        # Update current_chunk_index to match the next chunk to process
        state['current_chunk_index'] = chunks_processed
        logger.info(f"Continuing to next chunk: {state['current_chunk_index']}")
        return "continue"


# Import datetime for timestamps
from datetime import datetime
