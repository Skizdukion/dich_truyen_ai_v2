"""
Translation workflow graph for VietPhrase Reader Assistant.
Implements a memory-aware translation pipeline using LangGraph.
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
import json
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
