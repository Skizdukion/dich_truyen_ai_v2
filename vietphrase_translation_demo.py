#!/usr/bin/env python3
"""
VietPhrase Reader Assistant - Translation Demo
Task 5.0: Interact with agent to translate chapters 1 and 2
"""

import sys
import os

sys.path.append("src")

from agent.graph import create_translation_graph
from agent.configuration import Configuration
from agent.state import OverallState
from agent.utils import ChunkingConfig, chunk_vietnamese_text
from weaviate_client.client import WeaviateWrapperClient


def clear_all_data(wrapper: WeaviateWrapperClient):
    """Clear all data from Weaviate and reset memory context."""
    print("=== CLEARING ALL DATA ===")

    # Clear Weaviate node collection
    print("Clearing Weaviate node collection...")
    try:
        wrapper.delete_all_nodes()
        print("✓ Weaviate node collection cleared successfully!")
    except Exception as e:
        print(f"✗ Error clearing Weaviate: {str(e)}")

    print("All data cleared!\n")


def load_and_chunk_chapter(chapter_path):
    """Load and chunk a chapter file."""
    print(f"Loading {chapter_path}...")
    with open(chapter_path, "r", encoding="utf-8") as f:
        chapter_text = f.read()

    # Chunk the text
    chunking_config = ChunkingConfig()
    chunks = chunk_vietnamese_text(chapter_text, chunking_config)

    print(f"✓ Chapter loaded: {len(chunks)} chunks")
    print(f"✓ Original text length: {len(chapter_text)} characters\n")

    return chunks


def translate_chapter(chapter_name, chunks, graph, memory_context=None):
    """Translate a chapter using the agent."""
    print(f"=== TRANSLATING {chapter_name} ===")

    # Initialize state
    state = {
        "messages": [],
        "input_text": " ".join([chunk["chunk_text"] for chunk in chunks]),
        "chunks": chunks,
        "current_chunk_index": 0,
        "total_chunks": len(chunks),
        "translated_chunks": [],
        "memory_context": memory_context or [],
        "translation_state": {},
        "memory_state": {
            "retrieved_nodes": [],
            "search_queries": [],
            "created_nodes": [],
            "updated_nodes": [],
            "memory_operations": [],
        },
        "chunk_size": 1000,
        "processing_complete": False,
        "failed_chunks": [],
        "retry_count": 0,
    }

    print(f"Starting translation of {chapter_name}...")
    print(f"Initial state - chunks: {len(chunks)}, translated_chunks: {len(state['translated_chunks'])}")
    
    result_state = graph.invoke(state)

    print(f"✓ Translation completed!")
    print(f"✓ Translated chunks: {len(result_state['translated_chunks'])}")
    print(f"✓ Total chunks processed: {result_state['total_chunks']}")
    print(f"✓ Current chunk index: {result_state['current_chunk_index']}")
    print(
        f"✓ Memory nodes retrieved: {len(result_state['memory_state']['retrieved_nodes'])}\n"
    )

    return result_state


def export_translation(
    chapter_name, translated_chunks, output_dir="translated_chapters"
):
    """Export translation results to a file."""
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    filename = f"{output_dir}/{chapter_name.lower().replace(' ', '_')}_translated.txt"

    # Write all translated chunks to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"=== {chapter_name} TRANSLATION ===\n\n")

        for i, translated_chunk in enumerate(translated_chunks):
            f.write(f"Chunk {i+1}:\n")
            f.write("-" * 40 + "\n")
            f.write(translated_chunk + "\n")
            f.write("-" * 40 + "\n\n")

        f.write(f"Total translated chunks: {len(translated_chunks)}\n")

    print(f"✓ Translation exported to: {filename}")
    print(f"✓ Total translated chunks: {len(translated_chunks)}\n")


def main():
    """Main function to run the translation demo."""
    print("VietPhrase Reader Assistant - Translation Demo")
    print("=" * 50)

    # Initialize configuration and graph
    print("Initializing agent...")
    config = Configuration()
    graph = create_translation_graph(config)
    weaviate_client = WeaviateWrapperClient()
    print("✓ Agent initialized successfully!\n")

    # Clear all data first
    clear_all_data(weaviate_client)

    # Translate Chapter 1
    chap1_chunks = load_and_chunk_chapter("raw_text/demo/chap_1.txt")
    result_ch1 = translate_chapter("CHAPTER 1", chap1_chunks, graph)
    export_translation("CHAPTER 1", result_ch1["translated_chunks"])

    # Translate Chapter 2 (with memory context from Chapter 1)
    chap2_chunks = load_and_chunk_chapter("raw_text/demo/chap_2.txt")
    result_ch2 = translate_chapter(
        "CHAPTER 2", chap2_chunks, graph, memory_context=result_ch1["memory_context"]
    )
    export_translation("CHAPTER 2", result_ch2["translated_chunks"])

    print("=== TRANSLATION DEMO COMPLETED ===")
    print("✓ Chapter 1 translated with fresh memory")
    print("✓ Chapter 2 translated with context from Chapter 1")
    print("✓ All data was cleared before starting")


if __name__ == "__main__":
    main()
