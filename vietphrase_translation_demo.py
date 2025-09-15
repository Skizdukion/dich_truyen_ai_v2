#!/usr/bin/env python3
"""
VietPhrase Reader Assistant - Translation Demo
Task 5.0: Interact with agent to translate chapters 1 and 2
"""

import sys
sys.path.append("src")

from dotenv import load_dotenv
load_dotenv()

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


def _is_quota_error(err: Exception) -> bool:
    """Heuristically detect quota/rate-limit errors from LLM/HTTP layers."""
    msg = str(err)
    msg_lower = msg.lower()
    quota_terms = [
        "quota",
        "rate limit",
        "resource exhausted",
        "429",
        "exceeded",
        "too many requests",
    ]
    return any(term in msg_lower for term in quota_terms)


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


def load_demo_chapter_chunks(demo_dir: str, chapter_index: int):
    """Load already-split chunk files for a chapter from demo directory, sorted numerically."""
    import os
    import re

    pattern = re.compile(rf"^chap_{chapter_index}_chunk_(\d+)\.txt$")

    try:
        all_files = os.listdir(demo_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"Demo directory not found: {demo_dir}")

    matched = []
    for filename in all_files:
        m = pattern.match(filename)
        if m:
            chunk_num = int(m.group(1))
            matched.append((chunk_num, filename))

    matched.sort(key=lambda x: x[0])

    chunks = []
    for _, filename in matched:
        path = os.path.join(demo_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            chunks.append({"chunk_text": content})

    print(f"✓ Loaded {len(chunks)} chunks for chap_{chapter_index} from {demo_dir}")
    return chunks


def run_demo_directory(demo_dir: str, graph):
    """Run translation over the whole demo directory in numeric order by chapter and chunk.

    - Supports both pre-split files: chap_{n}_chunk_{m}.txt
      and whole chapter files:      chap_{n}.txt
    - Maintains memory context across chapters.
    """
    import os
    import re

    # Gather and sort all files numerically by inferred chapter and chunk
    files = []
    try:
        for filename in os.listdir(demo_dir):
            m_chunk = re.match(r"^chap_(\d+)_chunk_(\d+)\.txt$", filename)
            m_whole = re.match(r"^chap_(\d+)\.txt$", filename)
            if m_chunk:
                ch = int(m_chunk.group(1))
                ck = int(m_chunk.group(2))
                files.append((ch, ck, filename))
            elif m_whole:
                ch = int(m_whole.group(1))
                files.append((ch, 0, filename))
    except FileNotFoundError:
        raise FileNotFoundError(f"Demo directory not found: {demo_dir}")

    files.sort(key=lambda t: (t[0], t[1]))

    memory_context = None
    for ch, ck, filename in files:
        chapter_name = f"CHAPTER {ch}" if ck == 0 else f"CHAPTER {ch} - PART {ck}"
        input_path = os.path.join(demo_dir, filename)
        print(f"\n=== PROCESSING {chapter_name} ({filename}) ===")

        # Build per-file trace dir
        base_name = os.path.splitext(filename)[0]
        trace_dir = os.path.join("traces", base_name)

        # Resume: skip if this chapter output already exists and is non-empty
        if is_output_present(chapter_name):
            print(f"↷ Skipping {chapter_name} (already translated)")
            continue

        # Prepare chunks: treat each input file independently
        if re.match(r"^chap_\d+_chunk_\d+\.txt$", filename):
            # Already a chunk file: keep as single chunk
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = [{"chunk_text": content}]
        else:
            # Whole chapter: chunk on the fly
            chunks = load_and_chunk_chapter(input_path)

        # Translate with trace directory
        try:
            result = translate_chapter(chapter_name, chunks, graph, memory_context, trace_dir=trace_dir)
            export_translation(chapter_name, result["translated_chunks"])
        except Exception as e:
            if _is_quota_error(e):
                print(f"✗ Quota/rate limit detected: {e}")
                print("Exiting immediately. Re-run later to resume.")
                import sys as _sys
                _sys.exit(2)
            else:
                # Stop on other errors so a rerun will continue from here
                print(f"✗ Error processing {chapter_name}: {str(e)}")
                print("You can rerun the script to continue from this point.")
                break

        # Carry memory forward
        memory_context = result.get("memory_context")


def translate_chapter(chapter_name, chunks, graph, memory_context=None, trace_dir: str = None):
    """Translate a chapter using the agent."""
    print(f"=== TRANSLATING {chapter_name} ===")

    # Normalize chunks to ensure required fields exist
    normalized_chunks = []
    for ch in chunks:
        chunk_obj = {
            "chunk_text": ch.get("chunk_text", ""),
            "is_processed": ch.get("is_processed", False),
            "translation_attempts": ch.get("translation_attempts", 0),
        }
        normalized_chunks.append(chunk_obj)

    # Initialize state
    state = {
        "messages": [],
        "input_text": " ".join([chunk["chunk_text"] for chunk in normalized_chunks]),
        "chunks": normalized_chunks,
        "current_chunk_index": 0,
        "total_chunks": len(normalized_chunks),
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
        "trace_dir": trace_dir,
    }

    print(f"Starting translation of {chapter_name}...")
    print(f"Initial state - chunks: {len(chunks)}, translated_chunks: {len(state['translated_chunks'])}")
    
    result_state = graph.invoke(state, config={"recursion_limit": 50})

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


def get_output_filename(chapter_name: str, output_dir: str = "translated_chapters") -> str:
    import os
    return f"{output_dir}/{chapter_name.lower().replace(' ', '_')}_translated.txt"


def is_output_present(chapter_name: str, output_dir: str = "translated_chapters") -> bool:
    import os
    path = get_output_filename(chapter_name, output_dir)
    return os.path.exists(path) and os.path.getsize(path) > 0


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

    # Continuation mechanism: Only reset Weaviate if no outputs yet
    import os
    out_dir = "translated_chapters"
    os.makedirs(out_dir, exist_ok=True)
    has_outputs = any(
        f.endswith("_translated.txt") and os.path.getsize(os.path.join(out_dir, f)) > 0
        for f in os.listdir(out_dir)
    )
    if not has_outputs:
        clear_all_data(weaviate_client)
    else:
        print("Detected existing translations; skipping Weaviate reset and resuming...")

    # Run the whole demo directory using numerically sorted chapters and chunks
    demo_dir = "raw_text/demo"
    run_demo_directory(demo_dir, graph)


if __name__ == "__main__":
    main()
