import os
from pathlib import Path
from src.agent.utils import ChunkingConfig, chunk_vietnamese_text

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

def store_chunks_to_input_folder(original_file_path, input_folder="input"):
    """
    Store all chunks into an input folder with naming convention: {original_file}_chunk_{i}.txt
    
    Args:
        chunks: List of text chunks
        original_file_path: Path to the original file
        input_folder: Target folder to store chunks (default: "input")
    """
    # Create input folder if it doesn't exist
    Path(input_folder).mkdir(parents=True, exist_ok=True)
    
    # Extract original filename without extension
    original_filename = Path(original_file_path).stem
    
    chunks = load_and_chunk_chapter(original_file_path)
    
    # Store each chunk
    for i, chunk in enumerate(chunks, 1):
        chunk_filename = f"{original_filename}_chunk_{i}.txt"
        chunk_path = Path(input_folder) / chunk_filename
        
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk["chunk_text"])
        
        print(f"✓ Saved chunk {i}: {chunk_path}")
    
    print(f"✓ All {len(chunks)} chunks saved to {input_folder}/ folder")

store_chunks_to_input_folder("raw_text/thien_chi_ha/chap_2.txt", 'raw_text/demo')
