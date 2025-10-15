import os
import re
from pathlib import Path
from src.agent.utils import ChunkingConfig, chunk_vietnamese_text

def count_words(text):
    """Count words in Vietnamese text."""
    # Remove extra whitespace and split by spaces
    words = re.findall(r'\S+', text)
    return len(words)

def load_and_chunk_chapter(chapter_path):
    """Load and chunk a chapter file."""
    print(f"Loading {chapter_path}...")
    with open(chapter_path, "r", encoding="utf-8") as f:
        chapter_text = f.read()

    # Count words in this chapter
    word_count = count_words(chapter_text)
    
    # Chunk the text
    chunking_config = ChunkingConfig()
    chunks = chunk_vietnamese_text(chapter_text, chunking_config)

    print(f"✓ Chapter loaded: {len(chunks)} chunks")
    print(f"✓ Original text length: {len(chapter_text)} characters")
    print(f"✓ Word count: {word_count} words\n")

    return chunks, word_count

def get_chapter_files(folder_path):
    """Get all chapter files sorted by chapter number."""
    folder = Path(folder_path)
    chapter_files = []
    
    # Find all files matching chap_*.txt pattern
    for file_path in folder.glob("chap_*.txt"):
        # Extract chapter number from filename
        match = re.search(r'chap_(\d+)\.txt', file_path.name)
        if match:
            chapter_num = int(match.group(1))
            chapter_files.append((chapter_num, file_path))
    
    # Sort by chapter number
    chapter_files.sort(key=lambda x: x[0])
    return [file_path for _, file_path in chapter_files]

def store_chunks_to_input_folder(folder_path, input_folder="input"):
    """
    Store all chunks from all chapters into an input folder with naming convention: {original_file}_chunk_{i}.txt
    
    Args:
        folder_path: Path to the folder containing chapter files
        input_folder: Target folder to store chunks (default: "input")
    """
    # Create input folder if it doesn't exist
    Path(input_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all chapter files sorted by chapter number
    chapter_files = get_chapter_files(folder_path)
    
    if not chapter_files:
        print(f"No chapter files found in {folder_path}")
        return
    
    print(f"Found {len(chapter_files)} chapter files:")
    for file_path in chapter_files:
        print(f"  - {file_path.name}")
    print()
    
    total_chunks = 0
    total_words = 0
    
    # Process each chapter
    for chapter_path in chapter_files:
        print(f"Processing {chapter_path.name}...")
        
        # Load and chunk the chapter
        chunks, word_count = load_and_chunk_chapter(chapter_path)
        
        # Extract original filename without extension
        original_filename = Path(chapter_path).stem
        
        # Store each chunk
        for i, chunk in enumerate(chunks, 1):
            chunk_filename = f"{original_filename}_chunk_{i}.txt"
            chunk_path = Path(input_folder) / chunk_filename
            
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk["chunk_text"])
            
            print(f"✓ Saved chunk {i}: {chunk_path}")
        
        total_chunks += len(chunks)
        total_words += word_count
        
        print(f"✓ Chapter {original_filename}: {len(chunks)} chunks, {word_count} words")
        print()
    
    print("=" * 50)
    print(f"SUMMARY:")
    print(f"✓ Total chapters processed: {len(chapter_files)}")
    print(f"✓ Total chunks created: {total_chunks}")
    print(f"✓ Total words across all chapters: {total_words}")
    print(f"✓ All chunks saved to {input_folder}/ folder")

# Process all chapters in the thien_chi_ha folder
store_chunks_to_input_folder("raw_text/thien_chi_ha", 'raw_text/demo')
