# VietPhrase Reader Assistant (VRA)

An AI-powered assistant specialized in converting *VietPhrase* (a Vietnamese text format) into fluent, high-quality Vietnamese suitable for casual reading and translation validation.

## What is VietPhrase?

VietPhrase is a Vietnamese text format commonly used by Vietnamese translators and readers. It consists of Vietnamese text segments that may be literal translations or rough drafts that need refinement for better readability.

The format typically looks like this:

```
Chương 1: Khởi đầu mới

Trương Tam là một lập trình viên trẻ, anh ta sống ở Bắc Kinh.

Hôm nay, anh ta sẽ tham gia một cuộc họp quan trọng.

Chủ đề của cuộc họp là về sự phát triển của trí tuệ nhân tạo.
```

## The Problem

While VietPhrase provides Vietnamese text, these scripts are often literal translations that are rarely fluent, often awkward, and difficult for general readers to enjoy. The transformation into readable Vietnamese remains manual and slow.

## The Solution

VRA automatically:
- References previous translation decisions (character names, glossary terms, recurring phrases)
- Updates memory with new findings
- Generates fluent Vietnamese output from rough Vietnamese input
- Maintains consistency across chapters using semantic memory

## VietPhrase Format Guidelines

When preparing text for VRA, follow this format:

1. **Contains only Vietnamese text**
2. **Each segment should be on separate lines**
3. **Empty lines can be used to separate paragraphs**
4. **The system will process this format to generate fluent Vietnamese text**
5. **No Chinese text is included in the input**

## Features

- **Intelligent Chunking**: Splits chapters into logical translation units
- **Semantic Memory**: Uses Weaviate to store and retrieve contextual knowledge
- **Consistent Translation**: Maintains character names, terminology, and style across chapters
- **Quality Enhancement**: Converts rough Vietnamese into fluent Vietnamese
- **Memory Management**: Automatically creates and updates glossary terms and character information

## Architecture

- **LangGraph**: Orchestrates the translation workflow
- **Google Gemini**: Powers the LLM-driven translation refinement
- **Weaviate**: Provides semantic memory storage with vector embeddings
- **Python**: Core implementation language

## Getting Started

### 1. Environment Setup

Set up environment variables:
   - `GOOGLE_API_KEY`: For Gemini API access
   - `WEAVIATE_URL`: Weaviate instance URL (default: http://localhost:8080)
   - `WEAVIATE_API_KEY`: Weaviate API key (optional for local setup)

### 2. Install Dependencies

```bash
# Create conda environment
conda env create -f environment.yml

# Activate the environment
conda activate dich_truyen_ai
```

### 3. Start Weaviate with Docker

```bash
# Start Weaviate using Docker Compose
docker-compose up -d

# Or start Weaviate directly with Docker
docker run -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
```

### 4. Prepare Your Data

Copy your chapter files to the appropriate directory:

```bash
# Create directory structure
mkdir -p raw_text/{your_story_name}

# Copy your chapter files (chap_1.txt, chap_2.txt, etc.) to the directory
cp your_chapters/* raw_text/{your_story_name}/
```

### 5. Run the Demo

#### Step 1: Chunk the chapters
```bash
# Edit chunk_chapter.py to point to your data folder
# Update line 108: store_chunks_to_input_folder("raw_text/{your_story_name}", 'raw_text/demo')
python chunk_chapter.py
```

#### Step 2: Run translation demo
```bash
python vietphrase_translation_demo.py
```

The translation results will be saved in the `translated_chapters/` directory.

### Quick Demo Workflow

1. **Setup environment**: `conda env create -f environment.yml && conda activate dich_truyen_ai`
2. **Start Weaviate**: `docker run -p 8080:8080 semitechnologies/weaviate:latest`
3. **Prepare data**: Copy your `chap_*.txt` files to `raw_text/{story_name}/`
4. **Chunk chapters**: Run `python chunk_chapter.py` (update the path in the script)
5. **Translate**: Run `python vietphrase_translation_demo.py`
6. **View results**: Check `translated_chapters/` for the translated output

## License

This project is licensed under the MIT License - see the LICENSE file for details. 