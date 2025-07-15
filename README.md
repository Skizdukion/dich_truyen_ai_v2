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

### Example Format

```
Chương 1: Khởi đầu mới

Trương Tam là một lập trình viên trẻ, anh ta sống ở Bắc Kinh.

Anh ta năm nay hai mươi lăm tuổi, làm việc tại một công ty công nghệ.
```

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

1. Set up environment variables:
   - `GOOGLE_API_KEY`: For Gemini API access
   - `WEAVIATE_URL`: Weaviate instance URL
   - `WEAVIATE_API_KEY`: Weaviate API key

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest src/tests/
   ```

## Example Usage

See `examples/vietphrase_format_example.txt` for a complete example of the VietPhrase format.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 