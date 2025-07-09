# VietPhrase Reader Assistant - Implementation Tasks

## Relevant Files

- `src/agent/state.py` - Contains the current state management for web research, needs to be extended for translation state
- `src/agent/graph.py` - Contains the current LangGraph workflow, needs to be refactored for translation pipeline
- `src/agent/tools_and_schemas.py` - Contains current schemas, needs new schemas for VietPhrase and Weaviate operations
- `src/agent/prompts.py` - Contains current prompts, needs new prompts for translation and memory operations
- `src/agent/configuration.py` - Contains current configuration, needs Weaviate and translation settings
- `src/agent/utils.py` - Contains utility functions, needs VietPhrase parsing and chunking utilities
- `src/agent/weaviate_client.py` - New file for Weaviate integration and memory operations
- `src/agent/translation_tools.py` - New file for translation-specific tools and operations
- `src/agent/chunking.py` - New file for chunking Vietnamese text input
- `src/agent/memory_manager.py` - New file for managing short-term and long-term memory
- `tests/test_chunking.py` - Unit tests for Vietnamese text chunking functionality
- `tests/test_weaviate_client.py` - Unit tests for Weaviate integration
- `tests/test_translation_tools.py` - Unit tests for translation tools
- `tests/test_memory_manager.py` - Unit tests for memory management
- `tests/test_graph.py` - Integration tests for the translation workflow

### Notes

- Unit tests should typically be placed alongside the code files they are testing (e.g., `MyComponent.py` and `test_MyComponent.py` in the same directory).
- Use `pytest [optional/path/to/test/file]` to run tests. Running without a path executes all tests found by the pytest configuration.
- Frontend will be implemented using Streamlit for easy integration with the Python backend.

## Tasks

- [x] 1.0 Refactor State Management for Translation Workflow
  - [x] 1.1 Replace web research state with translation state in `state.py`
  - [x] 1.2 Create `TranslationState` TypedDict for chunk processing
  - [x] 1.3 Create `MemoryState` TypedDict for Weaviate operations
  - [x] 1.4 Create `ChunkState` TypedDict for individual chunk processing
  - [x] 1.5 Update `OverallState` to include translation-specific fields (chunks, translated_text, memory_context)
  - [x] 1.6 Add state validation for translation workflow

- [x] 2.0 Implement Vietnamese Text Chunking System

- [ ] 3.0 Integrate Weaviate for Semantic Memory Storage
  - [x] 3.1 Create `weaviate_client.py` with connection management
  - [x] 3.2 Define `KnowledgeNode` schema with type, label, content, aliases fields
  - [x] 3.3 Create node type constants (GlossaryTerm, Character, Event)
  - [ ] 3.4 Add vector embedding generation using `text2vec-google` module
  - [ ] 3.5 Implement `search_node(query, type)` function for hybrid semantic search
  - [ ] 3.6 Implement `create_node(type, content, metadata)` function
  - [ ] 3.7 Implement `update_node(id, new_content)` function
  - [ ] 3.8 Add error handling for Weaviate connection issues
  - [ ] 3.9 Add unit tests for Weaviate operations

- [ ] 4.0 Build Translation Pipeline with Memory Context
  - [ ] 4.1 Update `prompts.py` with translation-specific prompts for LLM
  - [ ] 4.2 Refactor `graph.py` to replace web research nodes with translation nodes
  - [ ] 4.3 Create `chunk_input()` node for processing Vietnamese text chunks
  - [ ] 4.4 Create `search_memory()` node for LLM to querying Weaviate before translation
  - [ ] 4.5 Create `translate_chunk()` node using Gemini for fluent Vietnamese with queried node, and recent translate context
  - [ ] 4.6 Create `memory_update()` node for LLM to decide create/update operations on Weaviate base on the queried node
  - [ ] 4.7 Create `recent_context_update()` node for LLM to update recent context if it too long
  - [ ] 4.7 Implement memory context injection (node + recent translate context) into translation prompts
  - [ ] 4.8 Add translation retry logic
  - [ ] 4.9 Add progress tracking and streaming output

- [ ] 5.0 Create python jupyter notebook to interacting with the agent
  - [ ] 5.1 Create `vra_notebook.ipynb` with example usage and documentation
  - [ ] 5.2 Implement simple function interface for Vietnamese text input
  - [ ] 5.3 Add progress display using tqdm or similar
  - [ ] 5.4 Create memory inspection widgets for viewing stored knowledge
  - [ ] 5.5 Add translation quality feedback cells
  - [ ] 5.6 Implement configuration cells for settings (chunk size, memory settings)
  - [ ] 5.7 Add export functionality for translated text
  - [ ] 5.8 Create memory visualization using matplotlib/plotly
  - [ ] 5.9 Add example notebooks for different use cases (novels, documents, etc.)