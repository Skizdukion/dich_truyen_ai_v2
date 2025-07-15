# Enhanced Translation Flow Implementation Tasks

## Relevant Files

- `src/agent/state.py` - Contains the state management classes that need to be extended for the new flow.
- `src/agent/state.py` - Unit tests for state management classes.
- `src/agent/graph.py` - Contains the main translation workflow graph that needs to be updated.
- `src/agent/graph.py` - Unit tests for the translation workflow graph.
- `src/agent/translation_tools.py` - Contains translation tools that need to be enhanced with review and feedback capabilities.
- `src/agent/translation_tools.py` - Unit tests for translation tools.
- `src/agent/prompts.py` - Contains prompts that need to be updated for the new review and feedback flow.
- `src/agent/prompts.py` - Unit tests for prompts.
- `src/agent/chunking_utils.py` - New file for chunking utilities to handle big chunks (16k) and small chunks (~500 words).
- `src/agent/chunking_utils.py` - Unit tests for chunking utilities.
- `src/agent/review_agent.py` - New file for the review agent that rates translations and provides feedback.
- `src/agent/review_agent.py` - Unit tests for review agent.
- `src/agent/enhanced_graph.py` - New file for the enhanced translation workflow with the new flow.
- `src/agent/enhanced_graph.py` - Unit tests for enhanced workflow.
- `src/agent/configuration.py` - May need updates to support new model configurations for review agent.
- `src/agent/configuration.py` - Unit tests for configuration.

### Notes

- Unit tests should typically be placed alongside the code files they are testing (e.g., `MyComponent.tsx` and `MyComponent.test.tsx` in the same directory).
- Use `npx jest [optional/path/to/test/file]` to run tests. Running without a path executes all tests found by the Jest configuration.

## Tasks

- [ ] 1.0 Extend State Management for Enhanced Flow
  - [ ] 1.1 Add new state classes for big chunk processing (BigChunkState)
  - [ ] 1.2 Add new state classes for small chunk processing (SmallChunkState)
  - [ ] 1.3 Add review state management (ReviewState) with rating and feedback fields
  - [ ] 1.4 Update OverallState to include big chunks, small chunks, and review tracking
  - [ ] 1.5 Add retry mechanism state for failed translations with feedback

- [ ] 2.0 Implement Chunking Utilities
  - [ ] 2.1 Create chunking utility to split text into big chunks (16k limit)
  - [ ] 2.2 Create chunking utility to split big chunks into small chunks (~500 words)
  - [ ] 2.3 Implement chunk metadata tracking (size, position, relationships)
  - [ ] 2.4 Add validation for chunk size limits and overlap handling
  - [ ] 2.5 Create utility functions for chunk reassembly and context preservation

- [ ] 3.0 Implement Review Agent System
  - [ ] 3.1 Create ReviewAgent class with rating capabilities (0-10.0 scale)
  - [ ] 3.2 Implement feedback generation for translations below 7.0 rating
  - [ ] 3.3 Add prompt templates for review and feedback generation
  - [ ] 3.4 Implement review agent configuration and LLM integration
  - [ ] 3.5 Add review result caching and history tracking

- [ ] 4.0 Enhance Translation Tools
  - [ ] 4.1 Update translation tools to handle small chunk translation
  - [ ] 4.2 Add feedback incorporation in retranslation process
  - [ ] 4.3 Implement recent context update mechanism for small chunks
  - [ ] 4.4 Add translation quality tracking and metrics
  - [ ] 4.5 Enhance memory context retrieval for big chunks

- [ ] 5.0 Create Enhanced Translation Graph
  - [ ] 5.1 Design new workflow: big chunk → memory search → small chunks → translation → review → retry if needed
  - [ ] 5.2 Implement big chunk processing node
  - [ ] 5.3 Implement small chunk processing node with sequential translation
  - [ ] 5.4 Implement review node with rating and feedback
  - [ ] 5.5 Implement retry node for failed translations with feedback
  - [ ] 5.6 Implement final node update for related nodes after all small chunks complete
  - [ ] 5.7 Add conditional edges for review feedback loop and completion

- [ ] 6.0 Update Prompts and Instructions
  - [ ] 6.1 Create prompts for big chunk memory search
  - [ ] 6.2 Create prompts for small chunk translation with context
  - [ ] 6.3 Create prompts for review agent rating and feedback
  - [ ] 6.4 Create prompts for retranslation with feedback incorporation
  - [ ] 6.5 Update existing prompts to work with new flow structure

- [ ] 7.0 Implement Memory and Context Management
  - [ ] 7.1 Enhance memory search for big chunks to retrieve related nodes
  - [ ] 7.2 Implement recent context update for each small chunk translation
  - [ ] 7.3 Add context preservation between small chunks within big chunks
  - [ ] 7.4 Implement final node update mechanism after all translations complete
  - [ ] 7.5 Add memory operation logging and tracking

- [ ] 8.0 Add Configuration and Testing
  - [ ] 8.1 Update configuration to support review agent models
  - [ ] 8.2 Add configuration for chunk size limits and thresholds
  - [ ] 8.3 Create comprehensive unit tests for all new components
  - [ ] 8.4 Add integration tests for the complete enhanced flow
  - [ ] 8.5 Add performance tests for chunking and review processes

- [ ] 9.0 Documentation and Integration
  - [ ] 9.1 Update README with new enhanced flow documentation
  - [ ] 9.2 Add usage examples for the enhanced translation flow
  - [ ] 9.3 Create migration guide from old flow to new flow
  - [ ] 9.4 Add monitoring and logging for the enhanced flow
  - [ ] 9.5 Update API documentation for new endpoints and parameters 