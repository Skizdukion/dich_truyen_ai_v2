# Project Requirements Document (PRD)

## Title

**VietPhrase Reader Assistant (VRA)**\
An AI-powered assistant specialized in converting *VietPhrase* (a bilingual format derived from Chinese texts) into fluent, high-quality Vietnamese suitable for casual reading and translation validation. It uses LLM-driven translation refinement and integrates Weaviate-based storage to persist and enrich contextual knowledge such as characters, glossary terms, key events, and style corrections.


---

## 2. Background & Problem Statement

Vietnamese fan translators often rely on VietPhrase—line-by-line bilingual scripts that map Chinese segments to Vietnamese meanings—to consume or retranslate novels. However, these scripts are rarely fluent, often awkward, and difficult for general readers to enjoy. Additionally, transformation into readable Vietnamese remains manual and slow.

> **Opportunity:** Build a smart translation agent that automatically references previous decisions (character names, glossary terms, recurring phrases), updates memory with new findings, and generates fluent output from rigid input.

---

## 3. Weaviate Integration & Autonomous Agent Logic

### Key Concepts

The agent maintains a long-term semantic memory in Weaviate. It treats all domain elements (characters, terminology, key events, unique expressions) as nodes. During each translation session:

1. **Input**: A user submits a full chapter (structured VietPhrase).
2. **Chunking**: System splits it into coherent segments (phrases or paragraphs).
3. **Search Phase**:
   - For each chunk, the agent queries Weaviate for similar entries (e.g., previously seen glosses, names, context phrases).
   - Retrieved nodes are used to inform the translation.
4. **Translation Phase**:
   - Agent composes fluent Vietnamese, enhanced by retrieved context.
5. **Update Phase**:
   - If new glossary terms, characters, or named concepts are detected, the agent creates new nodes in Weaviate.
   - If existing nodes were reused with modification, it triggers an update.
6. **Memory Update**:
   - Recent context is stored and summarized for reuse in upcoming chunks.

### Operations

| Operation                              | Purpose                                                             | Interface                         |
| -------------------------------------- | ------------------------------------------------------------------- | --------------------------------- |
| `search_node(query, type)`             | Retrieve contextually similar memory nodes (term, character, event) | Hybrid semantic search            |
| `create_node(type, content, metadata)` | Persist new entity learned during translation                       | Auto-generated from LLM output    |
| `update_node(id, new_content)`         | Revise existing glossary or memory entity                           | Based on LLM feedback & edits     |
| `store_context(chunk, references)`     | Save short-term translation context                                 | Used for continuity across chunks |

### Node Types

- `GlossaryTerm`
- `Character`
- `Event`

All nodes are searchable via hybrid retrieval and have vector embeddings produced by Weaviate’s `text2vec-google` module (Gemini API).

### Example Schema (Flexible Node Type)

```json
{
  "class": "KnowledgeNode",
  "properties": [
    {"name": "type", "dataType": ["text"]},
    {"name": "label", "dataType": ["text"]},
    {"name": "content", "dataType": ["text"]},
    {"name": "aliases", "dataType": ["text[]"]},
  ]
}
```

---

## 4. User Stories

| ID    | As a…      | I want to…                                                 | So that…                                            |
| ----- | ---------- | ---------------------------------------------------------- | --------------------------------------------------- |
| US‑T1 | translator | import a VietPhrase-formatted chapter                      | I can convert it automatically to fluent Vietnamese |
| US‑A1 | agent      | search related characters, terms, and phrasing nodes       | improve translation accuracy and coherence          |
| US‑A2 | agent      | create new glossary/character/event nodes when translating | update system memory without user input             |
| US‑A3 | agent      | update previous entries if the translation improves them   | maintain consistency and reduce redundancy          |
| US‑A4 | agent      | persist recent context to guide next chunk's translation   | ensure continuity within the chapter                |

---

## 5. Scope

### In‑Scope

- Chunking and context-chained translation
- Glossary / entity auto-suggestion
- Flexible node system stored in Weaviate
- Semantic memory-based prompting to LLM

### Out‑of‑Scope (Phase 1)

- Manual glossary editing
- Multi-user collaboration
- UI for memory graph navigation

---

## 6. Functional Requirements

| #  | Requirement                                                                                      | Priority |
| -- | ------------------------------------------------------------------------------------------------ | -------- |
| F1 | The system SHALL chunk input chapters into logical translation units                             | Must     |
| F2 | The system SHALL search relevant Weaviate nodes before translating each chunk                    | Must     |
| F3 | The system SHALL use retrieved knowledge as context for translation                              | Must     |
| F4 | The system SHALL generate fluent Vietnamese per chunk                                            | Must     |
| F5 | The system SHALL create or update `KnowledgeNode` entries when translation introduces novel data | Must     |
| F6 | The system SHALL store recent context to short-term memory                                       | Must     |
| F7 | The system SHALL re-use short-term memory in subsequent chunks                                   | Must     |

---

## 7. Technical Architecture

```
┌────────┐  HTTP   ┌────────────--------------┐   ┌─────────────┐
│ Client │────────▶│ Orchestrator             │──▶│ LLM + Tools │
└────────┘         │  LangGraph               │   └─────────────┘
     ▲             │ Tools:                   │
     │             │  - chunk_input           │
     │             │  - search_node           │
     │             │  - translate_chunk       │
     │             │  - create_or_update_node │
     ▼             └────────────--------------┘
  SSE / Auth       ┌────────────┐
                   │ Weaviate   │
                   └────────────┘
```

*Weaviate is deployed with the **`text2vec-google`** vectorizer (Gemini embeddings).*

---

## 8. Workflow Example

1. **User** uploads a VietPhrase chapter
2. **Agent** splits it into 10 chunks
3. For each chunk:
   - Queries Weaviate for glossary/character/event memory
   - Sends retrieved context + chunk text to LLM
   - Receives translated result
   - Auto-creates/updates relevant nodes
   - Updates internal working memory
4. Final output is combined and exported

---

## 10. Risks & Mitigations

| Risk                    | Impact             | Likelihood | Mitigation                          |
| ----------------------- | ------------------ | ---------- | ----------------------------------- |
| Redundant node creation | Memory bloat       | High       | Canonical alias & fuzzy merge logic |
| LLM hallucination       | Bad memory updates | Med        | Feedback loop, review queue         |
| Token limits            | Context overflow   | High       | Prioritized pruning, summaries      |

---

## 11. Appendix

- Glossary: VietPhrase, Weaviate, LLM, LangGraph
- Related: AutoGlossary, NovelTermExtractor, SegmentAligner

