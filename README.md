# Developing LlamaIndex Systems

A comprehensive Claude Code skill for building production-grade agentic RAG systems with LlamaIndex in Python.

[![SkillzWave Marketplace](https://img.shields.io/badge/SkillzWave-Marketplace-blue)](https://skillzwave.ai/skill/SpillwaveSolutions__developing-llamaindex-systems__developing-llamaindex-systems__SKILL/)
[![Agent Skill Standard](https://img.shields.io/badge/Agent_Skill-Standard-green)](https://agentskills.io/)

## Overview

This skill provides deep expertise in LlamaIndex's agentic capabilities, covering six core pillars:

| Pillar | What You'll Learn |
|--------|-------------------|
| **Semantic Ingestion** | SemanticSplitterNodeParser, CodeSplitter, IngestionPipeline, metadata extractors |
| **Retrieval Strategies** | BM25Retriever, hybrid search, alpha weighting for fusion |
| **Property Graphs** | PropertyGraphIndex, Neo4j integration, graph extractors |
| **Context RAG** | RouterQueryEngine, SubQuestionQueryEngine, LLMRerank |
| **Orchestration** | ReAct agents, event-driven Workflows, multi-agent systems |
| **Observability** | Arize Phoenix, custom handlers, evaluation pipelines |

## When to Use This Skill

This skill activates when you ask Claude Code to:

- "Build a LlamaIndex agent"
- "Set up semantic chunking"
- "Index source code with CodeSplitter"
- "Implement hybrid search"
- "Create a knowledge graph with LlamaIndex"
- "Implement query routing"
- "Debug RAG pipeline"
- "Add Phoenix observability"
- "Create an event-driven workflow"

Or when discussing: `PropertyGraphIndex`, `SemanticSplitterNodeParser`, `CodeSplitter`, `BM25Retriever`, `hybrid search`, `ReAct agent`, `Workflow pattern`, `LLMRerank`, `Text-to-Cypher`

---

## Installing with Skilz (Universal Installer)

The recommended way to install this skill across different AI coding agents is using the **skilz** universal installer. This skill supports the [Agent Skill Standard](https://agentskills.io/), which means it works with 14+ coding agents including Claude Code, OpenAI Codex, Cursor, and Gemini CLI.

### Install Skilz

```bash
pip install skilz
```

### Quick Install (Claude Code)

```bash
# Install to user home (available in all projects)
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems

# Install to current project only
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --project
```

### Install from SkillzWave Marketplace

```bash
# Claude Code (user home)
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems

# Claude Code (project level)
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems --project
```

### Other Agents

| Agent | Command |
|-------|---------|
| **OpenCode** | `skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --agent opencode` |
| **OpenAI Codex** | `skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --agent codex` |
| **Gemini CLI** | `skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --agent gemini` |

Add `--project` to any command above for project-level installation.

Skilz supports 14+ coding agents including Windsurf, Qwen Code, Aidr, and more. For the full list of supported platforms, visit [SkillzWave.ai/platforms](https://skillzwave.ai/platforms/) or see the [skilz-cli GitHub repository](https://github.com/SpillwaveSolutions/skilz-cli).

**View this skill on the marketplace:** [SkillzWave Listing](https://skillzwave.ai/skill/SpillwaveSolutions__developing-llamaindex-systems__developing-llamaindex-systems__SKILL/)

---

## Quick Start

```python
# 1. Semantic chunking
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=OpenAIEmbedding()
)
nodes = splitter.get_nodes_from_documents(docs)

# 2. Build index
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex(nodes)

# 3. Query
response = index.as_query_engine().query("What is X?")
```

## Directory Structure

```
developing-llamaindex-systems/
├── SKILL.md                        # Main skill definition
├── README.md                       # This file
├── CONTINUATION-GUIDE.md           # Session continuation guide
├── references/
│   ├── ingestion.md               # Chunking strategies, IngestionPipeline
│   ├── retrieval-strategies.md    # BM25, hybrid search, fusion
│   ├── property-graphs.md         # PropertyGraphIndex, extractors, retrievers
│   ├── context-rag.md             # Query routing, decomposition, reranking
│   ├── orchestration.md           # ReAct agents, Workflows, multi-agent
│   └── observability.md           # Phoenix, debugging, evaluation
└── scripts/
    ├── requirements.txt           # Pinned dependencies
    ├── ingest_semantic.py         # Production ingestion script
    └── agent_workflow.py          # Event-driven workflow template
```

## Reference Guide

| Task | Reference File |
|------|----------------|
| Configure chunking | [references/ingestion.md](references/ingestion.md) |
| Implement BM25 or hybrid search | [references/retrieval-strategies.md](references/retrieval-strategies.md) |
| Build knowledge graph | [references/property-graphs.md](references/property-graphs.md) |
| Implement query routing | [references/context-rag.md](references/context-rag.md) |
| Create agents/workflows | [references/orchestration.md](references/orchestration.md) |
| Debug and evaluate | [references/observability.md](references/observability.md) |

## Requirements

- Python 3.9+
- LlamaIndex 0.10+
- OpenAI API key (or configure local models)

Install dependencies:
```bash
pip install -r scripts/requirements.txt
```

## Key Features

### Semantic Chunking
Embedding-based chunking that preserves logical coherence, ideal for legal documents, technical manuals, and research papers.

### Code Splitting
Language-aware splitting for source code with configurable chunk sizes and overlap.

### Hybrid Retrieval
Combine BM25 keyword search with vector similarity using configurable alpha weighting.

### Property Graphs
Hybrid retrieval combining vector search with graph traversal. Supports Neo4j and in-memory graph stores.

### Query Routing
LLM-based routing to direct queries to specialized engines based on intent.

### Event-Driven Workflows
Type-safe, async workflows with branching, cycles, and human-in-the-loop support.

### Observability
One-line Arize Phoenix integration for full tracing, plus custom handlers for metrics and alerting.

## When NOT to Use This Skill

- **LangChain projects** - Different framework
- **Non-Python environments** - Python 3.9+ only
- **Simple Q&A bots** - Overkill if you don't need graphs, routing, or workflows
- **Offline/local-only setups** - Scripts default to OpenAI APIs; modification required for local models

---

## Links

- [SkillzWave Marketplace](https://skillzwave.ai/) - Largest Agentic Marketplace for Agent Skills
- [SpillWave](https://spillwave.com/) - Leaders in AI Agent Development
- [Agent Skill Standard](https://agentskills.io/) - Cross-platform skill specification

## License

MIT
