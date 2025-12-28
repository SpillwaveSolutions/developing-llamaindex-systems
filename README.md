# Developing LlamaIndex Systems

A comprehensive Claude Code skill for building production-grade agentic RAG systems with LlamaIndex in Python.

## Overview

This skill provides deep expertise in LlamaIndex's agentic capabilities, covering five core pillars:

| Pillar | What You'll Learn |
|--------|-------------------|
| **Semantic Ingestion** | SemanticSplitterNodeParser, IngestionPipeline, metadata extractors |
| **Property Graphs** | PropertyGraphIndex, Neo4j integration, graph extractors |
| **Context RAG** | RouterQueryEngine, SubQuestionQueryEngine, reranking |
| **Orchestration** | ReAct agents, event-driven Workflows, multi-agent systems |
| **Observability** | Arize Phoenix, custom handlers, evaluation pipelines |

## When to Use This Skill

This skill activates when you ask Claude Code to:

- "Build a LlamaIndex agent"
- "Set up semantic chunking"
- "Create a knowledge graph with LlamaIndex"
- "Implement query routing"
- "Debug RAG pipeline"
- "Add Phoenix observability"
- "Create an event-driven workflow"

Or when discussing: `PropertyGraphIndex`, `SemanticSplitterNodeParser`, `ReAct agent`, `Workflow pattern`, `LLMRerank`, `Text-to-Cypher`

## Installing with Skilz (Universal Installer)

The recommended way to install this skill across different AI coding agents is using the **skilz** universal installer.

### Install Skilz

```bash
pip install skilz
```

This skill supports [Agent Skill Standard](https://agentskills.io/) which means it supports 14 plus coding agents including Claude Code, OpenAI Codex, Cursor and Gemini.


### Git URL Options

You can use either `-g` or `--git` with HTTPS or SSH URLs:

```bash
# HTTPS URL
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems

# SSH URL
skilz install --git git@github.com:SpillwaveSolutions/developing-llamaindex-systems.git
```

### Claude Code

Install to user home (available in all projects):
```bash
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems
```

Install to current project only:
```bash
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --project
```

### OpenCode

Install for [OpenCode](https://opencode.ai):
```bash
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --agent opencode
```

Project-level install:
```bash
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --project --agent opencode
```

### Gemini

Project-level install for Gemini:
```bash
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --agent gemini
```

### OpenAI Codex

Install for OpenAI Codex:
```bash
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --agent codex
```

Project-level install:
```bash
skilz install -g https://github.com/SpillwaveSolutions/developing-llamaindex-systems --project --agent codex
```


### Install from Skillzwave Marketplace
```bash
# Claude to user home dir ~/.claude/skills
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems

# Claude skill in project folder ./claude/skills
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems --project

# OpenCode install to user home dir ~/.config/opencode/skills
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems --agent opencode

# OpenCode project level
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems --agent opencode --project

# OpenAI Codex install to user home dir ~/.codex/skills
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems

# OpenAI Codex project level ./.codex/skills
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems --agent opencode --project

# Gemini CLI (project level) -- only works with project level
skilz install SpillwaveSolutions_developing-llamaindex-systems/developing-llamaindex-systems --agent gemini
```

See this site [skill Listing](https://skillzwave.ai/skill/SpillwaveSolutions__developing-llamaindex-systems__developing-llamaindex-systems__SKILL/) to see how to install this exact skill to 14+ different coding agents.


### Other Supported Agents

Skilz supports 14+ coding agents including Windsurf, Qwen Code, Aidr, and more.

For the full list of supported platforms, visit [SkillzWave.ai/platforms](https://skillzwave.ai/platforms/) or see the [skilz-cli GitHub repository](https://github.com/SpillwaveSolutions/skilz-cli)


<a href="https://skillzwave.ai/">Largest Agentic Marketplace for Agent Skills</a> and
<a href="https://spillwave.com/">SpillWave: Leaders in AI Agent Development.</a>

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
├── SKILL.md                    # Main skill definition
├── README.md                   # This file
├── references/
│   ├── ingestion.md           # Chunking strategies, IngestionPipeline
│   ├── property-graphs.md     # PropertyGraphIndex, extractors, retrievers
│   ├── context-rag.md         # Query routing, decomposition, reranking
│   ├── orchestration.md       # ReAct agents, Workflows, multi-agent
│   └── observability.md       # Phoenix, debugging, evaluation
└── scripts/
    ├── requirements.txt       # Pinned dependencies
    ├── ingest_semantic.py     # Production ingestion script
    └── agent_workflow.py      # Event-driven workflow template
```

## Reference Guide

| Task | Reference File |
|------|----------------|
| Configure chunking | [references/ingestion.md](references/ingestion.md) |
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

## License

MIT
