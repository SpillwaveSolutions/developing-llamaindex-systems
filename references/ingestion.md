# Semantic Ingestion

Deep dive into data ingestion, chunking strategies, and metadata enrichment.

## Contents

- [IngestionPipeline](#ingestionpipeline)
- [Node Parsers](#node-parsers)
  - [CodeSplitter](#codesplitter)
  - [SemanticSplitterNodeParser](#semanticsplitternodeparser)
  - [SentenceSplitter](#sentencesplitter)
  - [SentenceWindowNodeParser](#sentencewindownodeparser)
- [Comparison Table](#node-parser-comparison)
- [Metadata Extractors](#metadata-extractors)
- [Complete Pipeline Example](#complete-pipeline-example)
- [Performance Tuning](#performance-tuning)
- [See Also](#see-also)

---

## IngestionPipeline

Central processing unit for transforming documents into queryable nodes.

### Architecture

```
Documents → [Parse] → [Transform] → [Embed] → Nodes
```

Three stages:
1. **Parsing**: Raw files (PDF, HTML, JSON) → Document objects
2. **Transformation**: Documents → Node objects via node parsers
3. **Embedding**: Nodes → Vector representations

### Basic Setup

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

pipeline = IngestionPipeline(
    transformations=[
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=OpenAIEmbedding()
        ),
        OpenAIEmbedding(),
    ]
)

nodes = pipeline.run(documents=documents)
```

### Incremental Updates

Pipeline supports deduplication for incremental indexing via caching.

#### In-Memory Cache

```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

cache = IngestionCache()
pipeline = IngestionPipeline(
    transformations=[...],
    cache=cache,
)

# First run: processes all
nodes = pipeline.run(documents=docs)

# Second run: only processes new/changed documents
nodes = pipeline.run(documents=updated_docs)
```

#### Redis Cache (Production)

```python
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.core.ingestion import IngestionCache

# Connect to Redis
redis_kvstore = RedisKVStore(
    host="localhost",
    port=6379,
    # password="your-password",  # If auth required
)

cache = IngestionCache(cache=redis_kvstore)

pipeline = IngestionPipeline(
    transformations=[splitter, embed_model],
    cache=cache,
)

# Documents are fingerprinted; unchanged docs skip processing
nodes = pipeline.run(documents=docs)
```

**Prerequisites:**
```bash
pip install redis llama-index-storage-kvstore-redis
```

#### Cache Behavior

| Scenario | Behavior |
|----------|----------|
| Same document, same content | Skipped (cache hit) |
| Same document, changed content | Reprocessed |
| New document | Processed and cached |
| Document removed | Not automatically cleaned |

#### Clearing Cache

```python
# Clear entire cache
cache.clear()

# Or with Redis, delete specific keys
redis_kvstore.delete("ingestion_cache:doc_hash_abc123")
```

---

## Node Parsers

### CodeSplitter

AST-aware code chunking that respects function and class boundaries. Uses tree-sitter for parsing.

#### Why AST-Aware Chunking?

Standard text splitters break code at arbitrary character boundaries, losing context:

```python
# Bad: Function split mid-implementation
def calculate_tax(amount):
    rate = 0.15
    if amount > 10000:
        rate = 0.25
# --- chunk boundary ---
    return amount * rate  # Lost context!
```

CodeSplitter chunks at logical boundaries (functions, classes, methods).

#### Basic Usage

```python
from llama_index.core.node_parser import CodeSplitter

splitter = CodeSplitter(
    language="python",      # Required: target language
    chunk_lines=40,         # Lines per chunk
    chunk_lines_overlap=15, # Overlap between chunks
    max_chars=1500,         # Maximum characters per chunk
)

# From documents
nodes = splitter.get_nodes_from_documents(documents)
```

**Prerequisites:**
```bash
pip install llama-index tree-sitter tree-sitter-languages
```

#### Supported Languages

| Language | Tree-sitter Name | File Extensions |
|----------|------------------|-----------------|
| Python | `python` | .py |
| TypeScript | `typescript` | .ts |
| TSX | `tsx` | .tsx |
| JavaScript | `javascript` | .js |
| JSX | `jsx` | .jsx |
| Java | `java` | .java |
| Go | `go` | .go |
| Rust | `rust` | .rs |
| C++ | `cpp` | .cpp, .hpp |
| C | `c` | .c, .h |

#### Configuration

```python
splitter = CodeSplitter(
    language="python",
    chunk_lines=40,         # Target lines per chunk
    chunk_lines_overlap=15, # Context preservation
    max_chars=1500,         # Hard limit for LLM context
)
```

**Parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_lines=40` | 40 | Target chunk size in lines |
| `chunk_lines_overlap=15` | 15 | Lines of context overlap |
| `max_chars=1500` | 1500 | Character limit (prevents huge functions) |

#### Extracting Code Metadata

Combine CodeSplitter with metadata extraction for richer search:

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.extractors import SummaryExtractor

pipeline = IngestionPipeline(
    transformations=[
        CodeSplitter(language="python", chunk_lines=40),
        SummaryExtractor(),  # Generate natural language descriptions
        embed_model,
    ]
)

nodes = pipeline.run(documents=code_documents)
# Each node now has:
# - text: The code chunk
# - metadata.section_summary: "This function calculates tax..."
```

#### Multi-Language Corpus

Handle mixed-language codebases:

```python
from pathlib import Path

LANGUAGE_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".java": "java",
    ".go": "go",
}


def get_code_splitter(file_path: str) -> CodeSplitter:
    """Return appropriate splitter for file type."""
    ext = Path(file_path).suffix.lower()
    language = LANGUAGE_MAP.get(ext, "python")
    return CodeSplitter(
        language=language,
        chunk_lines=40,
        chunk_lines_overlap=15,
    )


# Process each file with correct splitter
all_nodes = []
for doc in documents:
    splitter = get_code_splitter(doc.metadata.get("file_path", ""))
    nodes = splitter.get_nodes_from_documents([doc])

    # Add language metadata
    for node in nodes:
        node.metadata["language"] = splitter.language
        node.metadata["source_type"] = "code"

    all_nodes.extend(nodes)
```

#### When to Use

- **Source code indexing**: SDK code, library implementations
- **Code search**: Find functions by name or behavior
- **Documentation generation**: Code + docs in unified corpus
- **Tutorial writing**: Reference actual implementation patterns

---

### SemanticSplitterNodeParser

Embedding-based chunking that preserves logical coherence.

#### Mechanism

1. Text divided into sentences
2. Buffer of sentences encoded to vectors
3. Cosine similarity calculated between adjacent buffers
4. Split occurs when similarity drops below threshold

#### Configuration

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

splitter = SemanticSplitterNodeParser(
    buffer_size=1,                        # Sentences per buffer (1-5)
    breakpoint_percentile_threshold=95,   # Split sensitivity (0-100)
    embed_model=OpenAIEmbedding(),        # Required
)
```

#### Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `buffer_size=1` | 1-5 | Compare individual sentences (sensitive, noisy) |
| `buffer_size=3` | 1-5 | Compare rolling windows (stable, may miss subtle shifts) |
| `threshold=95` | 0-100 | Split only on major topic changes (fewer, larger chunks) |
| `threshold=70` | 0-100 | Split on moderate changes (more, smaller chunks) |

#### When to Use

- Legal documents (preserve clause boundaries)
- Technical manuals (preserve procedure steps)
- Research papers (preserve argument flow)
- Any document where logical coherence matters

#### Trade-offs

- **Pro**: Semantically coherent chunks
- **Con**: Requires embedding calls during ingestion (cost + latency)
- **Con**: Variable chunk sizes (harder to predict token usage)

---

### SentenceSplitter

Fixed token-window splitting. Fast but semantically blind.

#### Configuration

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1024,      # Tokens per chunk
    chunk_overlap=20,     # Overlap between chunks
)
```

#### When to Use

- Large corpus bulk processing
- Speed priority over precision
- Homogeneous content (news articles, blog posts)
- Budget constraints (no embedding cost during ingestion)

---

### SentenceWindowNodeParser

Single-sentence nodes with surrounding context stored in metadata.

#### Configuration

```python
from llama_index.core.node_parser import SentenceWindowNodeParser

splitter = SentenceWindowNodeParser(
    window_size=3,              # Sentences before/after
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
```

#### Retrieval Pattern

Requires `MetadataReplacementPostProcessor` to expand context at query time:

```python
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ]
)
```

#### When to Use

- Fine-grained retrieval needs (exact sentence matching)
- QA over dense technical content
- When both precision and context matter

---

## Node Parser Comparison

| Feature | CodeSplitter | SemanticSplitter | SentenceSplitter | SentenceWindow |
|---------|--------------|------------------|------------------|----------------|
| Splitting Logic | AST boundaries | Embedding similarity | Token count | Single sentence |
| Context Preservation | Function/class scope | Thematic | Arbitrary overlap | Metadata window |
| Chunk Size | Variable (logical) | Variable | Fixed | 1 sentence |
| Ingestion Cost | Low (parsing) | High (embeddings) | Negligible | Low |
| Best For | Source code | Complex reasoning | Bulk processing | Fine-grained QA |

---

## Metadata Extractors

Enrich nodes with self-describing context.

### TitleExtractor

Infers document and section titles via LLM.

```python
from llama_index.core.extractors import TitleExtractor

extractor = TitleExtractor(
    nodes=5,  # Number of nodes to derive title from
)
```

**Output metadata**: `{"document_title": "...", "section_title": "..."}`

### SummaryExtractor

Generates concise summary of node content.

```python
from llama_index.core.extractors import SummaryExtractor

extractor = SummaryExtractor(
    summaries=["self", "prev", "next"],  # What to summarize
)
```

**Output metadata**: `{"section_summary": "...", "prev_section_summary": "..."}`

### KeywordExtractor

Extracts entities and keywords for hybrid search.

```python
from llama_index.core.extractors import KeywordExtractor

extractor = KeywordExtractor(
    keywords=5,  # Number of keywords
)
```

**Output metadata**: `{"keywords": ["revenue", "Q3", "growth"]}`

### Combining Extractors

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    TitleExtractor, SummaryExtractor, KeywordExtractor
)

pipeline = IngestionPipeline(
    transformations=[
        splitter,
        TitleExtractor(),
        SummaryExtractor(),
        KeywordExtractor(keywords=5),
        embed_model,
    ]
)
```

**Result**: Each node contains:
```python
Node(
    text="...",
    metadata={
        "document_title": "Q3 Earnings Report",
        "section_title": "Risk Factors",
        "section_summary": "Discusses market volatility...",
        "keywords": ["risk", "volatility", "market"],
    }
)
```

---

## Complete Pipeline Example

Production-ready ingestion with semantic chunking and full metadata:

```python
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.extractors import (
    TitleExtractor, SummaryExtractor, KeywordExtractor
)
from llama_index.embeddings.openai import OpenAIEmbedding

# Setup
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# Load documents
docs = SimpleDirectoryReader("./data").load_data()

# Build pipeline
pipeline = IngestionPipeline(
    transformations=[
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        ),
        TitleExtractor(),
        KeywordExtractor(keywords=5),
        embed_model,
    ]
)

# Execute
nodes = pipeline.run(documents=docs, show_progress=True)

print(f"Created {len(nodes)} nodes")
```

---

## Performance Tuning

### Reduce Semantic Chunking Latency

1. **Lower buffer_size**: `buffer_size=1` is fastest
2. **Local embeddings**: Replace OpenAI with HuggingFace (see below)
3. **Batch processing**: Process documents in batches to amortize overhead

### Local Embedding Models

Eliminate API costs and latency with local models.

#### HuggingFace Embeddings

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Small and fast (recommended for development)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Better quality (production)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Best quality (if GPU available)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
```

**Prerequisites:**
```bash
pip install llama-index-embeddings-huggingface sentence-transformers
```

#### Model Comparison

| Model | Dimensions | Speed | Quality | Size |
|-------|------------|-------|---------|------|
| bge-small-en-v1.5 | 384 | Fast | Good | 130MB |
| bge-base-en-v1.5 | 768 | Medium | Better | 440MB |
| bge-large-en-v1.5 | 1024 | Slow | Best | 1.3GB |
| text-embedding-3-small | 1536 | API | Good | N/A |

#### GPU Acceleration

```python
# Automatic GPU detection
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    device="cuda",  # or "mps" for Apple Silicon
)

# Verify GPU usage
import torch
print(f"Using device: {embed_model._device}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

#### Using with Semantic Splitter

```python
# Local embeddings for both splitting and indexing
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model,  # Uses local model
)

# No API calls during ingestion
nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
```

### Hybrid Strategy for Large Corpora

Use SentenceSplitter for bulk content, SemanticSplitter for critical documents:

```python
# Bulk content
bulk_splitter = SentenceSplitter(chunk_size=1024)
bulk_nodes = bulk_splitter.get_nodes_from_documents(bulk_docs)

# Critical documents
semantic_splitter = SemanticSplitterNodeParser(...)
critical_nodes = semantic_splitter.get_nodes_from_documents(critical_docs)

# Combine
all_nodes = bulk_nodes + critical_nodes
```

---

## See Also

- [../SKILL.md](../SKILL.md) — Return to main skill overview
- [property-graphs.md](property-graphs.md) — Store nodes in PropertyGraphIndex
- [context-rag.md](context-rag.md) — Query routing and postprocessing
- [observability.md](observability.md) — Debug ingestion issues
