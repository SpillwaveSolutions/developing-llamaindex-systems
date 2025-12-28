# Retrieval Strategies

Advanced retrieval approaches: BM25 keyword search, hybrid fusion, and search mode selection.

## Contents

- [BM25Retriever](#bm25retriever)
  - [Core Concept](#core-concept)
  - [Basic Usage](#basic-usage)
  - [Persistence](#persistence)
- [Hybrid Search](#hybrid-search)
  - [Why Hybrid?](#why-hybrid)
  - [Fusion Strategies](#fusion-strategies)
  - [Alpha Weighting](#alpha-weighting)
  - [Implementation Pattern](#implementation-pattern)
- [Search Mode Selection](#search-mode-selection)
- [Performance Considerations](#performance-considerations)
- [Complete Examples](#complete-examples)
- [See Also](#see-also)

---

## BM25Retriever

Sparse keyword-based retrieval using the BM25 algorithm. Complements vector search by finding exact term matches that semantic embeddings may miss.

### Core Concept

BM25 (Best Matching 25) ranks documents by term frequency with diminishing returns:
- **Term Frequency (TF)**: Documents mentioning query terms more often rank higher
- **Inverse Document Frequency (IDF)**: Rare terms get higher weight than common terms
- **Length Normalization**: Long documents don't automatically rank higher

**When BM25 Excels:**
- Exact string matching (function names, error codes, IDs)
- Technical documentation with specific terminology
- Queries containing unique identifiers
- When semantic similarity misses literal matches

### Basic Usage

```python
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode

# Create nodes (typically from ingestion pipeline)
nodes = [
    TextNode(text="The RecursiveCharacterTextSplitter handles chunking...", id_="node1"),
    TextNode(text="Error code ERROR_CODE_404 indicates missing resource...", id_="node2"),
    TextNode(text="Vector embeddings capture semantic meaning...", id_="node3"),
]

# Build BM25 index
retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5,
)

# Query
results = retriever.retrieve("ERROR_CODE_404")
# Returns node2 as top result (exact term match)
```

**Prerequisites:**
```bash
pip install llama-index-retrievers-bm25
```

### Persistence

Save and load BM25 index for production use:

```python
from pathlib import Path

# Save index
persist_path = Path("./bm25_index")
persist_path.mkdir(exist_ok=True)
retriever.persist(str(persist_path))

# Load index (later)
retriever = BM25Retriever.from_persist_dir(str(persist_path))
```

### Async Support

```python
# Async retrieval
results = await retriever.aretrieve("search query")
```

---

## Hybrid Search

Combines vector semantic search with BM25 keyword search for robust retrieval.

### Why Hybrid?

| Query Type | Vector Search | BM25 Search | Hybrid |
|------------|---------------|-------------|--------|
| "authentication handler" | Finds related concepts | Exact matches only | Best of both |
| "function calculate_tax" | May miss exact name | Finds exact match | Guaranteed match |
| "how to handle errors" | Semantic understanding | Too generic | Semantic + context |
| "ERROR_CODE_404" | May miss literal | Perfect match | Perfect match |

**Key Insight:** Vector search understands meaning; BM25 finds exact terms. Together, they cover more ground.

### Fusion Strategies

#### Reciprocal Rank Fusion (RRF)

Combines rankings without score normalization. Position-based, robust to score scale differences.

```python
def rrf_score(rank: int, k: int = 60) -> float:
    """RRF formula: 1 / (k + rank)"""
    return 1.0 / (k + rank)

# Document in position 1 from both retrievers:
# RRF = 1/(60+1) + 1/(60+1) = 0.0328
```

**Characteristics:**
- Simple and robust
- No score normalization needed
- Works with any number of retrievers

#### Relative Score Fusion (RSF)

Normalizes scores to 0-1 range then combines with weighting.

```python
# Normalize scores
max_vector_score = max(r.score for r in vector_results) or 1.0
max_bm25_score = max(r.score for r in bm25_results) or 1.0

for result in combined_results:
    vector_normalized = result.vector_score / max_vector_score
    bm25_normalized = result.bm25_score / max_bm25_score
    result.score = alpha * vector_normalized + (1 - alpha) * bm25_normalized
```

**Characteristics:**
- Tunable via alpha parameter
- Requires score normalization
- More control over strategy balance

### Alpha Weighting

The `alpha` parameter controls vector vs keyword balance:

| Alpha | Vector Weight | BM25 Weight | Best For |
|-------|---------------|-------------|----------|
| `1.0` | 100% | 0% | Pure semantic search |
| `0.7` | 70% | 30% | Semantic with term boost |
| `0.5` | 50% | 50% | Equal balance (default) |
| `0.3` | 30% | 70% | Technical docs, exact terms |
| `0.0` | 0% | 100% | Pure keyword search |

**Tuning Guidelines:**
- **Conceptual queries** (how, why, explain): Higher alpha (0.7-0.9)
- **Technical queries** (function names, error codes): Lower alpha (0.3-0.5)
- **Mixed queries**: Default alpha (0.5)

### Implementation Pattern

Production-ready hybrid search combining vector and BM25:

```python
from typing import Optional
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever


class HybridRetriever:
    """
    Combines vector and BM25 retrieval with configurable fusion.
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha  # 1.0 = pure vector, 0.0 = pure BM25

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[NodeWithScore]:
        """Execute hybrid search with score fusion."""

        # 1. Run both retrievers
        vector_results = await self.vector_retriever.aretrieve(query)
        bm25_results = await self.bm25_retriever.aretrieve(query)

        # 2. Normalize scores
        max_vector = max((r.score for r in vector_results), default=1.0) or 1.0
        max_bm25 = max((r.score for r in bm25_results), default=1.0) or 1.0

        # 3. Combine results
        combined: dict[str, dict] = {}

        for result in vector_results:
            node_id = result.node.node_id
            combined[node_id] = {
                "node": result.node,
                "vector_score": result.score / max_vector,
                "bm25_score": 0.0,
            }

        for result in bm25_results:
            node_id = result.node.node_id
            bm25_normalized = result.score / max_bm25

            if node_id in combined:
                combined[node_id]["bm25_score"] = bm25_normalized
            else:
                combined[node_id] = {
                    "node": result.node,
                    "vector_score": 0.0,
                    "bm25_score": bm25_normalized,
                }

        # 4. Calculate final scores
        fused_results = []
        for data in combined.values():
            final_score = (
                self.alpha * data["vector_score"] +
                (1 - self.alpha) * data["bm25_score"]
            )
            fused_results.append(
                NodeWithScore(node=data["node"], score=final_score)
            )

        # 5. Sort and return top_k
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:top_k]
```

---

## Search Mode Selection

Pattern for supporting multiple search modes via API or CLI:

```python
from enum import Enum


class QueryMode(str, Enum):
    VECTOR = "vector"   # Pure semantic search
    BM25 = "bm25"       # Pure keyword search
    HYBRID = "hybrid"   # Combined (default)


class QueryService:
    """Executes queries based on selected mode."""

    def __init__(
        self,
        vector_retriever,
        bm25_retriever: BM25Retriever,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = HybridRetriever(
            vector_retriever, bm25_retriever
        )

    async def query(
        self,
        text: str,
        mode: QueryMode = QueryMode.HYBRID,
        alpha: float = 0.5,
        top_k: int = 5,
    ) -> list[NodeWithScore]:
        """Execute query with specified mode."""

        if mode == QueryMode.VECTOR:
            return await self.vector_retriever.aretrieve(text)
        elif mode == QueryMode.BM25:
            return await self.bm25_retriever.aretrieve(text)
        else:  # HYBRID
            self.hybrid_retriever.alpha = alpha
            return await self.hybrid_retriever.aretrieve(text, top_k=top_k)
```

---

## Performance Considerations

### BM25 Index Size

BM25 adds sparse index storage:
- Typically 20-50% of vector index size
- Stores term frequencies, document lengths
- Fast to build (no embeddings required)

### Query Latency

| Mode | Latency | Notes |
|------|---------|-------|
| Vector only | ~50-200ms | Depends on vector store |
| BM25 only | ~10-50ms | In-memory, very fast |
| Hybrid | ~60-250ms | Parallel execution recommended |

### Parallel Execution

Run vector and BM25 searches concurrently:

```python
import asyncio

async def hybrid_search_parallel(self, query: str):
    """Execute both searches in parallel."""
    vector_task = asyncio.create_task(
        self.vector_retriever.aretrieve(query)
    )
    bm25_task = asyncio.create_task(
        self.bm25_retriever.aretrieve(query)
    )

    vector_results, bm25_results = await asyncio.gather(
        vector_task, bm25_task
    )
    return self._fuse_results(vector_results, bm25_results)
```

---

## Complete Examples

### Production Hybrid Search Service

```python
import asyncio
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb


class HybridSearchService:
    """Production-ready hybrid search with persistence."""

    def __init__(self, persist_dir: str = "./search_index"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        self.embed_model = OpenAIEmbedding()
        self._vector_index = None
        self._bm25_retriever = None

    async def index_documents(self, documents: list):
        """Build both vector and BM25 indexes."""

        # 1. Create nodes
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=1024),
                self.embed_model,
            ]
        )
        nodes = pipeline.run(documents=documents)

        # 2. Build vector index
        chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir / "chroma")
        )
        collection = chroma_client.get_or_create_collection("documents")
        vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        self._vector_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

        # 3. Build BM25 index
        self._bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)
        self._bm25_retriever.persist(str(self.persist_dir / "bm25"))

        return len(nodes)

    def load(self):
        """Load existing indexes."""
        # Load vector index
        chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir / "chroma")
        )
        collection = chroma_client.get_collection("documents")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        self._vector_index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self.embed_model
        )

        # Load BM25 index
        self._bm25_retriever = BM25Retriever.from_persist_dir(
            str(self.persist_dir / "bm25")
        )

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        alpha: float = 0.5,
        top_k: int = 5,
    ):
        """Execute search with specified mode."""
        vector_retriever = self._vector_index.as_retriever(
            similarity_top_k=top_k
        )

        if mode == "vector":
            return await vector_retriever.aretrieve(query)
        elif mode == "bm25":
            return await self._bm25_retriever.aretrieve(query)
        else:
            # Hybrid: parallel execution
            vector_task = asyncio.create_task(
                vector_retriever.aretrieve(query)
            )
            bm25_task = asyncio.create_task(
                self._bm25_retriever.aretrieve(query)
            )

            vector_results, bm25_results = await asyncio.gather(
                vector_task, bm25_task
            )

            return self._fuse_results(
                vector_results, bm25_results, alpha, top_k
            )

    def _fuse_results(
        self,
        vector_results,
        bm25_results,
        alpha: float,
        top_k: int,
    ):
        """Fuse results with alpha weighting."""
        # Normalize
        max_v = max((r.score for r in vector_results), default=1.0) or 1.0
        max_b = max((r.score for r in bm25_results), default=1.0) or 1.0

        combined = {}

        for r in vector_results:
            combined[r.node.node_id] = {
                "node": r.node,
                "v": r.score / max_v,
                "b": 0.0,
            }

        for r in bm25_results:
            nid = r.node.node_id
            if nid in combined:
                combined[nid]["b"] = r.score / max_b
            else:
                combined[nid] = {"node": r.node, "v": 0.0, "b": r.score / max_b}

        # Score and sort
        from llama_index.core.schema import NodeWithScore

        results = [
            NodeWithScore(
                node=d["node"],
                score=alpha * d["v"] + (1 - alpha) * d["b"]
            )
            for d in combined.values()
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# Usage
async def main():
    service = HybridSearchService()

    # Index documents
    from llama_index.core import SimpleDirectoryReader
    docs = SimpleDirectoryReader("./data").load_data()
    count = await service.index_documents(docs)
    print(f"Indexed {count} nodes")

    # Search
    results = await service.search(
        "RecursiveCharacterTextSplitter",
        mode="hybrid",
        alpha=0.3,  # Favor BM25 for exact term
    )

    for r in results:
        print(f"Score: {r.score:.4f}")
        print(f"Text: {r.node.text[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## See Also

- [../SKILL.md](../SKILL.md) - Return to main skill overview
- [ingestion.md](ingestion.md) - Create nodes for indexing
- [context-rag.md](context-rag.md) - Query routing and reranking
- [property-graphs.md](property-graphs.md) - Graph-based hybrid retrieval
