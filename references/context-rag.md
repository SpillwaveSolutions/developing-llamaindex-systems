# Context RAG

Dynamic query routing, decomposition, and retrieval refinement.

## Contents

- [RouterQueryEngine](#routerqueryengine)
  - [Selector Types](#selector-types)
  - [QueryEngineTool Setup](#queryenginetool-setup)
  - [Router Implementation](#router-implementation)
  - [Query Routing Examples](#query-routing-examples)
- [SubQuestionQueryEngine](#subquestionqueryengine)
- [NodePostprocessors](#nodepostprocessors)
  - [SimilarityPostprocessor](#similaritypostprocessor)
  - [LLMRerank](#llmrerank)
  - [CohereRerank](#coherererank)
  - [SentenceTransformerRerank](#sentencetransformerrerank)
- [Postprocessor Comparison](#postprocessor-comparison)
- [Reranking Pipeline](#reranking-pipeline)
- [Complete Examples](#complete-examples)
- [See Also](#see-also)

---

## RouterQueryEngine

Routes queries to the optimal data source from a pool of available engines.

### Architecture

```
User Query → Selector → [Engine A | Engine B | Engine C] → Response
```

The selector analyzes the query and chooses which engine(s) to invoke based on tool descriptions.

### Selector Types

#### LLMSingleSelector

Chooses exactly one engine. Best for mutually exclusive sources.

```python
from llama_index.core.selectors import LLMSingleSelector

selector = LLMSingleSelector.from_defaults()
```

**Use Case:** "SQL Database" vs "Vector Store" — query goes to one or the other.

#### LLMMultiSelector

Chooses one or more engines. Aggregates results.

```python
from llama_index.core.selectors import LLMMultiSelector

selector = LLMMultiSelector.from_defaults()
```

**Use Case:** "Compare sales data with market sentiment" — triggers both SQL and news engines.

#### PydanticSingleSelector

Structured JSON output for reliability. Reduces parsing errors.

```python
from llama_index.core.selectors import PydanticSingleSelector

selector = PydanticSingleSelector.from_defaults()
```

**Use Case:** Production systems requiring deterministic selection.

### QueryEngineTool Setup

Each engine wrapped as a tool with a description. **The description is critical** — it's the "system prompt" for routing decisions.

```python
from llama_index.core.tools import QueryEngineTool

# Good descriptions: specific, action-oriented
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_engine,
    description=(
        "Useful for high-level summaries, thematic overviews, "
        "and questions about what the document is about."
    )
)

detail_tool = QueryEngineTool.from_defaults(
    query_engine=detail_engine,
    description=(
        "Useful for retrieving specific facts, numbers, dates, "
        "and precise details from the document."
    )
)

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_engine,
    description=(
        "Useful for structured data queries: sales figures, "
        "revenue, counts, and database records."
    )
)
```

**Description Guidelines:**
- Start with "Useful for..."
- List specific query types
- Mention data characteristics
- Keep under 200 characters

### Router Implementation

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, detail_tool, sql_tool],
    verbose=True,  # Log routing decisions
)

# Queries route automatically
response = router.query("What is this document about?")  # → summary_tool
response = router.query("What was Q3 revenue?")          # → sql_tool
```

### Query Routing Examples

| Input Query | Routed To | Reason |
|-------------|-----------|--------|
| "What is this document about?" | `summary_tool` | High-level overview request matches summary description |
| "What was Q3 revenue?" | `sql_tool` | Specific number query matches structured data description |
| "List the key themes" | `summary_tool` | Thematic request matches "thematic overviews" |
| "How many employees joined in 2024?" | `sql_tool` | Count query matches "counts, and database records" |
| "What date was the contract signed?" | `detail_tool` | Specific date matches "dates, and precise details" |
| "Summarize the main findings" | `summary_tool` | Summary request matches "high-level summaries" |

---

## SubQuestionQueryEngine

Decomposes complex queries into atomic sub-queries, executes them (optionally in parallel), and synthesizes results.

### Mechanism

```
Complex Query → [Sub-Q1, Sub-Q2, ...] → [Answer1, Answer2, ...] → Synthesis
```

### Implementation

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Define tools for different data sources
tools = [
    QueryEngineTool.from_defaults(
        query_engine=apple_engine,
        description="Information about Apple products and specifications"
    ),
    QueryEngineTool.from_defaults(
        query_engine=samsung_engine,
        description="Information about Samsung products and specifications"
    ),
]

engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    verbose=True,
)

# Complex query triggers decomposition
response = engine.query(
    "Compare the battery life of iPhone 15 and Galaxy S24"
)
# Sub-Q1: "What is the battery life of iPhone 15?" → apple_engine
# Sub-Q2: "What is the battery life of Galaxy S24?" → samsung_engine
# Synthesis: Combined comparison
```

### When to Use

- Comparative questions ("Compare X and Y")
- Multi-part queries ("What is A and how does it relate to B?")
- Cross-source aggregation
- Questions requiring multiple data sources

### Configuration

```python
engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    use_async=True,   # Parallel sub-query execution
    verbose=True,
)
```

---

## NodePostprocessors

Refine retrieved nodes after initial retrieval. Applied in sequence.

### SimilarityPostprocessor

Filters nodes below a similarity threshold.

```python
from llama_index.core.postprocessor import SimilarityPostprocessor

postprocessor = SimilarityPostprocessor(
    similarity_cutoff=0.7,  # Remove nodes below 0.7 similarity
)
```

**When to Use:**
- Prevent low-quality context from reaching LLM
- Weak vector matches causing hallucinations
- Cost reduction (fewer tokens to process)

### LLMRerank

LLM reads query + nodes and re-scores relevance.

```python
from llama_index.core.postprocessor import LLMRerank

postprocessor = LLMRerank(
    top_n=3,                    # Keep top 3 after reranking
    choice_batch_size=5,        # Process 5 nodes at a time
)
```

**Characteristics:**
- Highest precision
- High latency and cost (LLM call per batch)
- Best for critical queries

### CohereRerank

Cross-encoder reranking via Cohere API.

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

postprocessor = CohereRerank(
    api_key="your-api-key",
    top_n=3,
    model="rerank-english-v3.0",
)
```

**Prerequisites:**
```bash
pip install llama-index-postprocessor-cohere-rerank
```

**Characteristics:**
- Excellent accuracy
- Faster than LLMRerank
- API cost per request

### SentenceTransformerRerank

Local cross-encoder model. No API costs.

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

postprocessor = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
)
```

**Characteristics:**
- No API costs
- Runs locally (GPU recommended)
- Good accuracy, fast inference

---

## Postprocessor Comparison

| Postprocessor | Mechanism | Speed | Cost | Accuracy |
|---------------|-----------|-------|------|----------|
| SimilarityPostprocessor | Threshold filter | Instant | None | Low (just filtering) |
| SentenceTransformerRerank | Local cross-encoder | Fast | None | Good |
| CohereRerank | API cross-encoder | Medium | API | Excellent |
| LLMRerank | Full LLM scoring | Slow | High | Highest |

### Selection Guide

```
Budget constrained?
├─ Yes → SentenceTransformerRerank (local, free)
│
└─ No, accuracy priority:
    ├─ Latency sensitive → CohereRerank
    └─ Quality critical → LLMRerank
```

---

## Reranking Pipeline

Combine postprocessors for optimal results.

### Pattern: Filter → Rerank

```python
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    SentenceTransformerRerank,
)

query_engine = index.as_query_engine(
    similarity_top_k=10,  # Retrieve more initially
    node_postprocessors=[
        # Step 1: Remove obvious misses
        SimilarityPostprocessor(similarity_cutoff=0.5),
        # Step 2: Rerank survivors
        SentenceTransformerRerank(top_n=3),
    ]
)
```

### Pattern: Two-Stage Reranking

```python
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    SentenceTransformerRerank,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank

query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[
        # Stage 1: Fast local rerank
        SentenceTransformerRerank(top_n=10),
        # Stage 2: Precise API rerank
        CohereRerank(top_n=3),
    ]
)
```

### Pattern: Conditional Reranking

Apply expensive reranking only when needed:

```python
class ConditionalReranker:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.expensive_reranker = LLMRerank(top_n=3)
    
    def postprocess_nodes(self, nodes, query_bundle):
        # Check if top result is confident
        if nodes and nodes[0].score > self.threshold:
            return nodes[:3]  # Skip reranking
        # Otherwise, apply expensive reranking
        return self.expensive_reranker.postprocess_nodes(nodes, query_bundle)
```

---

## Complete Examples

### Multi-Source Router with Reranking

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.postprocessor import SentenceTransformerRerank

# Build engines with reranking
def build_engine(index):
    return index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[
            SentenceTransformerRerank(top_n=3),
        ]
    )

# Setup tools
tools = [
    QueryEngineTool.from_defaults(
        query_engine=build_engine(docs_index),
        description="Technical documentation and guides"
    ),
    QueryEngineTool.from_defaults(
        query_engine=build_engine(faq_index),
        description="Frequently asked questions and answers"
    ),
    QueryEngineTool.from_defaults(
        query_engine=build_engine(changelog_index),
        description="Version history and release notes"
    ),
]

# Router
router = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=tools,
)

response = router.query("What changed in version 2.0?")  # → changelog
```

### SubQuestion with Cross-Source Synthesis

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

tools = [
    QueryEngineTool.from_defaults(
        query_engine=financials_engine,
        description="Financial data: revenue, costs, margins"
    ),
    QueryEngineTool.from_defaults(
        query_engine=market_engine,
        description="Market analysis: competitors, trends, sentiment"
    ),
    QueryEngineTool.from_defaults(
        query_engine=product_engine,
        description="Product specifications and roadmap"
    ),
]

engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    use_async=True,
)

response = engine.query(
    "How does our product positioning compare to competitors "
    "and what's the revenue impact?"
)
# Decomposes into product, market, and financial sub-queries
```

---

## See Also

- [../SKILL.md](../SKILL.md) — Return to main skill overview
- [ingestion.md](ingestion.md) — Data preparation before retrieval
- [property-graphs.md](property-graphs.md) — Graph-based retrieval as router target
- [orchestration.md](orchestration.md) — Integrate routers into agent workflows
- [observability.md](observability.md) — Debug routing decisions
