# Property Graphs

Knowledge graph construction, storage backends, extraction strategies, and retrieval modes.

## Contents

- [PropertyGraphIndex Overview](#propertygraphindex-overview)
- [Graph Storage Backends](#graph-storage-backends)
  - [SimplePropertyGraphStore](#simplepropertygraphstore)
  - [Neo4jPropertyGraphStore](#neo4jpropertygraphstore)
  - [Other Backends](#other-backends)
- [Knowledge Extraction](#knowledge-extraction)
  - [ImplicitPathExtractor](#implicitpathextractor)
  - [SimpleLLMPathExtractor](#simplellmpathextractor)
  - [SchemaLLMPathExtractor](#schemallmpathextractor)
- [Extractor Comparison](#extractor-comparison)
- [Retrieval Strategies](#retrieval-strategies)
  - [VectorContextRetriever](#vectorcontextretriever)
  - [TextToCypherRetriever](#texttocypherretriever)
  - [CypherTemplateRetriever](#cyphertemplateretriever)
- [Retriever Comparison](#retriever-comparison)
- [Complete Example](#complete-example)
- [Schema Design Guidelines](#schema-design-guidelines)
- [See Also](#see-also)

---

## PropertyGraphIndex Overview

Hybrid index combining vector embeddings with labeled property graph structure.

### Graph Model

- **Nodes**: Entities (Person, Company) or text chunks
- **Edges**: Relationships (FOUNDED, WORKS_AT, MENTIONS)
- **Properties**: Metadata on nodes and edges (dates, scores)

### Key Capability

Vector embeddings attach to graph nodes, enabling "Vector-to-Graph" retrieval:
1. Semantic search finds relevant nodes
2. Graph traversal discovers connected facts
3. Combined context returned to LLM

### Basic Construction

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

docs = SimpleDirectoryReader("./data").load_data()

index = PropertyGraphIndex.from_documents(
    docs,
    embed_model=OpenAIEmbedding(),
    show_progress=True,
)
```

---

## Graph Storage Backends

### SimplePropertyGraphStore

In-memory or file-based storage. No external dependencies.

```python
from llama_index.core.graph_stores import SimplePropertyGraphStore

graph_store = SimplePropertyGraphStore()

index = PropertyGraphIndex.from_documents(
    docs,
    property_graph_store=graph_store,
    embed_model=embed_model,
)

# Persist to disk
index.storage_context.persist(persist_dir="./storage")
```

**Characteristics:**
- Serializes to JSON/Dict
- No native Cypher support
- Supports networkx visualization
- Best for: Prototyping, small datasets (<10K nodes)

### Neo4jPropertyGraphStore

Production-grade graph database integration.

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
    database="neo4j",
)

index = PropertyGraphIndex.from_documents(
    docs,
    property_graph_store=graph_store,
    embed_model=embed_model,
)
```

**Characteristics:**
- Native Cypher query execution
- ACID transactions
- Built-in vector index support
- Best for: Production, large scale, complex queries

**Prerequisites:**
```bash
pip install llama-index-graph-stores-neo4j
# Neo4j server running (Docker or cloud)
```

### Other Backends

| Backend | Use Case |
|---------|----------|
| `NebulaPropertyGraphStore` | Distributed, high availability |
| `TiDBPropertyGraphStore` | MySQL-compatible, HTAP workloads |
| `FalkorDBPropertyGraphStore` | Redis-based, low latency |

Install pattern: `pip install llama-index-graph-stores-{backend}`

---

## Knowledge Extraction

Extractors analyze text and output graph triples (Subject, Predicate, Object).

### ImplicitPathExtractor

Extracts document structure without LLM. Zero cost.

```python
from llama_index.core.indices.property_graph import ImplicitPathExtractor

extractor = ImplicitPathExtractor()
```

**Generated Relationships:**
- `NEXT` / `PREVIOUS` — Sequential chunks
- `SOURCE` — Chunk to source document
- `PARENT` — Chunk to parent section

**When to Use:**
- Document navigation ("read next section")
- Structural queries ("show parent document")
- Baseline graph with no LLM cost

### SimpleLLMPathExtractor

LLM-powered extraction with dynamic ontology.

```python
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.openai import OpenAI

extractor = SimpleLLMPathExtractor(
    llm=OpenAI(model="gpt-4-turbo"),
    max_paths_per_chunk=10,  # Limit triples per node
)
```

**Example Output:**
```
Text: "Apple released the Vision Pro headset in 2024"
→ (Apple)--[RELEASED]-->(Vision Pro)
→ (Vision Pro)--[TYPE]-->(Headset)
→ (Vision Pro)--[RELEASED_IN]-->(2024)
```

**Custom Prompt:**

```python
extractor = SimpleLLMPathExtractor(
    llm=llm,
    extract_prompt=(
        "Extract entities and relationships from the text. "
        "Focus on people, organizations, and their interactions. "
        "Output as (entity1)--[RELATIONSHIP]-->(entity2)"
    ),
)
```

**When to Use:**
- Discovery and exploration
- Broad knowledge mapping
- Unknown or variable schemas

### SchemaLLMPathExtractor

LLM extraction constrained to predefined ontology.

```python
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

# Define allowed entities and relationships
schema = {
    "PERSON": ["WORKS_AT", "FOUNDED", "INVESTED_IN"],
    "COMPANY": ["LOCATED_IN", "ACQUIRED", "PRODUCES"],
    "PRODUCT": ["RELEASED_BY", "COMPETES_WITH"],
}

extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=list(schema.keys()),
    possible_relations=list(set(r for rels in schema.values() for r in rels)),
    strict=True,  # Reject non-conforming extractions
)
```

**When to Use:**
- Regulated domains (finance, healthcare)
- Consistent querying requirements
- Preventing ontology drift (WORKS_AT vs EMPLOYED_BY)

---

## Extractor Comparison

| Extractor | LLM Cost | Schema | Best For |
|-----------|----------|--------|----------|
| ImplicitPathExtractor | None | Fixed (doc structure) | Navigation, baseline |
| SimpleLLMPathExtractor | High | Dynamic | Discovery, exploration |
| SchemaLLMPathExtractor | High | Strict | Regulated, consistent |

### Combining Extractors

Use multiple extractors for comprehensive graphs:

```python
index = PropertyGraphIndex.from_documents(
    docs,
    kg_extractors=[
        ImplicitPathExtractor(),  # Structure (free)
        SimpleLLMPathExtractor(max_paths_per_chunk=5),  # Concepts
    ],
    embed_model=embed_model,
)
```

---

## Retrieval Strategies

### VectorContextRetriever

Most robust. Vector search + graph traversal.

```python
from llama_index.core.indices.property_graph import VectorContextRetriever

retriever = VectorContextRetriever(
    index.property_graph_store,
    embed_model=embed_model,
    include_text=True,   # Include node text in context
    path_depth=2,        # Traversal hops from matched nodes
    similarity_top_k=5,  # Initial vector matches
)

# Or via index
retriever = index.as_retriever(
    include_text=True,
    similarity_top_k=5,
)
```

**Mechanism:**
1. Vector search finds semantically similar nodes
2. Graph traversal collects connected nodes (up to `path_depth`)
3. Combined context returned

**When to Use:**
- General-purpose retrieval
- Robustness priority (no LLM code generation)
- Unknown query patterns

### TextToCypherRetriever

LLM generates Cypher queries from natural language.

```python
from llama_index.core.indices.property_graph import TextToCypherRetriever

retriever = TextToCypherRetriever(
    index.property_graph_store,
    llm=llm,
)

# Query
nodes = retriever.retrieve("How many employees does Apple have?")
# LLM generates: MATCH (c:Company {name: 'Apple'})-[:EMPLOYS]->(e) RETURN count(e)
```

**When to Use:**
- Complex aggregations (COUNT, SUM, AVG)
- Filtering with conditions
- Trusted environments only

**Risks:**
- Cypher syntax errors
- Injection vulnerabilities (sandbox required)
- Non-deterministic results

### CypherTemplateRetriever

Parameterized Cypher for safety + flexibility.

```python
from llama_index.core.indices.property_graph import CypherTemplateRetriever

retriever = CypherTemplateRetriever(
    index.property_graph_store,
    llm=llm,
    cypher_template=(
        "MATCH (p:Person {name: $name})-[:WROTE]->(b:Book) "
        "RETURN b.title AS title, b.year AS year"
    ),
    template_params=["name"],
)

# Query: "What books did George Orwell write?"
# LLM extracts: name="George Orwell"
# Executes template with parameter
```

**When to Use:**
- Known query patterns
- Security-sensitive environments
- 100% syntactic correctness required

---

## Retriever Comparison

| Retriever | LLM Use | Safety | Flexibility | Best For |
|-----------|---------|--------|-------------|----------|
| VectorContextRetriever | None | High | Medium | General retrieval |
| TextToCypherRetriever | High | Low | High | Complex queries, trusted env |
| CypherTemplateRetriever | Medium | High | Low | Known patterns, production |

---

## Complete Example

Build PropertyGraphIndex with Neo4j, schema extraction, and hybrid retrieval:

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SchemaLLMPathExtractor,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 1. Setup stores
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
)

embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
llm = OpenAI(model="gpt-4-turbo")

# 2. Define schema
schema = {
    "PERSON": ["WORKS_AT", "FOUNDED"],
    "COMPANY": ["LOCATED_IN", "PRODUCES"],
    "PRODUCT": ["RELEASED_BY"],
}

# 3. Build index
docs = SimpleDirectoryReader("./data").load_data()

index = PropertyGraphIndex.from_documents(
    docs,
    property_graph_store=graph_store,
    embed_model=embed_model,
    kg_extractors=[
        ImplicitPathExtractor(),
        SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=list(schema.keys()),
            possible_relations=["WORKS_AT", "FOUNDED", "LOCATED_IN", "PRODUCES", "RELEASED_BY"],
            strict=True,
        ),
    ],
    show_progress=True,
)

# 4. Query with VectorContext
retriever = index.as_retriever(include_text=True, similarity_top_k=5)
nodes = retriever.retrieve("Who founded the company?")
```

---

## Schema Design Guidelines

1. **Use consistent relationship names**: Pick WORKS_AT or EMPLOYED_BY, not both
2. **Limit entity types**: 5-10 types for manageable graphs
3. **Include inverse relationships**: WORKS_AT ↔ EMPLOYS for bidirectional traversal
4. **Add temporal properties**: `{since: "2020"}` on edges for time-based queries
5. **Test with sample data**: Validate schema captures important relationships

---

## See Also

- [../SKILL.md](../SKILL.md) — Return to main skill overview
- [ingestion.md](ingestion.md) — Semantic chunking before graph construction
- [context-rag.md](context-rag.md) — Routing queries to graph vs. vector stores
- [observability.md](observability.md) — Debug graph retrieval issues
