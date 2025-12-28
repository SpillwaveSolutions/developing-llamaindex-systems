# Observability

Tracing, debugging, and evaluation for agentic systems.

## Contents

- [Instrumentation Module](#instrumentation-module)
  - [Dispatchers](#dispatchers)
  - [Spans and Events](#spans-and-events)
  - [Custom Handlers](#custom-handlers)
- [Arize Phoenix Integration](#arize-phoenix-integration)
  - [Setup](#setup)
  - [Trace Visualization](#trace-visualization)
- [Debugging Scenarios](#debugging-scenarios)
  - [Retrieval Failures](#retrieval-failures)
  - [Agent Loops](#agent-loops)
  - [Latency Issues](#latency-issues)
  - [Routing Problems](#routing-problems)
- [Evaluators](#evaluators)
  - [FaithfulnessEvaluator](#faithfulnessevaluator)
  - [RelevancyEvaluator](#relevancyevaluator)
  - [CorrectnessEvaluator](#correctnessevaluator)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Production Monitoring](#production-monitoring)
- [See Also](#see-also)

---

## Instrumentation Module

Replaces legacy callback system in LlamaIndex v0.10+. Provides deep visibility into LLM operations.

### Architecture

```
LlamaIndex Module → Dispatcher → [SpanHandler | EventHandler] → Output
```

### Dispatchers

Central hub that broadcasts events. Every module has its own dispatcher.

```python
from llama_index.core.instrumentation import get_dispatcher

# Get dispatcher for a specific module
dispatcher = get_dispatcher(__name__)
```

### Spans and Events

| Concept | Description | Example |
|---------|-------------|---------|
| **Span** | Duration of operation | "Retrieval took 250ms" |
| **Event** | Discrete point in time | "LLM prompt sent" |

Spans track:
- Retrieval operations
- LLM calls
- Embedding generation
- Tool execution

### Custom Handlers

Create custom handlers for logging, metrics, or external platforms.

#### Basic Logging Handler

```python
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler

class LoggingSpanHandler(BaseSpanHandler):
    def new_span(self, id, parent_span_id, **kwargs):
        print(f"Span started: {id}")

    def end_span(self, id, **kwargs):
        print(f"Span ended: {id}")

class LoggingEventHandler(BaseEventHandler):
    def handle(self, event):
        print(f"Event: {event.class_name()} - {event.dict()}")

# Register handlers
from llama_index.core.instrumentation import get_dispatcher
dispatcher = get_dispatcher()
dispatcher.add_span_handler(LoggingSpanHandler())
dispatcher.add_event_handler(LoggingEventHandler())
```

#### Metrics Collection Handler

```python
import time
from collections import defaultdict
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler

class MetricsSpanHandler(BaseSpanHandler):
    def __init__(self):
        super().__init__()
        self.span_starts = {}
        self.metrics = defaultdict(list)

    def new_span(self, id, parent_span_id, **kwargs):
        self.span_starts[id] = time.time()

    def end_span(self, id, **kwargs):
        if id in self.span_starts:
            duration = time.time() - self.span_starts[id]
            span_type = kwargs.get("span_type", "unknown")
            self.metrics[span_type].append(duration)
            del self.span_starts[id]

    def get_stats(self):
        return {
            span_type: {
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations) * 1000,
                "max_ms": max(durations) * 1000,
            }
            for span_type, durations in self.metrics.items()
        }

# Usage
metrics_handler = MetricsSpanHandler()
dispatcher.add_span_handler(metrics_handler)

# After some queries...
print(metrics_handler.get_stats())
# {"retrieval": {"count": 10, "avg_ms": 45.2, "max_ms": 120.1}, ...}
```

#### Token Usage Tracker

```python
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMCompletionEndEvent

class TokenTracker(BaseEventHandler):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @classmethod
    def class_name(cls) -> str:
        return "TokenTracker"

    def handle(self, event):
        if isinstance(event, LLMCompletionEndEvent):
            if hasattr(event, "token_counts"):
                self.total_prompt_tokens += event.token_counts.get("prompt", 0)
                self.total_completion_tokens += event.token_counts.get("completion", 0)

    def get_usage(self):
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

# Usage
token_tracker = TokenTracker()
dispatcher.add_event_handler(token_tracker)

# After queries...
print(token_tracker.get_usage())
```

#### Alerting Handler

```python
class AlertingEventHandler(BaseEventHandler):
    def __init__(self, latency_threshold_ms=5000):
        self.latency_threshold = latency_threshold_ms / 1000

    def handle(self, event):
        # Alert on slow retrievals
        if hasattr(event, "duration") and event.duration > self.latency_threshold:
            self.send_alert(f"Slow operation: {event.class_name()} took {event.duration:.2f}s")

        # Alert on errors
        if hasattr(event, "exception") and event.exception:
            self.send_alert(f"Error in {event.class_name()}: {event.exception}")

    def send_alert(self, message):
        # Integration: Slack, PagerDuty, email, etc.
        print(f"ALERT: {message}")
```

---

## Arize Phoenix Integration

Native observability platform adhering to OpenInference standard.

### Setup

One-line integration instruments entire LlamaIndex stack:

```python
import phoenix as px
import llama_index.core

# 1. Launch Phoenix server (local)
px.launch_app()

# 2. Set global handler
llama_index.core.set_global_handler("arize_phoenix")

# All subsequent operations are traced
response = query_engine.query("What is X?")
```

**Prerequisites:**
```bash
pip install arize-phoenix
```

Phoenix UI available at `http://localhost:6006` after launch.

### Cloud Phoenix

For production, use hosted Phoenix:

```python
import os
os.environ["PHOENIX_API_KEY"] = "your-api-key"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

import llama_index.core
llama_index.core.set_global_handler("arize_phoenix")
```

### Trace Visualization

Phoenix provides:

| View | Purpose |
|------|---------|
| **Trace Waterfall** | End-to-end request timeline |
| **Span Details** | Input/output for each operation |
| **Token Usage** | LLM token consumption |
| **Latency Breakdown** | Time per component |
| **Retrieval Inspector** | Retrieved chunks and scores |

---

## Debugging Scenarios

### Retrieval Failures

**Symptom:** Agent says "I don't know" despite relevant data existing.

**Debug Steps:**

1. Open Phoenix trace for the query
2. Find "Retrieval" span
3. Inspect retrieved chunks

**Common Causes:**

| Issue | Evidence in Phoenix | Solution |
|-------|---------------------|----------|
| Low similarity scores | All chunks < 0.5 | Improve embeddings or chunking |
| Wrong chunks retrieved | Chunks off-topic | Try SemanticSplitter |
| Too few chunks | Only 1-2 chunks | Increase `similarity_top_k` |
| Chunks too fragmented | Partial sentences | Increase chunk size |

**Fix Pattern:**

```python
# Before: fixed chunking
splitter = SentenceSplitter(chunk_size=256)

# After: semantic chunking
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)
```

### Agent Loops

**Symptom:** Agent keeps calling tools without reaching final answer.

**Debug Steps:**

1. Open Phoenix trace
2. Look for repeated "Action" spans
3. Check "Observation" payload for each

**Common Causes:**

| Issue | Evidence | Solution |
|-------|----------|----------|
| Tool returning errors | Observation: "Error: ..." | Fix tool implementation |
| Ambiguous tool output | Observation confuses agent | Improve tool return format |
| Missing stop condition | No "Final Answer" | Add explicit termination logic |
| Tool not answering query | Observation irrelevant | Improve tool description |

**Fix Pattern:**

```python
# Before: vague tool output
def search(query: str) -> str:
    results = db.search(query)
    return str(results)  # Raw object dump

# After: structured output
def search(query: str) -> str:
    """Search database and return formatted results."""
    results = db.search(query)
    if not results:
        return "No results found for this query."
    return json.dumps({
        "count": len(results),
        "results": results[:5],
        "summary": f"Found {len(results)} matches"
    })
```

### Latency Issues

**Symptom:** Queries taking too long (>10s).

**Debug Steps:**

1. Open Phoenix trace waterfall
2. Identify longest spans
3. Check component breakdown

**Common Causes:**

| Bottleneck | Evidence | Solution |
|------------|----------|----------|
| LLM calls | LLM spans dominate | Use faster model, reduce prompts |
| Embedding | Many embed calls | Batch embeddings, use local model |
| Retrieval | DB query slow | Add indices, reduce top_k |
| Reranking | Rerank span long | Use faster reranker, reduce candidates |

**Latency Optimization:**

```python
# Identify: Phoenix shows LLMRerank taking 3s

# Before
node_postprocessors=[
    LLMRerank(top_n=5)  # Slow
]

# After: two-stage
node_postprocessors=[
    SentenceTransformerRerank(top_n=10),  # Fast filter
    LLMRerank(top_n=3)  # Precise on smaller set
]
```

### Routing Problems

**Symptom:** Queries going to wrong engine in RouterQueryEngine.

**Debug Steps:**

1. Open Phoenix trace
2. Find "Router" span
3. Check selector decision

**Common Causes:**

| Issue | Evidence | Solution |
|-------|----------|----------|
| Vague descriptions | Selector confused | Rewrite tool descriptions |
| Overlapping descriptions | Wrong engine chosen | Make descriptions mutually exclusive |
| Missing description | Never selected | Add specific use cases |

**Fix Pattern:**

```python
# Before: vague
QueryEngineTool.from_defaults(
    query_engine=engine,
    description="Handles questions"  # Too generic
)

# After: specific
QueryEngineTool.from_defaults(
    query_engine=engine,
    description=(
        "Answers questions about product specifications, "
        "dimensions, materials, and technical details. "
        "Use for 'how big', 'what material', 'specs' queries."
    )
)
```

---

## Evaluators

LLM-as-judge evaluation of agent responses.

### FaithfulnessEvaluator

Checks if answer is derived solely from retrieved context (anti-hallucination).

```python
from llama_index.core.evaluation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator()

result = evaluator.evaluate_response(
    query="What is the return policy?",
    response=response,  # Agent response object
)

print(f"Faithful: {result.passing}")  # True/False
print(f"Score: {result.score}")       # 0.0-1.0
print(f"Feedback: {result.feedback}") # Explanation
```

**Use Case:** Detect when agent invents information not in documents.

### RelevancyEvaluator

Checks if answer actually addresses the user's query.

```python
from llama_index.core.evaluation import RelevancyEvaluator

evaluator = RelevancyEvaluator()

result = evaluator.evaluate_response(
    query="How do I reset my password?",
    response=response,
)

print(f"Relevant: {result.passing}")
```

**Use Case:** Detect tangential or off-topic responses.

### CorrectnessEvaluator

Grades response against a reference "gold standard" answer.

```python
from llama_index.core.evaluation import CorrectnessEvaluator

evaluator = CorrectnessEvaluator()

result = evaluator.evaluate(
    query="What is 2+2?",
    response="The answer is 4.",
    reference="4",  # Gold standard
)

print(f"Correct: {result.passing}")
print(f"Score: {result.score}")  # 1-5 scale
```

**Use Case:** Regression testing with known Q&A pairs.

---

## Evaluation Pipeline

Systematic evaluation across test dataset:

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
)

# Setup evaluators
faithfulness = FaithfulnessEvaluator()
relevancy = RelevancyEvaluator()

# Test queries
test_queries = [
    "What is the return policy?",
    "How do I contact support?",
    "What are the shipping options?",
]

# Run batch evaluation
runner = BatchEvalRunner(
    evaluators={
        "faithfulness": faithfulness,
        "relevancy": relevancy,
    },
    workers=4,
)

# Generate responses and evaluate
responses = [query_engine.query(q) for q in test_queries]
eval_results = await runner.aevaluate_responses(
    queries=test_queries,
    responses=responses,
)

# Aggregate results
for metric, results in eval_results.items():
    scores = [r.score for r in results]
    print(f"{metric}: {sum(scores)/len(scores):.2f} avg")
```

---

## Production Monitoring

### Key Metrics

| Metric | What to Track | Alert Threshold |
|--------|---------------|-----------------|
| Latency P95 | 95th percentile response time | > 10s |
| Faithfulness | % responses grounded in context | < 90% |
| Relevancy | % responses addressing query | < 85% |
| Token Usage | Tokens per request | > 10K |
| Error Rate | Failed requests | > 1% |

### Continuous Evaluation

```python
import asyncio
from datetime import datetime

async def monitor_loop(query_engine, evaluator, sample_queries):
    while True:
        for query in sample_queries:
            response = query_engine.query(query)
            result = evaluator.evaluate_response(query=query, response=response)
            
            # Log to monitoring system
            log_metric("faithfulness", result.score, timestamp=datetime.now())
            
            if not result.passing:
                alert(f"Faithfulness failure: {query}")
        
        await asyncio.sleep(300)  # Every 5 minutes
```

### Phoenix + Grafana

Export Phoenix metrics to Grafana for dashboards:

```python
# Phoenix exports OpenTelemetry format
# Configure OTLP exporter to send to Grafana Cloud
import os
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://otlp.grafana.com"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "Authorization=Basic ..."
```

---

## See Also

- [../SKILL.md](../SKILL.md) — Return to main skill overview
- [ingestion.md](ingestion.md) — Debug chunking issues identified in traces
- [context-rag.md](context-rag.md) — Debug routing decisions
- [orchestration.md](orchestration.md) — Trace workflow steps and tool calls
