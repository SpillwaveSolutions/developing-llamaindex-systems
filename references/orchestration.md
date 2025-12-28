# Agentic Orchestration

ReAct agents, function calling, event-driven Workflows, and multi-agent patterns.

## Contents

- [ReAct Agent Pattern](#react-agent-pattern) — Thought→Action→Observation loop
  - [Function Calling vs Text Parsing](#function-calling-vs-text-parsing) — Modern structured approach
  - [Parallel Tool Calls](#parallel-tool-calls) — Concurrent execution
- [Tools](#tools) — Wrapping functions for agent use
  - [FunctionTool](#functiontool) — Custom function wrapper + docstring best practices
  - [ToolSpecs](#toolspecs) — Pre-packaged tool bundles from LlamaHub
- [Workflows](#workflows) — Event-driven finite state machine (v0.10+)
  - [Core Components](#core-components) — Workflow, Event, @step, Context
  - [Events](#events) — Custom event definitions
  - [Steps](#steps) — Async methods with type-hint routing
  - [Context](#context) — Shared state across steps
  - [Basic Workflow](#basic-workflow) — Minimal working example
- [Advanced Patterns](#advanced-patterns) — Beyond simple loops
  - [Branching](#branching) — Conditional routing
  - [Loops](#loops) — Iterative refinement
  - [Human-in-the-Loop](#human-in-the-loop) — Approval flows
  - [Concierge Multi-Agent](#concierge-multi-agent) — Specialized sub-agents
- [Pattern Comparison](#pattern-comparison) — Decision guide
- [Complete Examples](#complete-examples) — Production-ready code
- [See Also](#see-also) — Related references

---

## ReAct Agent Pattern

Default architecture for tool-using agents. Reasoning and Acting in a loop.

### Mechanism

```
User Query → [Thought → Action → Observation] → ... → Final Answer
```

1. **Thought**: Agent analyzes input, decides if tool needed
2. **Action**: Agent generates tool call
3. **Observation**: Tool executes, returns result
4. **Repeat**: Agent analyzes observation, continues or answers

### Function Calling vs Text Parsing

#### Text Parsing (Legacy)

Agent outputs text like `Action: search("query")`, parsed via regex.

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,
)
```

**Drawbacks:** Parsing errors, prompt engineering required.

#### Function Calling (Modern)

LLM outputs structured JSON matching tool schema. Native support in GPT-4, Mistral.

```python
from llama_index.core.agent import FunctionCallingAgent

agent = FunctionCallingAgent.from_tools(
    tools,
    llm=llm,  # Must support function calling
    verbose=True,
)
```

**Benefits:** No parsing errors, reliable tool invocation.

### Parallel Tool Calls

LLM can request multiple tools in single turn. System executes in parallel threads.

```python
# Query: "Get weather for Tokyo and New York"
# LLM emits two tool calls simultaneously:
#   - get_weather(city="Tokyo")
#   - get_weather(city="New York")
# Both execute in parallel, results aggregated
```

#### Enabling Parallel Execution

```python
from llama_index.core.agent import FunctionCallingAgent

# Parallel calls enabled by default
agent = FunctionCallingAgent.from_tools(
    tools,
    llm=llm,
    allow_parallel_tool_calls=True,  # Default: True
    verbose=True,
)

# Query triggers parallel execution
response = agent.chat("Compare weather in Tokyo, London, and Sydney")
# Executes 3 get_weather calls in parallel
```

#### Controlling Parallelism

```python
# Disable if tools have dependencies or side effects
agent = FunctionCallingAgent.from_tools(
    tools,
    llm=llm,
    allow_parallel_tool_calls=False,  # Sequential execution
)

# Or control at tool level with dependencies
def get_user_orders(user_id: str) -> str:
    """Get orders. Requires user_id from get_current_user first."""
    ...
```

#### Parallel in Workflows

```python
from llama_index.core.workflow import Workflow, step, Event

class ParallelSearchEvent(Event):
    queries: list[str]

class SearchResultEvent(Event):
    query: str
    results: list[str]

class ParallelWorkflow(Workflow):

    @step
    async def parallel_search(self, ev: ParallelSearchEvent) -> list[SearchResultEvent]:
        # Emit multiple events - all processed in parallel
        events = []
        for query in ev.queries:
            events.append(SearchResultEvent(query=query, results=self.search(query)))
        return events  # Return list = parallel emission
```

---

## Tools

### FunctionTool

Wraps any Python function as an agent tool.

```python
from llama_index.core.tools import FunctionTool

def search_database(query: str, limit: int = 10) -> str:
    """
    Search the product database.
    
    Args:
        query: Search terms
        limit: Maximum results to return
    
    Returns:
        JSON string of matching products
    """
    # Implementation
    results = db.search(query, limit=limit)
    return json.dumps(results)

tool = FunctionTool.from_defaults(fn=search_database)
```

**Critical:** The docstring becomes the tool description. Write verbose, clear docstrings.

#### Docstring Best Practices

```python
def calculate_shipping(
    weight_kg: float,
    destination: str,
    express: bool = False
) -> str:
    """
    Calculate shipping cost for a package.
    
    Use this tool when the user asks about shipping costs,
    delivery prices, or postage for sending items.
    
    Args:
        weight_kg: Package weight in kilograms
        destination: Destination country code (e.g., "US", "UK")
        express: Whether to use express shipping (2-day vs 7-day)
    
    Returns:
        JSON with cost breakdown and estimated delivery date
    """
    ...
```

### ToolSpecs

Pre-packaged tool bundles from LlamaHub.

```python
from llama_index.tools.google import GmailToolSpec

# Load entire Gmail capability
gmail_spec = GmailToolSpec()
tools = gmail_spec.to_tool_list()

# Tools include: search_messages, create_draft, send_email, etc.
agent = FunctionCallingAgent.from_tools(tools, llm=llm)
```

**Available ToolSpecs:**
- `GmailToolSpec` — Email operations
- `GoogleCalendarToolSpec` — Calendar management
- `SlackToolSpec` — Slack messaging
- `NotionToolSpec` — Notion pages/databases
- `WikipediaToolSpec` — Wikipedia search

Install: `pip install llama-index-tools-google`

---

## Workflows

Event-driven finite state machine. Introduced in LlamaIndex v0.10.

### Why Workflows?

ReAct limitations:
- Linear loop only
- No branching or cycles
- Hard to suspend/resume
- Complex state management

Workflows enable:
- Branching logic
- Cyclic flows (loops)
- Human-in-the-loop
- Multi-agent coordination
- Clean state management

### Core Components

| Component | Purpose |
|-----------|---------|
| `Workflow` | Class encapsulating agent logic and state |
| `Event` | Pydantic object for inter-step data passing |
| `@step` | Decorator marking async methods as workflow steps |
| `Context` | Global state shared across all steps |
| `StartEvent` | Built-in event that triggers workflow |
| `StopEvent` | Built-in event that ends workflow |

### Events

Define custom events for data passing:

```python
from llama_index.core.workflow import Event

class QueryEvent(Event):
    """Carries the user query for processing."""
    query: str

class RetrievalEvent(Event):
    """Carries query with retrieved context."""
    query: str
    context: list[str]

class ClassificationEvent(Event):
    """Carries query classification result."""
    query: str
    category: str  # "factual", "analytical", "conversational"
```

### Steps

Steps listen for specific event types (via input type hint):

```python
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

class MyWorkflow(Workflow):
    
    @step
    async def classify(self, ev: StartEvent) -> QueryEvent:
        """First step: receives StartEvent."""
        query = ev.get("query")
        return QueryEvent(query=query)
    
    @step
    async def retrieve(self, ev: QueryEvent) -> RetrievalEvent:
        """Triggered by QueryEvent."""
        context = self.retriever.retrieve(ev.query)
        return RetrievalEvent(query=ev.query, context=context)
    
    @step
    async def respond(self, ev: RetrievalEvent) -> StopEvent:
        """Final step: emits StopEvent to end workflow."""
        response = self.llm.complete(f"Context: {ev.context}\nQuery: {ev.query}")
        return StopEvent(result=str(response))
```

### Context

Shared state accessible from all steps:

```python
from llama_index.core.workflow import Context

class StatefulWorkflow(Workflow):
    
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> NextEvent:
        # Store in context
        await ctx.set("user_id", ev.get("user_id"))
        await ctx.set("history", [])
        return NextEvent()
    
    @step
    async def step_two(self, ctx: Context, ev: NextEvent) -> StopEvent:
        # Retrieve from context
        user_id = await ctx.get("user_id")
        history = await ctx.get("history")
        return StopEvent(result=f"User {user_id}")
```

### Basic Workflow

```python
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Event

class QueryEvent(Event):
    query: str

class SimpleAgent(Workflow):
    def __init__(self, query_engine, **kwargs):
        super().__init__(**kwargs)
        self.query_engine = query_engine
    
    @step
    async def route(self, ev: StartEvent) -> QueryEvent:
        return QueryEvent(query=ev.get("query"))
    
    @step
    async def answer(self, ev: QueryEvent) -> StopEvent:
        response = self.query_engine.query(ev.query)
        return StopEvent(result=str(response))

# Run
async def main():
    agent = SimpleAgent(query_engine=engine, timeout=60, verbose=True)
    result = await agent.run(query="What is the capital of France?")
    print(result)
```

---

## Advanced Patterns

### Branching

One step emits different events based on conditions:

```python
class ClassifyEvent(Event):
    query: str
    category: str

class TechnicalEvent(Event):
    query: str

class GeneralEvent(Event):
    query: str

class BranchingWorkflow(Workflow):
    
    @step
    async def classify(self, ev: StartEvent) -> TechnicalEvent | GeneralEvent:
        query = ev.get("query")
        
        # Classification logic
        if "code" in query.lower() or "error" in query.lower():
            return TechnicalEvent(query=query)
        else:
            return GeneralEvent(query=query)
    
    @step
    async def handle_technical(self, ev: TechnicalEvent) -> StopEvent:
        response = self.technical_engine.query(ev.query)
        return StopEvent(result=str(response))
    
    @step
    async def handle_general(self, ev: GeneralEvent) -> StopEvent:
        response = self.general_engine.query(ev.query)
        return StopEvent(result=str(response))
```

### Loops

Step emits event that triggers itself or earlier step:

```python
class RefineEvent(Event):
    query: str
    response: str
    iterations: int

class LoopingWorkflow(Workflow):
    
    @step
    async def generate(self, ev: StartEvent | RefineEvent) -> RefineEvent | StopEvent:
        if isinstance(ev, StartEvent):
            query = ev.get("query")
            iterations = 0
        else:
            query = ev.query
            iterations = ev.iterations
        
        response = self.llm.complete(query)
        
        # Check quality
        if self.is_good_enough(response) or iterations >= 3:
            return StopEvent(result=str(response))
        else:
            return RefineEvent(
                query=f"Improve this: {response}",
                response=str(response),
                iterations=iterations + 1
            )
```

### Human-in-the-Loop

Suspend workflow until external input using built-in events.

#### Built-in Events

| Event | Purpose | Key Fields |
|-------|---------|------------|
| `InputRequiredEvent` | Request human input | `prefix` (prompt), `payload` (context) |
| `HumanResponseEvent` | Carry human response | `response` (string input) |

#### Basic Pattern

```python
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent

class ApprovalWorkflow(Workflow):

    @step
    async def propose(self, ev: StartEvent) -> InputRequiredEvent:
        proposal = self.generate_proposal(ev.get("task"))
        # Suspend and wait for human
        return InputRequiredEvent(
            prefix="Please approve this proposal:",
            payload=proposal
        )

    @step
    async def execute(self, ev: HumanResponseEvent) -> StopEvent:
        if ev.response.lower() == "approved":
            result = self.execute_proposal()
            return StopEvent(result=result)
        else:
            return StopEvent(result="Proposal rejected")
```

#### Running with Human Input

```python
async def run_with_approval():
    workflow = ApprovalWorkflow(timeout=300)  # Longer timeout for human

    # Start workflow - will suspend at InputRequiredEvent
    handler = workflow.run(task="Deploy to production")

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            print(f"{event.prefix}")
            print(f"Context: {event.payload}")

            # Get human input (from UI, CLI, etc.)
            user_input = input("Your response: ")

            # Resume workflow with human response
            handler.ctx.send_event(HumanResponseEvent(response=user_input))

    result = await handler
    return result
```

#### Multi-Step Approval

```python
class MultiApprovalWorkflow(Workflow):

    @step
    async def review_step(self, ev: StartEvent | HumanResponseEvent) -> InputRequiredEvent | StopEvent:
        ctx = self.ctx

        if isinstance(ev, StartEvent):
            # First review
            await ctx.set("reviews", [])
            await ctx.set("current_reviewer", 0)
            reviewers = ["Manager", "Security", "Legal"]
            await ctx.set("reviewers", reviewers)
        else:
            # Process previous response
            reviews = await ctx.get("reviews")
            reviews.append(ev.response)
            await ctx.set("reviews", reviews)

        reviewers = await ctx.get("reviewers")
        current = await ctx.get("current_reviewer")

        if current < len(reviewers):
            await ctx.set("current_reviewer", current + 1)
            return InputRequiredEvent(
                prefix=f"Awaiting {reviewers[current]} approval:",
                payload=await ctx.get("reviews")
            )
        else:
            return StopEvent(result={"approvals": await ctx.get("reviews")})
```

### Concierge Multi-Agent

Entry agent routes to specialized sub-agents:

```python
class TravelEvent(Event):
    query: str

class SupportEvent(Event):
    query: str

class ConciergeWorkflow(Workflow):
    
    @step
    async def concierge(self, ev: StartEvent) -> TravelEvent | SupportEvent | StopEvent:
        query = ev.get("query")
        intent = self.classify_intent(query)
        
        if intent == "travel":
            return TravelEvent(query=query)
        elif intent == "support":
            return SupportEvent(query=query)
        else:
            # Handle directly
            return StopEvent(result=self.general_response(query))
    
    @step
    async def travel_agent(self, ev: TravelEvent) -> StopEvent:
        # Specialized travel handling
        result = self.travel_engine.query(ev.query)
        return StopEvent(result=str(result))
    
    @step
    async def support_agent(self, ev: SupportEvent) -> StopEvent:
        # Specialized support handling
        result = self.support_engine.query(ev.query)
        return StopEvent(result=str(result))
```

---

## Pattern Comparison

| Pattern | Use Case | Complexity |
|---------|----------|------------|
| ReAct Agent | Simple tool loops | Low |
| Linear Workflow | Deterministic pipelines | Low |
| Branching Workflow | Conditional routing | Medium |
| Looping Workflow | Iterative refinement | Medium |
| Human-in-the-Loop | Approval flows | Medium |
| Concierge Multi-Agent | Specialized sub-agents | High |

### Decision Guide

```
Simple tool usage?
├─ Yes → FunctionCallingAgent (ReAct)
│
└─ No, need:
    ├─ Conditional logic → Branching Workflow
    ├─ Iteration/refinement → Looping Workflow
    ├─ Human approval → Human-in-the-Loop Workflow
    ├─ Multiple specialists → Concierge Workflow
    └─ All of the above → Compose patterns in single Workflow
```

---

## Complete Examples

### Production Agent with Tools and Workflow

```python
import asyncio
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Event
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Define tools
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return engine.query(query)

def search_web(query: str) -> str:
    """Search the web for current information."""
    return web_search(query)

tools = [
    FunctionTool.from_defaults(fn=search_docs),
    FunctionTool.from_defaults(fn=search_web),
]

# Workflow with tool selection
class ToolEvent(Event):
    query: str
    tool_name: str

class SmartAgent(Workflow):
    def __init__(self, tools, llm, **kwargs):
        super().__init__(**kwargs)
        self.tools = {t.metadata.name: t for t in tools}
        self.llm = llm
    
    @step
    async def select_tool(self, ev: StartEvent) -> ToolEvent | StopEvent:
        query = ev.get("query")
        # LLM selects tool
        selection = self.llm.complete(
            f"Select tool for: {query}\nTools: {list(self.tools.keys())}"
        )
        tool_name = str(selection).strip()
        
        if tool_name in self.tools:
            return ToolEvent(query=query, tool_name=tool_name)
        else:
            return StopEvent(result="No suitable tool found")
    
    @step
    async def execute_tool(self, ev: ToolEvent) -> StopEvent:
        tool = self.tools[ev.tool_name]
        result = tool.call(ev.query)
        return StopEvent(result=result)

# Run
async def main():
    agent = SmartAgent(tools=tools, llm=OpenAI(), timeout=60)
    result = await agent.run(query="Find our refund policy")
    print(result)

asyncio.run(main())
```

---

## See Also

- [../SKILL.md](../SKILL.md) — Return to main skill overview
- [context-rag.md](context-rag.md) — Query engines to use as tools
- [property-graphs.md](property-graphs.md) — Graph retrieval in agent tools
- [observability.md](observability.md) — Trace agent steps and tool calls
