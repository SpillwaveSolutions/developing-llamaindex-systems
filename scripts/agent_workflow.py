#!/usr/bin/env python3
"""
Event-Driven Agent Workflow for LlamaIndex Agentic Systems

Demonstrates Workflow pattern with observability via Arize Phoenix.

Usage:
    python agent_workflow.py
    python agent_workflow.py --storage ./storage --query "What is X?"
    python agent_workflow.py --no-phoenix  # Disable observability

Configuration:
    Modify the CONFIG section below or use command-line arguments.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIG - Modify these values or override via CLI
# ============================================================================

CONFIG = {
    # Index storage location (from ingest_semantic.py)
    "persist_dir": "./storage",
    
    # Query settings
    "similarity_top_k": 5,
    "include_text": True,
    
    # Workflow settings
    "timeout": 60,
    "verbose": True,
    
    # Observability
    "enable_phoenix": True,
}

# ============================================================================
# IMPORTS
# ============================================================================

def check_imports():
    """Verify required packages are installed."""
    required = [
        "llama_index.core",
        "llama_index.embeddings.openai",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        sys.exit(1)

check_imports()

import nest_asyncio
nest_asyncio.apply()  # Enable nested async for Jupyter/REPL compatibility

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Event
from llama_index.embeddings.openai import OpenAIEmbedding

# ============================================================================
# EVENTS
# ============================================================================

class QueryEvent(Event):
    """Carries the user query for processing."""
    query: str


class RetrievalEvent(Event):
    """Carries query with retrieved context."""
    query: str
    context: str
    num_nodes: int


class ClassificationEvent(Event):
    """Carries query classification result."""
    query: str
    category: str  # "factual", "analytical", "conversational"


# ============================================================================
# WORKFLOW
# ============================================================================

class AgenticSkillWorkflow(Workflow):
    """
    Event-driven agent workflow with retrieval and response generation.
    
    Flow:
        StartEvent → ClassificationEvent → RetrievalEvent → StopEvent
    
    Extend by:
        - Adding new Event types
        - Adding new @step methods
        - Implementing branching logic in classify step
    """
    
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.query_engine = index.as_query_engine(
            similarity_top_k=CONFIG["similarity_top_k"],
            include_text=CONFIG["include_text"],
        )
    
    @step
    async def classify(self, ev: StartEvent) -> ClassificationEvent:
        """
        Step 1: Classify the incoming query.
        
        Extend this step to implement routing logic for different query types.
        """
        query = ev.get("query")
        
        # Simple classification logic (extend as needed)
        query_lower = query.lower()
        if any(word in query_lower for word in ["how many", "what is", "who is", "when"]):
            category = "factual"
        elif any(word in query_lower for word in ["compare", "analyze", "explain why"]):
            category = "analytical"
        else:
            category = "conversational"
        
        if self._verbose:
            print(f"[Classify] Query: '{query[:50]}...' → Category: {category}")
        
        return ClassificationEvent(query=query, category=category)
    
    @step
    async def retrieve(self, ev: ClassificationEvent) -> RetrievalEvent:
        """
        Step 2: Retrieve relevant context from the index.
        
        Extend this step to:
        - Adjust retrieval based on category
        - Add reranking
        - Implement hybrid retrieval
        """
        query = ev.query
        category = ev.category
        
        if self._verbose:
            print(f"[Retrieve] Searching for: '{query[:50]}...'")
        
        # Retrieve nodes
        retriever = self.index.as_retriever(
            similarity_top_k=CONFIG["similarity_top_k"],
        )
        nodes = retriever.retrieve(query)
        
        # Format context
        context_parts = []
        for i, node in enumerate(nodes):
            score = getattr(node, 'score', 'N/A')
            text = node.get_content()[:500]  # Truncate for display
            context_parts.append(f"[{i+1}] (score: {score:.3f})\n{text}")
        
        context = "\n\n".join(context_parts)
        
        if self._verbose:
            print(f"[Retrieve] Found {len(nodes)} relevant chunks")
        
        return RetrievalEvent(
            query=query,
            context=context,
            num_nodes=len(nodes),
        )
    
    @step
    async def respond(self, ev: RetrievalEvent) -> StopEvent:
        """
        Step 3: Generate response using retrieved context.
        
        Extend this step to:
        - Add response validation
        - Implement response refinement loops
        - Add citation formatting
        """
        if self._verbose:
            print(f"[Respond] Generating response with {ev.num_nodes} context chunks...")
        
        # Use query engine for response generation
        response = self.query_engine.query(ev.query)
        
        result = {
            "response": str(response),
            "num_sources": ev.num_nodes,
            "query": ev.query,
        }
        
        return StopEvent(result=result)


# ============================================================================
# OBSERVABILITY
# ============================================================================

def setup_phoenix():
    """Initialize Arize Phoenix for observability."""
    try:
        import phoenix as px
        import llama_index.core
        
        # Launch Phoenix app
        session = px.launch_app()
        print(f"Phoenix UI available at: {session.url}")
        
        # Set global handler
        llama_index.core.set_global_handler("arize_phoenix")
        print("Phoenix observability enabled - all operations will be traced")
        
        return True
    except ImportError:
        print("Warning: arize-phoenix not installed, observability disabled")
        print("Install with: pip install arize-phoenix")
        return False
    except Exception as e:
        print(f"Warning: Failed to initialize Phoenix: {e}")
        return False


# ============================================================================
# INDEX LOADING
# ============================================================================

def load_index(persist_dir: str):
    """Load persisted index from storage."""
    if not Path(persist_dir).exists():
        print(f"Error: Index not found at {persist_dir}")
        print("Run ingest_semantic.py first to build the index")
        sys.exit(1)
    
    print(f"Loading index from: {persist_dir}")
    
    # Set embedding model (must match ingestion)
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    
    print("Index loaded successfully")
    return index


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

async def interactive_loop(agent: AgenticSkillWorkflow):
    """Run interactive query loop."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter queries (type 'quit' or 'exit' to stop)")
    print()
    
    while True:
        try:
            query = input("Query> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            print()
            result = await agent.run(query=query)
            
            print("\n" + "-" * 40)
            print("RESPONSE:")
            print("-" * 40)
            print(result["response"])
            print(f"\n(Sources: {result['num_sources']} chunks)")
            print()
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Goodbye!")
            break
        except ConnectionError as e:
            print(f"Connection error (check network/API key): {e}")
            print()
        except TimeoutError as e:
            print(f"Request timed out: {e}")
            print("Try increasing --timeout or simplifying query")
            print()
        except ValueError as e:
            print(f"Invalid input: {e}")
            print()
        except Exception as e:
            print(f"Unexpected error ({type(e).__name__}): {e}")
            print("Check Phoenix traces for details if enabled")
            print()


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Event-driven agent workflow with observability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_workflow.py
    python agent_workflow.py --query "What are the key concepts?"
    python agent_workflow.py --storage ./my_index
    python agent_workflow.py --no-phoenix
        """
    )
    
    parser.add_argument(
        "--storage",
        default=CONFIG["persist_dir"],
        help=f"Index storage directory (default: {CONFIG['persist_dir']})"
    )
    parser.add_argument(
        "--query",
        help="Single query to run (omit for interactive mode)"
    )
    parser.add_argument(
        "--no-phoenix",
        action="store_true",
        help="Disable Phoenix observability"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose workflow logging"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=CONFIG["timeout"],
        help=f"Workflow timeout in seconds (default: {CONFIG['timeout']})"
    )
    
    return parser.parse_args()


async def main_async(args):
    """Async main entry point."""
    # Load index
    index = load_index(args.storage)
    
    # Create workflow
    agent = AgenticSkillWorkflow(
        index=index,
        timeout=args.timeout,
        verbose=not args.quiet,
    )
    
    if args.query:
        # Single query mode
        print(f"\nProcessing query: {args.query}\n")
        result = await agent.run(query=args.query)
        
        print("=" * 60)
        print("RESPONSE")
        print("=" * 60)
        print(result["response"])
        print(f"\n(Sources: {result['num_sources']} chunks)")
    else:
        # Interactive mode
        await interactive_loop(agent)


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("LLAMAINDEX AGENTIC WORKFLOW")
    print("=" * 60)
    
    # Verify API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Setup observability
    if not args.no_phoenix and CONFIG["enable_phoenix"]:
        setup_phoenix()
    
    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
