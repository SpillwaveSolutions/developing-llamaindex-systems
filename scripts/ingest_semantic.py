#!/usr/bin/env python3
"""
Semantic Ingestion Pipeline for LlamaIndex Agentic Systems

Builds a PropertyGraphIndex with semantic chunking and optional graph extraction.

Usage:
    python ingest_semantic.py --doc path/to/document.pdf
    python ingest_semantic.py --dir path/to/documents/
    python ingest_semantic.py --doc data.pdf --extractor schema --persist ./storage

Configuration:
    Modify the CONFIG section below or use command-line arguments.
"""

import argparse
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
    # Embedding model
    "embed_model_name": "text-embedding-3-small",
    
    # Semantic splitter settings
    "buffer_size": 1,                      # 1=sensitive, 3=stable
    "breakpoint_percentile_threshold": 95, # 95=fewer chunks, 70=more chunks
    
    # Graph extractor: "implicit", "simple", "schema", or "none"
    "extractor_type": "simple",
    "max_paths_per_chunk": 10,
    
    # Schema for SchemaLLMPathExtractor (used when extractor_type="schema")
    "schema_entities": ["PERSON", "COMPANY", "PRODUCT", "LOCATION"],
    "schema_relations": ["WORKS_AT", "FOUNDED", "LOCATED_IN", "PRODUCES", "RELEASED_BY"],
    
    # LLM for extraction
    "llm_model": "gpt-4-turbo",
    
    # Persistence
    "persist_dir": "./storage",
}

# ============================================================================
# IMPORTS
# ============================================================================

def check_imports():
    """Verify required packages are installed."""
    required = [
        "llama_index.core",
        "llama_index.embeddings.openai",
        "llama_index.llms.openai",
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

from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    StorageContext,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
    SchemaLLMPathExtractor,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def create_embed_model(model_name: str) -> OpenAIEmbedding:
    """Create embedding model."""
    print(f"Initializing embedding model: {model_name}")
    return OpenAIEmbedding(model_name=model_name)


def create_splitter(
    embed_model: OpenAIEmbedding,
    buffer_size: int,
    threshold: int
) -> SemanticSplitterNodeParser:
    """Create semantic splitter with specified parameters."""
    print(f"Creating semantic splitter (buffer={buffer_size}, threshold={threshold})")
    return SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=threshold,
        embed_model=embed_model,
    )


def create_extractors(
    extractor_type: str,
    llm: OpenAI,
    config: dict
) -> list:
    """Create graph extractors based on type."""
    extractors = []
    
    # Always include implicit for document structure
    if extractor_type != "none":
        extractors.append(ImplicitPathExtractor())
        print("Added ImplicitPathExtractor (document structure)")
    
    if extractor_type == "simple":
        extractors.append(
            SimpleLLMPathExtractor(
                llm=llm,
                max_paths_per_chunk=config["max_paths_per_chunk"],
            )
        )
        print(f"Added SimpleLLMPathExtractor (max_paths={config['max_paths_per_chunk']})")
    
    elif extractor_type == "schema":
        extractors.append(
            SchemaLLMPathExtractor(
                llm=llm,
                possible_entities=config["schema_entities"],
                possible_relations=config["schema_relations"],
                strict=True,
            )
        )
        print(f"Added SchemaLLMPathExtractor (entities={config['schema_entities']})")
    
    elif extractor_type == "implicit":
        pass  # Already added above
    
    elif extractor_type == "none":
        print("No graph extractors (vector-only index)")
    
    return extractors


def load_documents(doc_path: str = None, dir_path: str = None) -> list:
    """Load documents from file or directory."""
    if doc_path:
        if not Path(doc_path).exists():
            print(f"Error: File not found: {doc_path}")
            sys.exit(1)
        print(f"Loading document: {doc_path}")
        reader = SimpleDirectoryReader(input_files=[doc_path])
    elif dir_path:
        if not Path(dir_path).is_dir():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)
        print(f"Loading documents from: {dir_path}")
        reader = SimpleDirectoryReader(input_dir=dir_path)
    else:
        print("Error: Must specify --doc or --dir")
        sys.exit(1)
    
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document(s)")
    return documents


def build_index(
    documents: list,
    embed_model: OpenAIEmbedding,
    extractors: list,
    splitter: SemanticSplitterNodeParser,
) -> PropertyGraphIndex:
    """Build PropertyGraphIndex with semantic chunking and graph extraction."""
    print("\nBuilding PropertyGraphIndex...")
    
    try:
        print("  - Semantic chunking in progress...")
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        print(f"  - Created {len(nodes)} semantic chunks")
        
        if not nodes:
            raise ValueError("No nodes created - check document content")
        
        print("  - Extracting graph relationships...")
        index = PropertyGraphIndex(
            nodes=nodes,
            embed_model=embed_model,
            kg_extractors=extractors if extractors else None,
            show_progress=True,
        )
        
        return index
        
    except ConnectionError as e:
        print(f"Error: Failed to connect to embedding API: {e}")
        print("Check OPENAI_API_KEY and network connection")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error building index ({type(e).__name__}): {e}")
        sys.exit(1)


def persist_index(index: PropertyGraphIndex, persist_dir: str):
    """Save index to disk."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"\nIndex persisted to: {persist_dir}")


def print_summary(index: PropertyGraphIndex, persist_dir: str):
    """Print index summary."""
    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Storage location: {persist_dir}")
    print("\nTo load this index:")
    print(f"""
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="{persist_dir}")
index = load_index_from_storage(storage_context)

# Create query engine
query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("Your question here")
""")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build PropertyGraphIndex with semantic chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ingest_semantic.py --doc report.pdf
    python ingest_semantic.py --dir ./documents --extractor schema
    python ingest_semantic.py --doc data.pdf --buffer 3 --threshold 85
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--doc", help="Path to single document")
    input_group.add_argument("--dir", help="Path to document directory")
    
    # Splitter config
    parser.add_argument(
        "--buffer",
        type=int,
        default=CONFIG["buffer_size"],
        help=f"Semantic splitter buffer size (default: {CONFIG['buffer_size']})"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=CONFIG["breakpoint_percentile_threshold"],
        help=f"Breakpoint threshold 0-100 (default: {CONFIG['breakpoint_percentile_threshold']})"
    )
    
    # Extractor config
    parser.add_argument(
        "--extractor",
        choices=["implicit", "simple", "schema", "none"],
        default=CONFIG["extractor_type"],
        help=f"Graph extractor type (default: {CONFIG['extractor_type']})"
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=CONFIG["max_paths_per_chunk"],
        help=f"Max paths per chunk for LLM extractors (default: {CONFIG['max_paths_per_chunk']})"
    )
    
    # Output
    parser.add_argument(
        "--persist",
        default=CONFIG["persist_dir"],
        help=f"Directory to save index (default: {CONFIG['persist_dir']})"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Update config with CLI args
    config = CONFIG.copy()
    config["buffer_size"] = args.buffer
    config["breakpoint_percentile_threshold"] = args.threshold
    config["extractor_type"] = args.extractor
    config["max_paths_per_chunk"] = args.max_paths
    config["persist_dir"] = args.persist
    
    print("=" * 60)
    print("LLAMAINDEX SEMANTIC INGESTION")
    print("=" * 60)
    
    # Verify API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    # Initialize components
    embed_model = create_embed_model(config["embed_model_name"])
    llm = OpenAI(model=config["llm_model"])
    splitter = create_splitter(
        embed_model,
        config["buffer_size"],
        config["breakpoint_percentile_threshold"]
    )
    extractors = create_extractors(config["extractor_type"], llm, config)
    
    # Load documents
    documents = load_documents(doc_path=args.doc, dir_path=args.dir)
    
    # Build and persist index
    index = build_index(documents, embed_model, extractors, splitter)
    persist_index(index, config["persist_dir"])
    print_summary(index, config["persist_dir"])


if __name__ == "__main__":
    main()
