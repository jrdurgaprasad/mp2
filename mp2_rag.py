"""MP2 · Mini-RAG — Starter Template
====================================

You'll build a complete RAG pipeline over the Sherlock Holmes corpus in this
file. Fill in every TODO. The reference solution is ~250 lines, but yours can
be shorter or longer — what matters is that it works end-to-end.

Pipeline you're building:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence (once you've filled in the TODOs):
    pip install -r requirements.txt
    source .env                 # exports your OpenAI + Qdrant credentials
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl

Tip: get the CORE pipeline working FIRST (Steps 1-7 below), THEN come back to
polish and add your 3 questions. Don't try to perfect each step before moving
on — you'll learn more from a rough end-to-end loop than a polished half.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv(Path(__file__).with_name(".env"))

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters

openai = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE_URL"]
)
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY"),
)

# ─── Logging ───────────────────────────────────────────────────────────

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with standard format."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger(__name__)

# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).
    """
    # TODO: your code here
    # Empty list to hold the dictionaries of each file content
    list_of_files = []

    # Iterate over each .txt file in the input directory path of corpus and append to above list_of_files list
    for path in corpus_dir.glob("*.txt"):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("Skipping empty file: %s", path.name)
            continue

        title = next(
            (line.strip() for line in text.splitlines() if line.strip()),
            ""
        )

        list_of_files.append(
            {
                "source": path.name,
                "title": title,
                "text": text
            }
        )
    return list_of_files


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: source, title, section, text.
    """
    # TODO: your code here
    chunks = []
    text = doc['text']
    if not text:
        logger.warning("Received empty text for chunking from source: %s", doc.get('source', 'Unknown'))
        return chunks
    
    start = 0
    text_length = len(text) if text else 0
    
    try:
        while start < text_length:
            end = min(start + TARGET_CHUNK_SIZE, text_length)
            if end < text_length:
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunk_text = text[start:end].strip()
            if not chunk_text:
                break

            chunk_dict = {
                "source": doc.get('source', 'Unknown'),
                "title": doc.get('title', 'Untitled'),
                "section": f"Section {len(chunks) + 1}",
                "text": chunk_text,
            }
            chunks.append(chunk_dict)

            if end == text_length:
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    except Exception as e:
        logger.error(f"Error while chunking document from source: {doc.get('source', 'Unknown')}. Error: {e}")
        raise
    return chunks

# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).
    """
    if not texts:
        logger.warning("Received empty list of texts for embedding.")
        return []

    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
    except Exception as e:
        logger.error(f"Error while embedding texts using OpenAI API: {e}")
        raise
    return [item.embedding for item in response.data]


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection."""

    try:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    except Exception as e:
        logger.error(f"Error while updating the Qdrant collection '{COLLECTION_NAME}': {e}")
        raise

# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant.

    TODO:
      - Call embed_texts on the chunk texts
      - Build PointStruct objects (id=uuid, vector, payload=chunk dict)
      - qdrant.upsert
    """
    if not chunks:
        logger.warning("No chunks to ingest.")
        return
    
    try:
        vectors_embeddings = embed_texts([chunk["text"] for chunk in chunks])
        print(f"  Embedding {len(chunks)} chunks…")
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=chunk,
            )
            for chunk, vector in zip(chunks, vectors_embeddings)
        ]
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
            )
    except Exception as e:
        logger.error(f"Error while ingesting embedded chunks into Qdrant db: {e}")
        raise

# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query."""
    
    if not query.strip():
        logger.warning("Received empty query for retrieval.")
        return []
    
    try:
        query_vector = embed_texts([query])[0]
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        retrieved_chunks: list[dict[str, Any]] = []
        for item in response.points:
            retrieved_chunks.append(
                { 
                    "chunk": item.payload['text'],
                    "score": float(item.score),
                    "filename": item.payload.get('source', 'Unknown'),
                    "title": item.payload.get('title', 'Untitled'),
                    "section": item.payload.get('section', 'Unknown')
                    }
                )
        return retrieved_chunks
    except Exception as e:
        logger.error(f"Error while retrieving chunks for query '{query}': {e}")
        raise

# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result."""

    start = time.perf_counter()
    retrieved_chunks = retrieve(question, k=k)

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "I could not find relevant excerpts in the provided Sherlock Holmes corpus.",
            "citations": [],
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }

    context_parts = []
    citations = []
    for chunk in retrieved_chunks:
        title = chunk.get("title", "Unknown")
        section = chunk.get("section", "Unknown")
        text = chunk.get("chunk", "")
        context_parts.append(f"[Source: {title} — {section}]\n{text}\n")
        citations.append({
            "source": chunk.get("filename", "Unknown"),
            "title": title,
            "section": section,
            "score": chunk.get("score", 0.0),
        })

    context = "\n\n".join(context_parts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Question: {question}
            Relevant excerpts: {context}
            Answer should be relevant to the excerpts above.
            Include citations in the form of (Story Title - Section)"""
        }
    ]
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0
        )
    answer_text = response.choices[0].message.content.strip() or "No relevant excerpts found."
    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "latency_ms": latency_ms,
    }


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
