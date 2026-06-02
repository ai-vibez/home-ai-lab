#!/usr/bin/env python3
import sys
import json
import time
import argparse
import requests

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, FusionQuery, Prefetch, Fusion
from fastembed import SparseTextEmbedding

import config


def with_retry(fn, retries=3, base_delay=2.0):
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(base_delay * (attempt + 1))
    raise last_err


def truncate_context(chunks, max_chars):
    result, total = [], 0
    for chunk in chunks:
        if total + len(chunk) > max_chars:
            break
        result.append(chunk)
        total += len(chunk)
    return result


def llm_call(messages, model, stream=False):
    """Call llama-server (GPU) via OpenAI-compatible API."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream
    }
    response = requests.post(
        f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=stream,
        timeout=120
    )
    response.raise_for_status()

    if stream:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
    else:
        return response.json()["choices"][0]["message"]["content"]



def llm_call_blocking(messages, model):
    """Non-streaming LLM call — returns string directly."""
    response = requests.post(
        f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("query", nargs="+", help="Question to ask")
    parser.add_argument("--model", default=config.DEFAULT_LLM_MODEL)
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    parser.add_argument("--output", choices=["text", "json", "stream"], default="text")
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        print("ERROR: Empty query.")
        sys.exit(1)

    # Dense embedding (Ollama — CPU, embeddings only, acceptable)
    try:
        dense = with_retry(lambda: ollama.embeddings(model=config.EMBED_MODEL, prompt=query)["embedding"])
    except Exception as e:
        print(f"ERROR: Dense embedding failed — {e}")
        sys.exit(1)

    # Sparse embedding
    try:
        sparse_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
        sparse_result = list(sparse_model.embed([query]))[0]
        indices = sparse_result.indices.tolist()
        values = sparse_result.values.tolist()
    except Exception as e:
        print(f"ERROR: Sparse embedding failed — {e}")
        sys.exit(1)

    # Qdrant hybrid query
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        results = client.query_points(
            collection_name=config.COLLECTION,
            prefetch=[
                Prefetch(query=dense, using="dense", limit=args.limit),
                Prefetch(query=SparseVector(indices=indices, values=values), using="sparse", limit=args.limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=args.limit
        ).points
    except Exception as e:
        print(f"ERROR: Qdrant query failed — {e}")
        sys.exit(1)

    if not results:
        msg = "No relevant context found."
        if args.output == "json":
            print(json.dumps({"query": query, "answer": msg, "chunks_used": 0, "sources": []}))
        else:
            print(msg)
        sys.exit(0)

    # Deduplicate by text
    seen, unique_chunks = set(), []
    for r in results:
        text = r.payload.get("text", "").strip()
        if text and text not in seen:
            seen.add(text)
            unique_chunks.append((text, r.payload.get("source", "unknown")))

    if not unique_chunks:
        print("ERROR: Results have no text payload.")
        sys.exit(1)

    chunks = truncate_context([c[0] for c in unique_chunks], config.MAX_CONTEXT_CHARS)
    sources = list(set(c[1] for c in unique_chunks[:len(chunks)]))
    context = "\n\n".join(chunks)

    system_prompt = (
        "You are a precise assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say: 'This information is not in the provided documents.' "
        "Do not speculate or use outside knowledge."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    if args.output == "stream":
        print(f"=== ANSWER (model: {args.model}, GPU via llama-server) ===")
        try:
            for piece in llm_call(messages, args.model, stream=True):
                print(piece, end="", flush=True)
            print()
        except Exception as e:
            print(f"\nERROR: Stream failed — {e}")
            sys.exit(1)

    elif args.output == "json":
        try:
            answer = with_retry(lambda: llm_call_blocking(messages, args.model))
            print(json.dumps({
                "query": query,
                "answer": answer,
                "model": args.model,
                "chunks_used": len(chunks),
                "sources": sources
            }, ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"error": str(e), "query": query}))
            sys.exit(1)

    else:  # text
        print(f"=== CONTEXT ({len(chunks)} chunks from: {', '.join(sources)}) ===")
        print(context)
        print(f"\n=== ANSWER (model: {args.model}, GPU via llama-server) ===")
        try:
            answer = with_retry(lambda: llm_call_blocking(messages, args.model))
            print(answer)
        except Exception as e:
            print(f"ERROR: LLM call failed — {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
