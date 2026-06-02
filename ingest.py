#!/usr/bin/env python3
import os
import sys
import json
import hashlib
import argparse
import re
import uuid
from datetime import datetime, timezone

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    SparseVectorParams, SparseIndexParams,
    Filter, FieldCondition, MatchValue, FilterSelector
)
from fastembed import SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    from pypdf import PdfReader
    PDF_BACKEND = "pypdf"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    if os.path.exists(config.MANIFEST_FILE):
        try:
            with open(config.MANIFEST_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest):
    with open(config.MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def clean_text(text):
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def extract_pdf_text(path):
    if PDF_BACKEND == "pdfplumber":
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    else:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)


def ensure_collection(client):
    if not client.collection_exists(config.COLLECTION):
        client.create_collection(
            collection_name=config.COLLECTION,
            vectors_config={"dense": VectorParams(size=config.DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())}
        )
        print(f"Created collection: {config.COLLECTION}")


def delete_source_chunks(client, filename):
    client.delete(
        collection_name=config.COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=filename))])
        )
    )


def ingest_file(client, sparse_model, path, filename):
    raw_text = extract_pdf_text(path)
    if not raw_text.strip():
        raise ValueError("No text extracted from PDF")

    text = clean_text(raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = [c.strip() for c in splitter.split_text(text) if c.strip() and len(c.strip()) > 20]

    if not chunks:
        raise ValueError("No valid chunks after splitting")

    print(f"  {filename}: {len(chunks)} chunks — generating sparse embeddings (batch)...")
    sparse_embeddings = list(sparse_model.embed(chunks))

    print(f"  {filename}: generating dense embeddings (sequential)...")
    dense_embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 50 == 0 and i > 0:
            print(f"    {i}/{len(chunks)}")
        try:
            emb = ollama.embeddings(model=config.EMBED_MODEL, prompt=chunk)["embedding"]
            dense_embeddings.append(emb)
        except Exception as e:
            raise RuntimeError(f"Dense embedding failed at chunk {i}: {e}")

    points = []
    for i, (chunk, dense, sparse) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{i}"))
        points.append(PointStruct(
            id=point_id,
            vector={
                "dense": dense,
                "sparse": {
                    "indices": sparse.indices.tolist(),
                    "values": sparse.values.tolist()
                }
            },
            payload={
                "text": chunk,
                "source": filename,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }
        ))

    for start in range(0, len(points), config.UPSERT_BATCH_SIZE):
        client.upsert(collection_name=config.COLLECTION, points=points[start:start + config.UPSERT_BATCH_SIZE])

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument("--docs-dir", default=config.DOCS_DIR)
    parser.add_argument("--force", action="store_true", help="Re-ingest all files")
    parser.add_argument("--file", help="Ingest specific file only")
    args = parser.parse_args()

    docs_dir = os.path.expanduser(args.docs_dir)
    if not os.path.isdir(docs_dir):
        print(f"ERROR: docs dir not found: {docs_dir}")
        sys.exit(1)

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        client.get_collections()
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant: {e}")
        sys.exit(1)

    ensure_collection(client)

    print(f"Loading sparse model '{config.SPARSE_MODEL}' (downloads on first use)...")
    try:
        sparse_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
    except Exception as e:
        print(f"ERROR: Failed to load sparse model: {e}")
        sys.exit(1)

    manifest = load_manifest()

    if args.file:
        target = os.path.basename(args.file)
        pdf_files = [target] if os.path.exists(os.path.join(docs_dir, target)) else []
        if not pdf_files:
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
    else:
        pdf_files = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf"))

    if not pdf_files:
        print("No PDFs found.")
        sys.exit(0)

    stats = {"ingested": 0, "skipped": 0, "failed": 0}

    for filename in pdf_files:
        path = os.path.join(docs_dir, filename)
        file_hash = file_sha256(path)

        if not args.force and manifest.get(filename, {}).get("hash") == file_hash:
            print(f"  SKIP {filename} (unchanged)")
            stats["skipped"] += 1
            continue

        print(f"  INGEST {filename}...")
        if manifest.get(filename):
            delete_source_chunks(client, filename)

        try:
            count = ingest_file(client, sparse_model, path, filename)
            manifest[filename] = {
                "hash": file_hash,
                "chunks": count,
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }
            save_manifest(manifest)
            print(f"  OK {filename}: {count} chunks")
            stats["ingested"] += 1
        except Exception as e:
            print(f"  FAILED {filename}: {e}")
            stats["failed"] += 1

    print(f"\nDone. Ingested: {stats['ingested']} | Skipped: {stats['skipped']} | Failed: {stats['failed']}")


if __name__ == "__main__":
    main()
