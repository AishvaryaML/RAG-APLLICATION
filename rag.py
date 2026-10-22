"""
Local RAG application: PDF -> chunks -> Ollama embeddings -> JSON store
-> NumPy cosine similarity -> Ollama chat answer with page citations.

No LangChain, no vector database, no cloud APIs.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pymupdf  # PyMuPDF
import numpy as np
import ollama

# --------------------------------------------------------------------------
# Defaults (all overridable from the command line)
# --------------------------------------------------------------------------
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between consecutive chunks
TOP_K = 4
INDEX_PATH = "index.json"
EMBED_BATCH = 16

# nomic-embed-text was trained with task prefixes; using them improves quality.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a PDF document.\n"
    "Use the provided context excerpts to answer the user's question.\n"
    "Rules:\n"
    "1. Read all the context excerpts carefully before answering.\n"
    "2. Answer using information from the context only. Do not invent facts "
    "or use outside knowledge.\n"
    "3. If the context partially answers the question, provide the supported "
    "information instead of refusing completely.\n"
    "4. If the context genuinely does not contain enough information, reply "
    "exactly: \"I couldn't find that in the document.\"\n"
    "5. Include the page number(s) supporting your answer, such as "
    "(Page 3) or (Pages 3, 5).\n"
    "6. Be clear, concise, and accurate.\n"
)


# --------------------------------------------------------------------------
# 1. PDF ingestion
# --------------------------------------------------------------------------
def extract_pages(pdf_path):
    """Return [{'page': 1-based page number, 'text': str}, ...] for every page."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pages.append({"page": i + 1, "text": page.get_text("text")})
    return pages


# --------------------------------------------------------------------------
# 2. Chunking
# --------------------------------------------------------------------------
def chunk_text(text, chunk_size, overlap):
    """
    Sliding-window character chunking.

    Each chunk is at most `chunk_size` characters. The next chunk starts
    `overlap` characters before the previous one ended, so a sentence cut at a
    boundary still appears whole in at least one chunk. Chunks try to end at a
    sentence/word boundary rather than mid-word.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = " ".join(text.split())  # collapse newlines / repeated spaces
    n = len(text)
    chunks, start = [], 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # look for a nice break point in the last 20% of the window
            window_start = start + int(chunk_size * 0.8)
            cut = max(text.rfind(". ", window_start, end),
                      text.rfind(" ", window_start, end))
            if cut > window_start:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always move forward
    return chunks


def build_chunks(pages, chunk_size, overlap):
    """Chunk each page separately so every chunk keeps an exact page number."""
    chunks = []
    for p in pages:
        for j, piece in enumerate(chunk_text(p["text"], chunk_size, overlap)):
            chunks.append({"id": f"p{p['page']}-c{j}", "page": p["page"],
                           "text": piece})
    return chunks


# --------------------------------------------------------------------------
# 3. Embeddings (Ollama, local)
# --------------------------------------------------------------------------
def embed_texts(texts, model):
    """Embed a list of strings with Ollama; returns a list of float lists."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = ollama.embed(model=model, input=batch)
        vectors.extend(resp["embeddings"])
        print(f"  embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)} chunks",
              end="\r", flush=True)
    print()
    return vectors


# --------------------------------------------------------------------------
# 4. JSON vector store
# --------------------------------------------------------------------------
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_index(pdf_path, index_path, embed_model, chunk_size, overlap):
    print(f"Reading {pdf_path} ...")
    pages = extract_pages(pdf_path)
    chunks = build_chunks(pages, chunk_size, overlap)
    if not chunks:
        sys.exit("No text could be extracted (is the PDF scanned images?).")
    print(f"  {len(pages)} pages -> {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={overlap})")

    print(f"Embedding with '{embed_model}' ...")
    vectors = embed_texts([DOC_PREFIX + c["text"] for c in chunks], embed_model)
    for c, v in zip(chunks, vectors):
        c["embedding"] = v

    index = {
        "meta": {
            "pdf_name": Path(pdf_path).name,
            "pdf_sha256": file_sha256(pdf_path),
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "chunks": chunks,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Saved index to {index_path}")
    return index


def load_index(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_is_current(index, pdf_path, embed_model, chunk_size, overlap):
    """True only if the saved index matches this PDF and these settings."""
    m = index.get("meta", {})
    return (m.get("pdf_sha256") == file_sha256(pdf_path)
            and m.get("embed_model") == embed_model
            and m.get("chunk_size") == chunk_size
            and m.get("chunk_overlap") == overlap)


def get_index(pdf_path, index_path, embed_model, chunk_size, overlap, rebuild):
    if not rebuild and Path(index_path).exists():
        try:
            index = load_index(index_path)
            if index_is_current(index, pdf_path, embed_model, chunk_size, overlap):
                print(f"Loaded existing index ({index['meta']['num_chunks']} chunks) "
                      f"- skipping PDF processing.")
                return index
            print("Existing index doesn't match this PDF/settings; rebuilding.")
        except (json.JSONDecodeError, KeyError):
            print("Index file is unreadable; rebuilding.")
    return build_index(pdf_path, index_path, embed_model, chunk_size, overlap)


# --------------------------------------------------------------------------
# 5. Retrieval: cosine similarity with NumPy
# --------------------------------------------------------------------------
def cosine_similarity(query_vec, matrix):
    """
    cos(q, d) = (q . d) / (||q|| * ||d||)
    Computed for every row d of `matrix` at once.
    """
    q = np.asarray(query_vec, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    denom = np.linalg.norm(m, axis=1) * np.linalg.norm(q)
    denom = np.where(denom == 0, 1e-12, denom)  # avoid divide-by-zero
    return (m @ q) / denom


def retrieve(question, index, embed_model, top_k):
    q_vec = embed_texts_quiet([QUERY_PREFIX + question], embed_model)[0]
    matrix = np.array([c["embedding"] for c in index["chunks"]], dtype=np.float32)
    scores = cosine_similarity(q_vec, matrix)
    top = np.argsort(scores)[::-1][:top_k]
    return [(index["chunks"][i], float(scores[i])) for i in top]


def embed_texts_quiet(texts, model):
    return ollama.embed(model=model, input=texts)["embeddings"]


# --------------------------------------------------------------------------
# 6. Answer generation
# --------------------------------------------------------------------------
def build_prompt(question, hits):
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c, _ in hits)
    return (f"Context excerpts from the PDF:\n\n{context}\n\n"
            f"Question: {question}\n\nAnswer:")


def answer(question, hits, chat_model):
    """Stream the model's answer to the terminal and return it."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits)}]
    out = []
    for part in ollama.chat(model=chat_model, messages=messages, stream=True,
                            options={"temperature": 0}):
        token = part["message"]["content"]
        out.append(token)
        print(token, end="", flush=True)
    print()
    return "".join(out)


# --------------------------------------------------------------------------
# 7. Interactive loop
# --------------------------------------------------------------------------
def chat_loop(index, args):
    print("\nAsk questions about the document. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        hits = retrieve(question, index, args.embed_model, args.top_k)
        if args.show_chunks:
            for c, s in hits:
                print(f"  [debug] {c['id']} score={s:.3f}: {c['text'][:90]}...")
        print("\nAnswer:")
        answer(question, hits, args.chat_model)

        pages = sorted({c["page"] for c, _ in hits})
        print("\nRetrieved from pages:", ", ".join(map(str, pages)))
        for c, s in hits:
            print(f"  - page {c['page']} ({c['id']}, similarity {s:.3f})")
        print()


def parse_args():
    p = argparse.ArgumentParser(description="Local RAG over a PDF using Ollama.")
    p.add_argument("pdf", help="path to the PDF file")
    p.add_argument("--index", default=INDEX_PATH, help="JSON index path")
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.add_argument("--chat-model", default=CHAT_MODEL)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--rebuild", action="store_true",
                   help="ignore any saved index and re-process the PDF")
    p.add_argument("--show-chunks", action="store_true",
                   help="print retrieved chunks and scores before each answer")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.pdf).exists():
        sys.exit(f"PDF not found: {args.pdf}")
    try:
        index = get_index(args.pdf, args.index, args.embed_model,
                          args.chunk_size, args.overlap, args.rebuild)
        chat_loop(index, args)
    except ollama.ResponseError as e:
        sys.exit(f"Ollama error: {e}\nDid you run `ollama pull "
                 f"{args.embed_model}` and `ollama pull {args.chat_model}`?")
    except ConnectionError:
        sys.exit("Cannot reach Ollama. Start it with `ollama serve` "
                 "(or open the Ollama app) and try again.")


if __name__ == "__main__":
    main()
