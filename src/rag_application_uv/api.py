
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys

# Allow api.py to import rag.py from the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import rag

app = FastAPI(
    title="UV RAG Application",
    description="Ask questions about a PDF using local Ollama models",
    version="1.0.0",
)

PDF_PATH = ROOT / "NLP.pdf"
INDEX_PATH = ROOT / "NLP.index.json"

index = None


class QuestionRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_document():
    global index

    if not PDF_PATH.exists():
        raise RuntimeError(f"PDF not found: {PDF_PATH}")

    index = rag.get_index(
        str(PDF_PATH),
        str(INDEX_PATH),
        rag.EMBED_MODEL,
        rag.CHUNK_SIZE,
        rag.CHUNK_OVERLAP,
        False,
    )


@app.get("/")
def home():
    return {"message": "RAG API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    try:
        hits = rag.retrieve(
            request.question,
            index,
            rag.EMBED_MODEL,
            rag.TOP_K,
        )

        result = rag.answer(
            request.question,
            hits,
            rag.CHAT_MODEL,
        )

        return {
            "question": request.question,
            "answer": result,
            "source_pages": sorted(
                {chunk["page"] for chunk, score in hits}
            ),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
