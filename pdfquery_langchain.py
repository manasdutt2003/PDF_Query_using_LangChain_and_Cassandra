"""Ask questions about a PDF using LangChain, Astra DB (Cassandra) and Gemini.

The PDF text is split into overlapping chunks, embedded locally with a
sentence-transformers model, and stored in an Astra DB vector table. Each
question retrieves the closest chunks and Gemini answers from them.

Usage:
    python pdfquery_langchain.py budget_speech.pdf
    python pdfquery_langchain.py budget_speech.pdf --question "What about climate change?"
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

TABLE_NAME = "qa_mini_demo"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-2.5-flash"

ANSWER_PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, say so.

Context:
{context}

Question: {question}
"""


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"Missing {name}. Copy .env.example to .env and fill it in.")
    return value


def read_pdf(path: str) -> str:
    from PyPDF2 import PdfReader

    pages = (page.extract_text() for page in PdfReader(path).pages)
    text = "\n".join(page for page in pages if page)
    if not text.strip():
        sys.exit(f"No extractable text in {path!r}. Scanned PDFs need OCR first.")
    return text


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Split raw text into overlapping chunks for embedding."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    return splitter.split_text(text)


def build_vector_store():
    import cassio
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Cassandra

    cassio.init(
        token=require_env("ASTRA_DB_APPLICATION_TOKEN"),
        database_id=require_env("ASTRA_DB_ID"),
    )
    return Cassandra(
        embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        table_name=os.getenv("ASTRA_DB_TABLE", TABLE_NAME),
        session=None,
        keyspace=None,
    )


def answer(store, llm, question: str, top_k: int) -> None:
    documents = store.similarity_search(question, k=top_k)
    if not documents:
        print("No matching passages found.\n")
        return
    context = "\n\n".join(doc.page_content for doc in documents)
    reply = llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
    print(f"\n{reply.content}\n")
    print("Passages used:")
    for number, doc in enumerate(documents, 1):
        print(f"  {number}. {' '.join(doc.page_content.split())[:200]}...")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--question", help="Ask one question and exit")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Query the existing table without re-adding the PDF",
    )
    args = parser.parse_args()

    load_dotenv()
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("CHAT_MODEL", CHAT_MODEL),
        google_api_key=require_env("GOOGLE_API_KEY"),
        temperature=0,
    )
    store = build_vector_store()

    if not args.skip_ingest:
        chunks = split_text(read_pdf(args.pdf))
        store.add_texts(chunks)
        print(f"Stored {len(chunks)} chunks from {args.pdf}.")

    if args.question:
        answer(store, llm, args.question, args.top_k)
        return

    print("Ask a question about the document (blank line to quit).")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer(store, llm, question, args.top_k)


if __name__ == "__main__":
    main()
