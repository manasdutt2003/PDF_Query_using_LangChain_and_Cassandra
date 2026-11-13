# PDF Query using LangChain and Cassandra

Ask questions about a PDF in plain English. The document is split into
overlapping chunks, embedded locally with `all-MiniLM-L6-v2`, and stored in an
[Astra DB](https://astra.datastax.com) (Cassandra) vector table. Each question
retrieves the closest passages and Google Gemini answers from them.

```
PDF -> PyPDF2 -> chunks -> MiniLM embeddings -> Astra DB vector table
                                                     |
question -> MiniLM embedding -> top-k passages -> Gemini answer
```

## Contents

| File | Purpose |
| --- | --- |
| `pdfquery_langchain.py` | Command-line version: ingest a PDF, then ask questions |
| `PDFQuery_LangChain.ipynb` | Original Colab walkthrough with outputs |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then add your keys
```

You need:

- an Astra DB serverless database with vector search enabled, its database ID
  and an application token
- a [Google AI Studio](https://aistudio.google.com/apikey) API key

## Usage

Ingest a PDF and start an interactive session:

```bash
python pdfquery_langchain.py budget_speech.pdf
```

Ask one question against a PDF that is already stored:

```bash
python pdfquery_langchain.py budget_speech.pdf --skip-ingest \
    --question "What does the document say about climate change?"
```

The notebook runs the same steps cell by cell. Replace the `YOUR_...`
placeholders in the credentials cell, or load them from Colab secrets.

## Security

Credentials are read from `.env`, which is git-ignored. Do not paste keys into
the script or the notebook before committing.
