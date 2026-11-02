# Ola Support Agent — CrewAI, AutoGen, RAG & FastAPI

An offline, multi-agent customer support capstone demonstrating knowledge-base retrieval, synthetic ticket lookup, session memory, response review, guardrails, evaluation, and API deployment.

**GitHub:** https://github.com/bhoomikadiwan/ola-support-agent-capstone

**Disclaimer:** This is an educational project using synthetic data and policies. It is not affiliated with Ola and is not production-ready.

## 1. Project Overview

The system demonstrates a customer support workflow using:

- **CrewAI:** Role-separated agents for retrieval, ticket lookup, and response composition.
- **AutoGen:** A two-agent review workflow using `RoundRobinGroupChat`.
- **RAG:** SentenceTransformers embeddings and ChromaDB vector retrieval.
- **LangChain:** Session-scoped conversation history.
- **FastAPI:** HTTP and WebSocket interfaces.
- **Guardrails:** Input checks, phone-number masking, request limits, and grounded-answer fallback.
- **Evaluation:** Deterministic tests, retrieval comparisons, and saved demonstration results.

The project uses an offline mock language model and does not require a paid model API key.

## 2. Architecture

```text
Customer Query
      |
      v
FastAPI /ask or WebSocket
      |
      v
Input Validation and Guardrails
      |
      v
CrewAI Workflow
  |         |
  v         v
RAG       Ticket Lookup
  |         |
  +----+----+
       |
       v
Grounded Response Construction
       |
       v
AutoGen Review (HTTP /ask)
       |
       v
Structured API Response
```

**Implementation distinction:** CrewAI agents execute through `Crew.kickoff()`, but the current final response is constructed using deterministic grounding and ticket-lookup logic. The offline mock model is not equivalent to a hosted generative language model.

The HTTP `/ask` route invokes AutoGen review. The WebSocket route provides session-based responses but does not currently use the same AutoGen review path.

## 3. Dataset and Knowledge Base

`dataset.py` generates 60 reproducible synthetic customer support records using seed `260313`.

The dataset covers billing, technical issues, account access, product defects, and general inquiries.

`kb/policies.json` contains 12 synthetic support policies.

These records and policies are demonstration data, not official Ola customer records or service commitments.

## 4. Installation

**Recommended environment:** Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Generate the synthetic dataset and initialize retrieval:

```powershell
python dataset.py
python rag.py
```

The first SentenceTransformers model download requires internet access. Subsequent local inference can run without a hosted model API.

## 5. Run the Application

Start the FastAPI server:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### HTTP endpoints

**POST `/ask`**

Example request:

```json
{
  "query": "What is the SLA for critical tickets?"
}
```

The response includes an answer, source references, grounding status, review information, and a trace identifier.

**POST `/add-document`**

Adds a new knowledge-base document to the local retrieval collections.

### WebSocket

Connect to:

```text
ws://127.0.0.1:8000/ws/session-A
```

The WebSocket supports session-based interactions. Conversation history is stored in memory and resets when the server restarts.

## 6. Testing

Run the automated tests:

```powershell
python -m pytest -v
```

**Last verified result: 6 tests passed.**

The tests cover dataset behavior, phone masking, prompt-injection checks, request budgeting, ticket lookup, and deterministic review.

Run the demonstration:

```powershell
python demo.py
```

Run the evaluation:

```powershell
python evaluate.py
```

Generated evidence is saved in the `transcripts/` directory.

## 7. RAG Evaluation

The project compares fixed-size and sentence-based chunking strategies.

| Metric | Fixed chunks | Sentence chunks |
|---|---:|---:|
| Average precision at k=3 | 36.7% | 40.0% |
| Average recall | 100% | 100% |

The empirical similarity threshold is **0.38103**.

These results are based on a small synthetic evaluation set, with retrieval comparison across five queries. They should not be interpreted as general production performance.

Sentence-based retrieval is used as the default strategy.

## 8. Response Evaluation

The evaluation includes 15 queries, with deterministic proxy scoring for accuracy, grounding, completeness, and safety.

The recorded average scores were 1.0 for each dimension.

These scores reflect the project's rule-based evaluation methodology. They are not independent LLM judgments or evidence of perfect real-world accuracy.

## 9. AutoGen Review

The project includes two review implementations:

- `review.py` provides a deterministic `mock_review()` function used by tests and demonstrations.
- `offline_autogen.py` provides the AutoGen two-agent review workflow used by the HTTP API.

The AutoGen workflow uses a policy reviewer and final editor to produce a structured verdict.

The current offline review relies on exact-match evidence checks rather than independent semantic verification.

## 10. Safety and Governance

The project includes:

- Input checks for prompt-injection patterns.
- Indian-format phone-number masking.
- A conservative request-size limit.
- An explicit fallback for unsupported policy questions.
- Source references and grounding indicators.
- Synthetic ticket data.
- Review before HTTP response delivery.

The application cannot issue real refunds, payments, or account changes.

For production use, additional authentication, authorization, privacy controls, monitoring, semantic validation, and human approval workflows would be required.

## 11. Limitations

This is a working educational demonstration, not a production support platform.

Important limitations include:

- Offline mock model rather than a hosted generative model.
- Deterministic final-answer construction after CrewAI execution.
- Exact-match reviewer checks rather than independent semantic fact-checking.
- In-memory conversation history without persistent storage.
- Synthetic policies and customer records.
- Small evaluation dataset and rule-based evaluation scores.
- No verified production deployment or real customer integrations.

## 12. Project Files

| File | Purpose |
|---|---|
| `api.py` | FastAPI HTTP and WebSocket application |
| `crew.py` | CrewAI agent orchestration |
| `offline_autogen.py` | AutoGen review workflow |
| `review.py` | Deterministic reviewer and AutoGen team factory |
| `rag.py` | Embeddings, retrieval, and grounding |
| `dataset.py` | Synthetic dataset generation |
| `core.py` | Schemas, guardrails, and ticket logic |
| `memory.py` | Session-scoped conversation history |
| `evaluate.py` | Evaluation pipeline |
| `demo.py` | Reproducible demonstration |
| `tests/` | Automated tests |
| `transcripts/` | Saved demonstration and evaluation evidence |

## 13. Submission

**Repository:** https://github.com/bhoomikadiwan/ola-support-agent-capstone

**Branch:** `main`

The repository contains the implementation, setup instructions, tests, synthetic knowledge base, and evaluation evidence.