# Ola (Business Operations / Customer Support) — Final Capstone

**Track completed:** Ola (Business Operations / Customer Support). Synthetic educational demonstration; not affiliated with Ola. **MOCK_LLM**, zero API keys. CrewAI telemetry disabled with `CREWAI_DISABLE_TELEMETRY=true` before crew kickoff. Do not submit as production-ready without additional hardening.

## Dataset design (reproducible)

`dataset.py` uses seed **260313**, 60 records, category weights **[25,25,15,15,20]** in the order Billing, Technical Issue, Account Access, Product Defect, General Inquiry; status weights **[20,25,15,25,15]** in the order Open, In Progress, Escalated, Resolved, Closed. Synthetic disputed amount range **INR 50–2500**. Resolution time range **1–96 hours**, chosen to cover short routine requests and complex escalations. Escalated flag probability **0.19**. Counts: Billing 15, Technical Issue 18, Account Access 9, Product Defect 11, General Inquiry 7; statuses Open 11, In Progress 14, Escalated 7, Resolved 16, Closed 12; escalated percentage **18.33%**.

## Installation (Python 3.11 recommended)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python dataset.py
python rag.py
python evaluate.py
pytest -q
python demo.py
uvicorn api:app --reload
```

The first local SentenceTransformers model download requires internet access; **inference and mock transcripts use no API keys and no model API calls**. For strict offline runtime, pre-download and cache the model before disconnecting. Visit `http://127.0.0.1:8000/docs` to test `POST /ask` and `POST /add-document`. Connect WebSocket `ws://127.0.0.1:8000/ws/session-A`.

## Architecture and scoring evidence

- `dataset.py`: deterministic dataset, counts and validation.
- `kb/policies.json`: 12 original synthetic policies (not official Ola policies).
- `rag.py`: fixed 240-character chunks with 40-character overlap and sentence pairs; local MiniLM embeddings, two Chroma cosine collections, `upsert`, retrieval, empirical threshold calibration, grounded answers and normalized-query cache.
- `evaluate.py`: 15 test cases, per-query four mock-judge scores, averages and 5-query precision/recall arithmetic for each collection. **These are deterministic proxy judgments, not independent LLM judgments.**
- `core.py`: schema, phone masking, injection detection, request budget, ticket score.
- `crew.py`: three role-separated agents, two tools and `Crew.kickoff()` with custom `BaseLLM`.
- `memory.py`: session-isolated LangChain message history and multi-turn demo.
- `review.py`: AutoGen `RoundRobinGroupChat` factory and deterministic offline review verdict demonstration.
- `api.py`: two HTTP endpoints, WebSocket disconnect handling, JSONL trace logging with masked phone values.
- `demo.py`: reproducible demonstration transcript, including guardrail rejection, cache hit and review revision.

## Threshold and retrieval comparison

**Do not invent measurements.** Run `python evaluate.py` and paste actual `transcripts/evaluation.json` calibration scores and selected threshold here before submission. Threshold is computed as the midpoint between the lowest in-scope and highest out-of-scope top-1 cosine similarity; calibration fails loudly if the sets overlap. Report the fixed and sentence strategy precision and recall from that same file, then recommend the strategy supported by measured values. The demo currently defaults to sentence chunks.

## Escalation policy

`escalation_score = 0.65 * int(escalated) + 0.35 * days_since_created / 30`. Recommend escalation when score **>0.55**, so all already-escalated tickets trigger review while non-escalated aging tickets do not cross the threshold. This is a conservative synthetic routing rule, not an empirically optimized percentile. For grading, add the actual age distribution and justification.

## Governance

**Application:** Medium risk, because customer support responses can affect service outcomes but the demo cannot process payments or issue credits. Only the Lookup Agent receives the ticket lookup tool; the Retrieval and Composer agents have no access. **Runtime:** maximum 500 input words (a simple conservative proxy for a token budget), with explicit rejection. **Privacy:** Indian-format phone numbers are masked before model input and JSONL logging. **Human oversight:** refunds, safety escalations and credits require staff approval. **Data:** synthetic only. **Review:** policy compliance and final editing before customer release should be enforced; the included deterministic review demonstration is not yet integrated into the API response path.

## Important remaining validation before claiming full marks

This starter is intentionally transparent about limitations: AutoGen team factory is provided but the real AutoGen two-agent round robin is not executed by the default mock transcript; the API does not yet pass its response through AutoGen; the custom CrewAI mock adapter may require adjustment for the installed CrewAI release; the judge uses deterministic heuristics rather than a prompt-driven mock judge; and the runtime budget uses words rather than exact tokenizer counts. Fix these and run the full end-to-end acceptance tests before claiming all 100 marks. Do not present unexecuted code as validated evidence.

## GitHub submission

Create one **public** GitHub repository, then run:

```bash
git init
git add .
git commit -m "Initial Ola support capstone implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ola-support-capstone.git
git push -u origin main
```

Submit only the public repository URL. Do not commit `.env`, local cache, raw logs, or personal data.
