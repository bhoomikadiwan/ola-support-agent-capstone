
"""Ola Support Agent — FastAPI, CrewAI, AutoGen and RAG."""
from fastapi.concurrency import run_in_threadpool
import asyncio
import json
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core import Response, guard, mask, check_support_ticket_status
from crew import run_crew
from offline_autogen import review
from memory import invoke_chat
from rag import DOCS, index, client, grounded

app = FastAPI(title="Ola Support Agent (Synthetic Demo)")


# ---------- Request models ----------

class AskRequest(BaseModel):
    query: str


class DocumentRequest(BaseModel):
    title: str
    text: str


# ---------- Privacy-safe request logging ----------

def log_request(trace, query, elapsed, status):
    log_path = Path(__file__).parent / "requests.jsonl"

    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({
                "trace_id": trace,
                "query": mask(query),
                "duration_ms": round(elapsed * 1000, 2),
                "status": status
            }) + "\n"
        )


# ---------- Application startup ----------

@app.on_event("startup")
def startup():
    try:
        client().get_collection("ola_sentence")
    except Exception:
        index()


# ---------- Ask support agent ----------

@app.post("/ask", response_model=Response)
def ask(req: AskRequest):
    start = time.perf_counter()
    trace = str(uuid.uuid4())
    status = "ok"

    try:
        safe_query = guard(req.query)

        # Step 1: Execute CrewAI
        result = run_crew(safe_query)

        # Step 2: Retrieve evidence separately
        ticket_match = re.search(r"OLA-\d{4}", safe_query)

        if ticket_match:
            evidence_record = check_support_ticket_status(
                ticket_match.group()
            )
            evidence_answer = json.dumps(evidence_record)
        else:
            evidence_record = grounded(safe_query)
            evidence_answer = evidence_record["answer"]

        # Step 3: Independent AutoGen review
        verdict = asyncio.run(
            review(
                draft=result.answer,
                evidence=evidence_answer
            )
        )

        # Step 4: Apply review verdict
        result.answer = verdict.final_answer
        result.reviewed = True
        result.reason = verdict.reason
        result.trace_id = trace

        return result

    except ValueError as e:
        status = "rejected"
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        status = "error"
        raise

    finally:
        log_request(
            trace,
            req.query,
            time.perf_counter() - start,
            status
        )


# ---------- Add new knowledge-base document ----------

@app.post("/add-document")
def add_document(req: DocumentRequest):
    start = time.perf_counter()
    trace = str(uuid.uuid4())
    status = "ok"

    try:
        guard(req.title + " " + req.text)

        from rag import model, chunks, CACHE

        doc = {
            "id": "custom-" + str(uuid.uuid4()),
            "title": req.title,
            "text": req.text
        }

        DOCS.append(doc)

        for strategy in ("fixed", "sentence"):
            pieces = chunks(doc, strategy)

            client().get_collection(
                "ola_" + strategy
            ).upsert(
                ids=[
                    f"{doc['id']}-{i}"
                    for i in range(len(pieces))
                ],
                documents=pieces,
                metadatas=[
                    {"parent": doc["id"]}
                    for _ in pieces
                ],
                embeddings=model().encode(
                    pieces,
                    normalize_embeddings=True
                ).tolist()
            )

        CACHE.clear()

        return {
            "id": doc["id"],
            "added": True
        }

    except ValueError as e:
        status = "rejected"
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        status = "error"
        raise

    finally:
        log_request(
            trace,
            req.title + " " + req.text,
            time.perf_counter() - start,
            status
        )


# ---------- WebSocket chat with session memory ----------

@app.websocket("/ws/{session_id}")
async def websocket_chat(ws: WebSocket, session_id: str):
    await ws.accept()

    try:
        while True:
            query = await ws.receive_text()
            start = time.perf_counter()
            trace = str(uuid.uuid4())
            status = "ok"

            try:
                safe_query = guard(query)

                answer = await run_in_threadpool(
                    invoke_chat,
                    safe_query,
                    session_id
                )

                await ws.send_json({
                    "trace_id": trace,
                    "answer": answer
                })

            except ValueError as e:
                status = "rejected"

                await ws.send_json({
                    "error": str(e)
                })

            except Exception:
                status = "error"
                raise

            finally:
                log_request(
                    trace,
                    query,
                    time.perf_counter() - start,
                    status
                )

    except WebSocketDisconnect:
        pass
