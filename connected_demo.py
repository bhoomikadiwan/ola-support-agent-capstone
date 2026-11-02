"""CrewAI kickoff -> evidence -> AutoGen review, with offline mock mode."""
import asyncio, json, os
os.environ['CREWAI_DISABLE_TELEMETRY']='true'
os.environ['OTEL_SDK_DISABLED']='true'
from crew import run_crew
from core import guard, check_support_ticket_status
from rag import grounded, index
from offline_autogen import review
import re

def run_pipeline(query: str):
    safe = guard(query)
    draft = run_crew(safe)  # internally invokes Crew.kickoff()
    ticket = re.search(r'OLA-\d{4}', safe)
    if ticket:
        evidence = json.dumps(check_support_ticket_status(ticket.group()))
    else:
        evidence = grounded(safe)['answer']
    verdict = asyncio.run(review(draft.answer, evidence))
    final = draft.model_copy(update={'answer':verdict.final_answer,
                                     'reviewed':True,'reason':verdict.reason})
    return final, verdict

if __name__ == '__main__':
    index()
    result, verdict = run_pipeline('What is the refund policy?')
    print('FINAL:',result.model_dump_json(indent=2))
    print('AUTOGEN:',verdict.model_dump_json(indent=2))
    print('REVISION TEST:',asyncio.run(review('Refunds are always instant.', 'Refunds require review.')).model_dump_json(indent=2))
