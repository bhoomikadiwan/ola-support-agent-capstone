"""Three role-separated CrewAI agents with offline BaseLLM adapter."""
import os,re,json
os.environ.setdefault('CREWAI_DISABLE_TELEMETRY','true')
from crewai import Agent,Task,Crew,Process,LLM
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool
from rag import grounded
from core import check_support_ticket_status,guard,new_response
@tool('rag_lookup')
def rag_lookup(query:str)->str:
    """Retrieve grounded policy text for a customer question."""
    return json.dumps(grounded(query))
@tool('ticket_lookup')
def ticket_lookup(record_id:str)->str:
    """Look up a synthetic support ticket by its OLA-0001 identifier."""
    return json.dumps(check_support_ticket_status(record_id))
class MockLLM(BaseLLM):
    """Deterministic CrewAI ReAct adapter; only examines latest task/tool messages."""
    def __init__(self): super().__init__(model='offline-mock',temperature=0)
    def call(self,messages,tools=None,callbacks=None,available_functions=None,  **kwargs):
        latest=str(messages[-1].get('content','')) if isinstance(messages,list) and messages else str(messages)
        # Do not search the system prompt: it includes a misleading Observation example.
        match=re.search(r'Observation:\s*(\{.*\})',latest,re.S)
        if match:
            try:
                data=json.loads(match.group(1).split('\nThought:')[0])
                return 'Thought: I have the tool evidence.\nFinal Answer: '+json.dumps(data)
            except ValueError: pass
        if 'ticket_lookup' in latest or 'OLA-' in latest:
            match=re.search(r'OLA-\d{4}',latest)
            if match: return 'Thought: I need ticket evidence.\nAction: ticket_lookup\nAction Input: '+json.dumps({'record_id':match.group(0)})
        query=latest[-300:].replace('\n',' ')
        return 'Thought: I need policy evidence.\nAction: rag_lookup\nAction Input: '+json.dumps({'query':query})
    def supports_function_calling(self): return False
    def get_context_window_size(self): return 8192

def build_crew(query):
    llm=MockLLM()
    retrieval=Agent(role='Retrieval Agent',goal='Find policy evidence',backstory='Policy specialist',tools=[rag_lookup],llm=llm,verbose=False,max_iter=2)
    lookup=Agent(role='Lookup Agent',goal='Find ticket status',backstory='Authorized ticket operator',tools=[ticket_lookup],llm=llm,verbose=False,max_iter=2)
    composer=Agent(role='Response Composer',goal='Combine evidence into concise reply',backstory='Support editor',tools=[],llm=llm,verbose=False,max_iter=2)
    ticket=bool(re.search(r'OLA-\d{4}',query))
    selected=lookup if ticket else retrieval
    task1=Task(description=f'Use your assigned tool to answer: {query}',expected_output='Tool evidence as JSON',agent=selected)
    task2=Task(description='Compose an answer from prior evidence only; do not invent claims.',expected_output='Grounded final answer',agent=composer,context=[task1])
    return Crew(agents=[retrieval,lookup,composer],tasks=[task1,task2],process=Process.sequential,verbose=False)
def run_crew(query):
    safe=guard(query)
    raw=build_crew(safe).kickoff()
    # Independent deterministic output validation, not trust in a mock-generated claim.
    ticket=re.search(r'OLA-\d{4}',safe)
    if ticket:
        evidence=check_support_ticket_status(ticket.group())
        return new_response(json.dumps(evidence),[ticket.group()], 'error' not in evidence)
    evidence=grounded(safe)
    return new_response(evidence['answer'],evidence['sources'],evidence['grounded'])
