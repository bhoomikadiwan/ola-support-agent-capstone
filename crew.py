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
    """Offline CrewAI adapter that uses actual task and tool messages."""

    def __init__(self):
        super().__init__(model="offline-mock", temperature=0)

    def call(self, messages, tools=None, callbacks=None,
             available_functions=None, **kwargs):

        contents = [
            str(m.get("content", "")) if isinstance(m, dict) else str(m)
            for m in messages
        ]
        conversation = "\n".join(contents)

        # Extract the original customer question, not CrewAI instructions.
        match = re.search(
            r"Use your assigned tool to answer:\s*(.+)",
            conversation
        )
        question = match.group(1).strip() if match else ""

        # Identify real tool observations, excluding system-prompt examples.
        observations = []
        for content in contents[2:]:
            for line in content.splitlines():
                if line.startswith("Observation: "):
                    value = line[len("Observation: "):].strip()
                    try:
                        observations.append(json.loads(value))
                    except json.JSONDecodeError:
                        pass

        if observations:
            evidence = observations[-1]
            return (
                "Thought: I have verified tool evidence.\n"
                "Final Answer: " + json.dumps(evidence)
            )

        # Composer has no tools. Read the JSON produced by the previous task.
        if "Response Composer" in contents[0]:
            for content in contents[1:]:
                marker = "This is the context you're working with:"
                if marker not in content:
                    continue

                context = content.split(marker, 1)[1]
                context = context.split("Provide your complete response:", 1)[0].strip()

                # The retrieval task returns JSON directly.
                try:
                    evidence = json.loads(context)
                    return (
                        "Thought: I have the evidence from the previous agent.\n"
                        "Final Answer: " + json.dumps(evidence)
                    )
                except json.JSONDecodeError:
                    pass

            return (
                "Thought: No valid evidence was received.\n"
                "Final Answer: " + json.dumps({
                    "answer": "I don't know based on the available support policies.",
                    "sources": [],
                    "grounded": False
                })
            )


        # Choose the tool using the agent's assigned role.
        if "ticket_lookup" in contents[0]:
            ticket = re.search(r"OLA-\d{4}", question)
            if ticket:
                return (
                    "Thought: I need ticket evidence.\n"
                    "Action: ticket_lookup\n"
                    "Action Input: " +
                    json.dumps({"record_id": ticket.group(0)})
                )

        return (
            "Thought: I need policy evidence.\n"
            "Action: rag_lookup\n"
            "Action Input: " + json.dumps({"query": question})
        )

    def supports_function_calling(self):
        return False

    def get_context_window_size(self):
        return 8192

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
    safe = guard(query)

    # Execute the CrewAI workflow.
    crew_output = build_crew(safe).kickoff()

    # Read the actual Response Composer output.
    try:
        evidence = json.loads(str(crew_output))
    except json.JSONDecodeError:
        return new_response(
            "I don't know based on the available support policies.",
            [],
            False,
            reason="Composer returned invalid JSON"
        )

    # Handle a ticket lookup response.
    if "record_id" in evidence:
        ticket_id = evidence["record_id"]

        # Verify the Composer's claims against the authorized tool.
        verified = check_support_ticket_status(ticket_id)

        if evidence != verified:
            return new_response(
                "Ticket information could not be verified.",
                [],
                False,
                reason="Ticket evidence mismatch"
            )

        return new_response(
            answer=json.dumps(evidence),
            sources=[ticket_id],
            grounded="error" not in evidence
        )

    # Handle a policy RAG response.
    expected = grounded(safe)

    if evidence != expected:
        return new_response(
            "I don't know based on the available support policies.",
            [],
            False,
            reason="Policy evidence mismatch"
        )

    return new_response(
        answer=evidence["answer"],
        sources=evidence["sources"],
        grounded=evidence["grounded"]
    )
