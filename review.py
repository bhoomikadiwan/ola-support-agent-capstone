"""Offline deterministic review plus AutoGen RoundRobinGroupChat integration."""
from pydantic import BaseModel
from core import new_response
class Verdict(BaseModel):
    approved:bool
    final_answer:str
    reason:str

def mock_review(draft,context):
    if draft.strip()==context.strip(): return Verdict(approved=True,final_answer=draft,reason='Exact match to grounded evidence')
    return Verdict(approved=False,final_answer=context,reason='Draft revised to remove unsupported wording')

def build_autogen_team(model_client):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.messages import StructuredMessage
    policy=AssistantAgent('Policy_Compliance_Reviewer',model_client=model_client,system_message='Compare draft to evidence; flag unsupported claims.')
    editor=AssistantAgent('Final_Editor',model_client=model_client,output_content_type=Verdict,system_message='Return approved, final_answer and reason; only use evidence.')
    return RoundRobinGroupChat([policy,editor],max_turns=2,custom_message_types=[StructuredMessage[Verdict]])
