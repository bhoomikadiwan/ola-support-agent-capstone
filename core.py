"""Support tools, policy controls and structured responses."""
import re,uuid,os
from pydantic import BaseModel,Field
from dataset import SUPPORT_TICKETS
os.environ.setdefault('CREWAI_DISABLE_TELEMETRY','true')
PHONE=re.compile(r'(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)')
INJECTION=re.compile(r'ignore (?:all )?(?:previous|prior) instructions|reveal (?:your )?(?:system|secret) prompt|override (?:the )?policy',re.I)
class Response(BaseModel):
    trace_id:str
    response_format:str='support_response'
    answer:str
    sources:list[str]=Field(default_factory=list)
    grounded:bool
    reviewed:bool=False
    reason:str=''
def mask(text): return PHONE.sub('[PHONE_MASKED]',text)
def guard(text):
    if INJECTION.search(text): raise ValueError('Prompt injection rejected')
    if len(text.split())>500: raise ValueError('Request token budget exceeded')
    return mask(text)
def check_support_ticket_status(record_id:str)->dict:
    row=next((r for r in SUPPORT_TICKETS if r['record_id']==record_id),None)
    if row is None: return {'error':'Ticket not found'}
    score=round(0.65*int(row['escalated'])+0.35*row['days_since_created']/30,3)
    return {'record_id':record_id,'status':row['status'],'resolution_time_hours':row['resolution_time_hours'],'escalation_score':score,'recommend_escalation':score>0.55}
def new_response(answer,sources=None,grounded=False,reviewed=False,reason=''):
    return Response(trace_id=str(uuid.uuid4()),answer=answer,sources=sources or [],grounded=grounded,reviewed=reviewed,reason=reason)
