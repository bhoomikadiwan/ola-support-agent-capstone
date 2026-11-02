"""Deterministic synthetic Ola support tickets. No real customer data."""
import random
from collections import Counter
SEED = 260313
CATEGORIES = ['Billing','Technical Issue','Account Access','Product Defect','General Inquiry']
STATUSES = ['Open','In Progress','Escalated','Resolved','Closed']
CATEGORY_WEIGHTS = [25,25,15,15,20]
STATUS_WEIGHTS = [20,25,15,25,15]
AMOUNT_RANGE_INR = (50,2500)  # synthetic disputed amount

def generate_tickets(n=60):
    rng=random.Random(SEED)
    rows=[]
    for i in range(n):
        rows.append(dict(record_id=f'OLA-{i+1:04}',category=rng.choices(CATEGORIES,CATEGORY_WEIGHTS)[0],status=rng.choices(STATUSES,STATUS_WEIGHTS)[0],resolution_time_hours=round(rng.uniform(1,96),1),days_since_created=rng.randint(0,30),escalated=rng.random()<0.19,amount_inr=rng.randint(*AMOUNT_RANGE_INR)))
    return rows
SUPPORT_TICKETS=generate_tickets()
def validate():
    cats=Counter(x['category'] for x in SUPPORT_TICKETS); statuses=Counter(x['status'] for x in SUPPORT_TICKETS)
    pct=100*sum(x['escalated'] for x in SUPPORT_TICKETS)/len(SUPPORT_TICKETS)
    assert len(SUPPORT_TICKETS)>=40 and all(cats[c]>=3 for c in CATEGORIES)
    assert all(statuses[s]>=1 for s in STATUSES) and 10<=pct<=30
    return {'categories':dict(cats),'statuses':dict(statuses),'escalated_percent':round(pct,2)}
if __name__=='__main__': print(validate())
