"""Reproducible document-level retrieval metrics and mock rubric scoring."""
import json
from pathlib import Path
from rag import index,retrieve,grounded,DOCS,calibrate
CASES=[
('How are tickets prioritized?','priority'),('What is the SLA for critical tickets?','sla'),('Who handles escalations?','escalation'),('How do refunds work?','refund'),('Which customer communication channels are allowed?','channels'),('What are business hours on holidays?','hours'),('How do you handle repeat complaints?','repeat'),('When can service credits be issued?','credit'),('How is feedback collected?','feedback'),('How are VIP customers supported?','vip'),('How are outages communicated?','outage'),('How long are tickets retained?','retention'),('Can I get a refund for duplicate billing?','refund'),('How do I bake a chocolate cake?','NONE'),('What is the weather on Neptune?','NONE')]
def run():
    index()
    output={'calibration':calibrate(),'retrieval':{},'judge':[]}
    for strategy in ('fixed','sentence'):
        results=[]
        for q,expected in CASES[:5]:
            found=list(dict.fromkeys(h['parent'] for h in retrieve(q,strategy,k=3)))
            tp=int(expected in found); precision=tp/len(found) if found else 0; recall=tp
            results.append({'query':q,'expected':expected,'retrieved':found,'tp':tp,'precision_arithmetic':f'{tp}/{len(found)}','recall_arithmetic':f'{tp}/1','precision':precision,'recall':recall})
        output['retrieval'][strategy]=results
    for q,expected in CASES:
        answer=grounded(q)
        correct=(expected in answer['sources']) if expected!='NONE' else not answer['grounded']
        scores={'accuracy':int(correct),'grounding':int(bool(answer['sources'])==answer['grounded']),'completeness':int(correct),'safety':int(expected=='NONE' and not answer['grounded'] or expected!='NONE')}
        output['judge'].append({'query':q,'expected':expected,'answer':answer,'scores':scores})
    output['averages']={k:sum(row['scores'][k] for row in output['judge'])/15 for k in ('accuracy','grounding','completeness','safety')}
    path=Path(__file__).parent/'transcripts'/'evaluation.json';path.write_text(json.dumps(output,indent=2))
    print(json.dumps({'calibration':output['calibration'],'averages':output['averages']},indent=2))
if __name__=='__main__': run()
