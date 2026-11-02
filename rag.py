"""Local embeddings, two persistent Chroma indexes and grounded policy answers."""
import json,re
from pathlib import Path
from functools import lru_cache
ROOT=Path(__file__).parent
DOCS=json.loads((ROOT/'kb/policies.json').read_text())
@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
@lru_cache(maxsize=1)
def client():
    import chromadb
    return chromadb.PersistentClient(path=str(ROOT/'.chroma'))
def chunks(doc,strategy):
    text=doc['title']+'. '+doc['text']
    if strategy=='fixed':
        return [text[i:i+240] for i in range(0,len(text),200)]
    sentences=re.split(r'(?<=[.!?])\s+',text)
    return [' '.join(sentences[i:i+2]) for i in range(0,len(sentences),2)]
def index():
    for strategy in ('fixed','sentence'):
        collection=client().get_or_create_collection('ola_'+strategy,metadata={'hnsw:space':'cosine'})
        rows=[(doc['id'],i,chunk) for doc in DOCS for i,chunk in enumerate(chunks(doc,strategy))]
        vectors=model().encode([r[2] for r in rows],normalize_embeddings=True).tolist()
        collection.upsert(ids=[f'{d}-{i}' for d,i,_ in rows],documents=[t for _,_,t in rows],metadatas=[{'parent':d} for d,_,_ in rows],embeddings=vectors)
def retrieve(query,strategy='sentence',k=3):
    col=client().get_collection('ola_'+strategy)
    vector=model().encode([query],normalize_embeddings=True).tolist()
    result=col.query(query_embeddings=vector,n_results=k,include=['documents','metadatas','distances'])
    return [{'text':t,'parent':m['parent'],'similarity':round(1-dist,5)} for t,m,dist in zip(result['documents'][0],result['metadatas'][0],result['distances'][0])]
CALIBRATION_IN=['How do I request a refund for duplicate charge?','What is the critical incident SLA?','How long are support tickets retained?']
CALIBRATION_OUT=['How to grow tomatoes on Mars?','Who won the ancient chess tournament on Jupiter?']
def calibrate():
    inside=[retrieve(q)[0]['similarity'] for q in CALIBRATION_IN]
    outside=[retrieve(q)[0]['similarity'] for q in CALIBRATION_OUT]
    if min(inside)<=max(outside): raise ValueError(f'No separable threshold: {inside} versus {outside}; revise KB and queries')
    return {'in_scope':inside,'out_scope':outside,'threshold':(min(inside)+max(outside))/2}
@lru_cache(maxsize=1)
def threshold(): return calibrate()['threshold']
CALLS={'grounded':0,'cache_hits':0}
CACHE={}
def grounded(query):
    key=' '.join(query.lower().split())
    if key in CACHE:
        CALLS['cache_hits']+=1
        return CACHE[key]
    CALLS['grounded']+=1
    hits=retrieve(query)
    if not hits or hits[0]['similarity']<threshold():
        answer={'answer':"I don't know based on the available support policies.",'sources':[],'grounded':False}
    else:
        parent=hits[0]['parent']
        doc=next(d for d in DOCS if d['id']==parent)
        answer={'answer':doc['text'],'sources':[parent],'grounded':True}
    CACHE[key]=answer
    return answer
if __name__=='__main__':
    index(); print(json.dumps(calibrate(),indent=2))
