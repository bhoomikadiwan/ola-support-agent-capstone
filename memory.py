"""LangChain session-scoped in-memory conversation history."""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda,RunnableWithMessageHistory
from langchain_core.messages import HumanMessage,AIMessage
from crew import run_crew
HISTORIES={}
def history(session_id): return HISTORIES.setdefault(session_id,InMemoryChatMessageHistory())
def invoke_chat(query,session_id):
    prior=history(session_id).messages
    if query.lower().strip() in ('what was my previous question?','repeat my previous question'):
        previous=next((m.content for m in reversed(prior) if isinstance(m,HumanMessage)),None)
        answer=previous or 'No previous question in this session.'
    else:
        answer=run_crew(query).answer
    history(session_id).add_user_message(query)
    history(session_id).add_ai_message(answer)
    return answer
# RunnableWithMessageHistory is included for LangChain-compatible session integration.
chain=RunnableWithMessageHistory(RunnableLambda(lambda x:x),history,input_messages_key='input')
