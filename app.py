from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

__import__("pysqlite3")
import sys,os
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import streamlit as st
import sqlite3, json
import asyncio
import uuid

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma(persist_directory='gita_vectordb', embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_type='mmr',search_kwargs={'k':3, 'fetch_k':20, 'lambda_mult':0.5})

# model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')
model = ChatGroq(model="llama-3.1-8b-instant")

mqr = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=model
)

prompt = ChatPromptTemplate([
    ('system',"""
      You are a helpful Hindu spiritual assistant with deep knowledge of the Bhagavad Gita.
You are given verses in Sanskrit, their translations, and purports as context.

Your rules:
- Always print the original Sanskrit verse for reference.
- Use the provided context first to answer.
- If necessary, you may also rely on your prior knowledge of the Bhagavad Gita.
- Provide the translation and explanation (purport) in clear English.
- Stay strictly within Bhagavad Gita knowledge. Do not invent or hallucinate.
- If the context is insufficient, say "I don’t know based on the provided verses."
- Stay focused only on the Bhagavad Gita and spiritual wisdom derived from it.
- Be clear, concise, and respectful in your explanation.
- Don't Provide long answer keep it 100-200 words.
- Provide chapter number along with verse number whenever it is reffered.

Context:
{context}

Question:
{question}

Answer as a knowledgeable guide of the Bhagavad Gita:"""),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{question}')
])

rewrite_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
You are a query rewriter. Based on the conversation so far and the latest user question,
rewrite the question so it is a standalone query about the Bhagavad Gita.

Chat History:
{chat_history}

User Question:
{question}

Rewritten Question:
"""
)

chat_history = []

# def format_doc(retrieved_docs):
#   context_text = "\n\n".join(docs.page_content for docs in retrieved_docs)
#   return context_text
def format_doc(docs):
    context_text = ''
    for d in docs:
        context_text += f"""
Verse Reference: Chapter {d.metadata['chapter']}, Verse {d.metadata['verse']}
Sanskrit: {d.metadata['sanskrit']}
Translation: {d.metadata['translation']}
Purport: {d.metadata['purport']}
---------------------------------------
"""
    return context_text


def human_query(chat_history):
  return chat_history[-1].content

def update_query(query):
    chat_history[-1].content = query
    return query

def rewrite_prompt_input(query):
    return {'chat_history': chat_history, 'question': query}

parser = StrOutputParser()

parallel_chain = RunnableParallel({
  'context': RunnableLambda(human_query) | RunnableLambda(rewrite_prompt_input) | rewrite_prompt | model | parser | RunnableLambda(update_query) | retriever | RunnableLambda(format_doc),
  'chat_history': RunnablePassthrough(), 
  'question': RunnableLambda(human_query)  
})

chain = parallel_chain | prompt | model | parser

user_input = st.chat_input('Ask Your question to Bhagavad Gita')

conn = sqlite3.connect("chat_history.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    name TEXT,
    message_history TEXT,   -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


def save_history(session_id, history):
    history_json = json.dumps(history)
    c.execute("UPDATE sessions SET message_history=? WHERE session_id=?", 
              (history_json, session_id))
    conn.commit()

def create_session(user_id, name="New Chat"):
    c.execute("INSERT INTO sessions (user_id, name, message_history) VALUES (?, ?, ?)",
              (user_id, name, json.dumps([])))
    conn.commit()
    return c.lastrowid

def load_history(session_id):
    row = c.execute("SELECT message_history FROM sessions WHERE session_id=?", 
                    (session_id,)).fetchone()
    if row and row[0]:
        return json.loads(row[0])
    return []

def get_sessions(user_id):
    return c.execute(
        "SELECT session_id, name, created_at FROM sessions WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()

def delete_session(session_id):
    c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()

def update_session_name(session_id, new_name):
    c.execute("UPDATE sessions SET name=? WHERE session_id=?", (new_name, session_id))
    conn.commit()

def generate_title(user_message, ai_message):
    title_prompt = f"Summarize this conversation in 3-5 words:\nUser: {user_message}\nAI: {ai_message}\nTitle:"
    return model.invoke(title_prompt).content

def build_chat_history():
    history = []
    for msg in st.session_state["message_history"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            history.append(SystemMessage(content=msg["content"]))
    return history

if "user_id" not in st.session_state:
    # auto-generate first-time user
    st.session_state["user_id"] = str(uuid.uuid4())

sessions = get_sessions(st.session_state["user_id"])

with st.sidebar:
    st.header("Chat Sessions")
    sessions = get_sessions(st.session_state["user_id"])

    if st.button("➕ New Chat"):
        new_id = create_session(st.session_state["user_id"], f"Chat {len(sessions)+1}")
        st.session_state["current_session"] = new_id
        st.session_state["message_history"] = []

    for sid, name, created_at in sessions:
        col1, col2 = st.columns([4,1])  # session name + delete button
        with col1:
            if st.button(name, key=f"load_{sid}"):
                st.session_state["current_session"] = sid
                st.session_state["message_history"] = load_history(sid)
        with col2:
            if st.button("❌", key=f"delete_{sid}"):
                delete_session(sid)
                if "current_session" in st.session_state and st.session_state["current_session"] == sid:
                    st.session_state.pop("current_session")
                    st.session_state.pop("message_history", None)
                st.rerun()  # refresh UI after delete
    
    st.markdown("---")
    st.subheader("User ID")

    user_id_input = st.text_input("Enter your User ID", value=st.session_state.get("user_id", ""))

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Set User ID"):
            if user_id_input.strip():
                st.session_state["user_id"] = user_id_input.strip()
                st.rerun()

    with col2:
        if st.button("Generate New ID"):
            new_id = str(uuid.uuid4())
            st.session_state["user_id"] = new_id
            st.rerun()

# Default to first session if none selected
if "current_session" not in st.session_state:
    if sessions:
        st.session_state["current_session"] = sessions[0][0]
        st.session_state["message_history"] = load_history(sessions[0][0])
    else:
        new_session = create_session(st.session_state["user_id"], "Chat 1")
        st.session_state["current_session"] = new_session
        st.session_state["message_history"] = []

# Show messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])

if user_input:

  st.session_state['message_history'].append({'role':'user', 'content': user_input})
  with st.chat_message('user'):
    st.text(user_input)

  chat_history = build_chat_history()

  with st.chat_message('assistant'):
    # ai_message = chain.invoke(chat_history)
    
    # st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})
    # st.text(ai_message)

    def response_generator():
        full_response = ""
        for chunk in chain.stream(chat_history):  # stream from chain
            full_response += chunk
            yield chunk
        # return full response when generator completes
        st.session_state["last_ai_response"] = full_response

    # Stream directly into UI
    streamed_text = st.write_stream(response_generator())

    # Save final AI response in history & DB
    st.session_state["message_history"].append(
        {"role": "assistant", "content": streamed_text}
    )
    chat_history.append(AIMessage(content=streamed_text))
    save_history(
        st.session_state["current_session"],
        st.session_state["message_history"]
    )

  current_session_id = st.session_state["current_session"]
  row = [s for s in sessions if s[0] == current_session_id][0]
  _, current_name, _ = row  

  if current_name.startswith("Chat "):
    title = generate_title(user_input, streamed_text)
    update_session_name(current_session_id, title)

