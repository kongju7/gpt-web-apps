from typing import Any, Optional, Union
from uuid import UUID
from langchain.chat_models import ChatOllama
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st 
import time 

# -----------------------------------------------------------------------------------

st.set_page_config(
    page_title = "SecureDocumentAPP",
    page_icon = "🤐"
)


# -----------------------------------------------------------------------------------

st.title("SecureDocumentAPP")

st.markdown("""
    안녕하세요. Open-Source LLM 친구 **AI Kong**이에요. 
    
    파일을 업로드하면, Open-Source LLM 친구와 손잡고 그 파일의 내용에 대해 답변해 드려요. 
    
    로컬에 저장된 LLM 모델을 활용하기 때문에 DocumentAPP에 비해 보안성이 높습니다. 
    
    파일은 왼쪽 창에서 업로드 해주세요. 
    
    **※ [주의] 현재 로컬 LLM 모델은 용량 문제로 업로드 못한 상태로, 웹에서는 해당 기능을 사용하기 어렵다는 점 양해 부탁드립니다.**
    
            """)

# -----------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""**[참고] 로컬 LLM 모델을 활용하기 때문에 OpenAI API Key는 입력하지 않아도 됩니다.**""")  
    

with st.sidebar:
    file = st.file_uploader(".txt .pdf .docx 형식의 파일을 업로드하세요.", 
                            type = ["pdf", "txt", "docx"])

# -----------------------------------------------------------------------------------

# !pip install python-magic python-magic-bin
# !pip install tabulate pdf2image pytesseract

# -----------------------------------------------------------------------------------


class ChatCallbackHandler(BaseCallbackHandler): 
    
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
            
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")   

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model = "mistral:latest", 
    temperature = 0.1, 
    streaming = True,
    callbacks = [
        ChatCallbackHandler()
    ]
)

memory = ConversationSummaryBufferMemory(
    llm = llm,
    max_token_limit = 150,
    memory_key = "chat_history",
    return_messages = True,
)


# -----------------------------------------------------------------------------------

@st.cache_data(show_spinner = "파일을 임베딩 하는 중...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f: 
        f.write(file_content)
        
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n", 
        chunk_size = 1000, 
        chunk_overlap = 100 
        ) 
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter = splitter)
    embeddings = OllamaEmbeddings(
        model = "mistral:latest"
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append(
        {"message": message, "role": role}
    )

def save_memory(input, output):
    st.session_state["chat_history"].append(
        {"input": input, "output": output}
    )

def send_message(message, role, save = True):
    with st.chat_message(role):
        st.markdown(message)
    if save: 
        save_message(message, role)
        

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save = False)

def restore_memory():
    for history in st.session_state["chat_history"]:
        memory.save_context({"input": history["input"]}, {
                            "output": history["output"]})

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def load_memory(input):
    return memory.load_memory_variables({})["chat_history"]

def invoke_chain(chain, message):
    result = chain.invoke(message)
    save_memory(message, result.content)


# prompt = ChatPromptTemplate.from_template([
#             MessagesPlaceholder(variable_name="chat_history"),
#             """
#             Answer the question using ONLY the following context and not your training data. Answer it only in Korean. If you don't know the answer just say you don't know. DON'T make anything up.   
#             Context: {context}
#             Question: {question}
#             """
#         ])


# if file:
#     retriever = embed_file(file) 
#     send_message("답변할 준비가 완료되었습니다. 질문해 주십시오.", "ai", save = False)
#     restore_memory()
#     paint_history()
#     message = st.chat_input("업로드한 파일 내용에 관해 질문해 주세요...")
#     if message: 
#         send_message(message, "human")
#         chain = {
#             "context": retriever | RunnableLambda(format_docs),
#             "chat_history": load_memory,
#             "question": RunnablePassthrough()
#         } | prompt | llm
        
#         with st.chat_message("ai"):
#             invoke_chain(chain, message)
# else: 
#     st.session_state["messages"] = []
#     st.session_state["chat_history"] = []    
    
# ------------------------------------------------------------------------------------------------
 
with st.sidebar:    
    st.image('./images/DataKong.png', width = 100)
    
# ------------------------------------------------------------------------------------------------
 