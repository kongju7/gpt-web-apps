import streamlit as st 
import openai
from moviepy.editor import *
from pydub import AudioSegment
import math 
import glob
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

# ------------------------------------------------------------------------------------------------


class ChatCallbackHandler(BaseCallbackHandler): 
    
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
            
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")   

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature = 0.1,
)

chat_llm = ChatOpenAI(
    temperature = 0.1,
    streaming = True,
    callbacks = [ChatCallbackHandler()],
)

memory = ConversationBufferMemory(
    llm = chat_llm,
    max_token_limit = 150,
    memory_key = "chat_history",
    return_messages = True,
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 800,
    chunk_overlap = 100,
)


# -------------------------------------------------------------------------------------------


@st.cache_data()
def extract_audio_from_video(video_path):
    if video_path.endswith((".mp4", ".avi", ".mkv", ".mov")):
        audio_path = video_path.replace(video_path.split('.')[-1], "mp3")
    transcript_path = audio_path.replace(".mp3", ".txt")
    if os.path.exists(transcript_path):
        return 
    if os.path.exists(audio_path):
        return 
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    transcript_path = audio_path.replace(".mp3", ".txt")
    if os.path.exists(transcript_path):
        return 
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000 
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunks_{i}.mp3", format = "mp3") 


@st.cache_data()
def transcribe_chunks(chunk_folder, transcript_path):
    if os.path.exists(transcript_path):
        return 
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(transcript_path, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1", 
                audio_file, 
            )
            text_file.write(transcript.text)
            
            
@st.cache_data()
def save_summary(transcript_path, summary):
    summary_path = transcript_path.replace(".txt", "_sum.txt")
    with open(summary_path, "w") as summary_file:
        summary_file.write(summary)


@st.cache_data()
def save_summary_kr(transcript_path, summary_translation):
    summary_kr_path = transcript_path.replace(".txt", "_sum_kr.txt")
    with open(summary_kr_path, "w") as summary_kr_file:
        summary_kr_file.write(summary_translation)

# ------------------------------------------------------------------------------------------------
 
@st.cache_data()
def embed_file(file_path):        
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter = splitter)
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

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


prompt = ChatPromptTemplate.from_messages([
    ("system", 
            """
            Answer the question using ONLY the following context. Answer it only in Korean. If you don't know the answer just say you don't know. DON'T make anything up.
            Context: {context}
            """), 
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{question}")
    ])

# ------------------------------------------------------------------------------------------------
 

st.set_page_config(
    page_title = "MeetingAPP",
    page_icon = "📝"
)

st.title("MeetingAPP")

st.markdown("""
    안녕하세요. GPT Whisper 친구 **AI Kong**이에요. 
    
    비디오 파일을 업로드하면, 전사 텍스트(transcript)나 요약본을 제공해 드리거나
    
    내용에 대해 묻고 답할 수 있는 챗봇을 제공해 드려요.
    
    비디오 파일은 왼쪽 창에서 업로드 해주세요. 
    
            """)

# ------------------------------------------------------------------------------------------------

if "api_key" not in st.session_state:
    st.warning("왼쪽 사이드바에서 당신의 OpenAI API Key를 입력해 주세요.")
    
    with st.sidebar:
        st.session_state.api_key = st.text_input("당신의 OpenAI API Key를 입력해 주세요.", type="password") 
else: 
    openai_api_key = st.session_state.api_key
    
# ------------------------------------------------------------------------------------------------
 
with st.sidebar: 
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])
    
if video:
    chunks_folder = "./.cache/chunks"
    with st.status("비디오 로딩 중...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        if video_path.endswith((".mp4", ".avi", ".mkv", ".mov")):
            audio_path = video_path.replace(video_path.split('.')[-1], "mp3")
        transcript_path = audio_path.replace(".mp3", ".txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label = "오디오 추출 중...")
        extract_audio_from_video(video_path)
        status.update(label = "오디오 분할 중...")      
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label = "오디오 전사 중...")
        transcribe_chunks(chunks_folder, transcript_path)
        
    transcript_tab, summary_tab, trans_tab, qa_tab = st.tabs(
        [
            "전사", 
            "요약", 
            "요약 한글 번역",
            "Q&A"
        ]
    )

    with transcript_tab:
        if not os.path.exists(transcript_path):
            transcribe_chunks(chunks_folder, transcript_path)
        with open(transcript_path, "r") as file:
            st.write(file.read())
       
    with summary_tab:
        start = st.button("요약본 생성")
        if start:
            summary_path = transcript_path.replace(".txt", "_sum.txt")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as summary_file:
                    summary = summary_file.read()
            else:
                loader = TextLoader(transcript_path)
                docs = loader.load_and_split(text_splitter = splitter)
                first_summary_prompt = ChatPromptTemplate.from_template("""
                    Write a concise summary of the following:
                    "{text}"
                    CONCISE SUMMARY:                                                   
                """)
                first_summary_chain = first_summary_prompt | llm | StrOutputParser()
                
                summary = first_summary_chain.invoke({
                    "text": docs[0].page_content
                })
                
                refine_prompt = ChatPromptTemplate.from_template(
                    """
                    Your job is to produce a final summary. 
                    We have provided an exisiting summary up to a certain point: {existing_summary}
                    We have the opportunity to refine the existing summary (only if needed) with some more context below.
                    --------------
                    {context}
                    --------------
                    Given the new context, refine the original summary.
                    If the context isn't useful, RETURN the original summary.
                    """
                )
                refine_chain = refine_prompt | llm | StrOutputParser()
                
                with st.status("요약 중...") as status:
                    for i, doc in enumerate(docs[1:]):
                        status.update(label = f"{i+1} / {len(docs)-1}번째 문서 처리 중...")
                        summary = refine_chain.invoke(
                            {
                            "existing_summary": summary, 
                            "context": doc.page_content
                            }
                        )
            st.write(summary)
                
            save_summary(transcript_path, summary)
                


    with trans_tab:            
        translate = st.button("요약본 한글 번역 생성")
        if translate:
            summary_kr_path = transcript_path.replace(".txt", "_sum_kr.txt")
            if os.path.exists(summary_kr_path):
                with open(summary_kr_path, "r") as summary_kr_file:
                    summary_translation = summary_kr_file.read()
            else:
                summary_path = video_path.replace(".mp4", "_sum.txt")
                with open(summary_path, "r") as summary_file:
                    summary = summary_file.read()  
                
                translate_prompt = ChatPromptTemplate.from_template("""
                        Translate the following "{text}" into Korean
                        TRANSLATE:                                                   
                    """)
                translate_chain = translate_prompt | llm | StrOutputParser()

                summary_translation = translate_chain.invoke({
                        "text": summary
                    })
                save_summary_kr(transcript_path, summary_translation)
               
            st.write(summary_translation)
            
            
    with qa_tab:
        retriever = embed_file(transcript_path)
        send_message("답변할 준비가 완료되었습니다. 질문해 주십시오.", "ai", save = False)
        restore_memory()        
        paint_history()       
        message = st.text_input("업로드한 파일 내용에 관해 질문해 주세요...")
        if message: 
            send_message(message, "human")
            chain = {
                        "context": retriever | RunnableLambda(format_docs),
                        "chat_history": load_memory,
                        "question": RunnablePassthrough()
                    } | prompt | chat_llm
                    
            with st.chat_message("ai"):
                response = chain.invoke(message)    
        else: 
            st.session_state["messages"] = []
            st.session_state["chat_history"] = []
else: 
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []
       
# ------------------------------------------------------------------------------------------------
 
with st.sidebar:    
    st.image('./images/DataKong.png', width = 100)
    
# ------------------------------------------------------------------------------------------------
 