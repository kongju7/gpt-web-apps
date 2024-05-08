
from langchain.chat_models import ChatOpenAI 
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import re

# ------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title = "SiteAPP",
    page_icon = "🖥️"
)

st.title("SiteAPP")

# ------------------------------------------------------------------------------------------------

with st.sidebar:
    st.session_state.api_key = st.text_input("당신의 OpenAI API Key를 입력해 주세요.", type="password") 

openai_api_key = st.session_state.api_key

# ------------------------------------------------------------------------------------------------


llm = ChatOpenAI(
        temperature = 0.1, 
        streaming = True
        )

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question.  If the answer is unknown, simply state that you don't know. Don't make anything up.
                                                  
    Afterwards, rate the answer on a scale from 0 to 5.
    A high score should be given if the answer directly addresses the user's question, while a lower score is appropriate if it does not. 
    Make sure to always include the score for the answer, even if the score is 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
"""
)

def get_answers(inputs): 
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm 

    return {"question": question, 
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {"question": question, 
                        "context": doc.page_content
                        }).content, 
                    "source": doc.metadata["source"], 
                    "date":doc.metadata["lastmod"]
                } for doc in docs
                        ]
            }
    
choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Return the sources of the answers as they are, do not change them.
            
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)
    
    
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm 
    
    condensed = "\n\n".join(
                    f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
                    for answer in answers)
        
    return choose_chain.invoke({
            "question": question, 
            "answers": condensed,
            })

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header: 
        header.decompose()
    if footer:
        footer.decompose()
    return (str(soup.get_text())
            .replace("\n", " ")
            .replace("\xa0", " ")
            .replace("CloseSearch Submit Blog", ""))


@st.cache_data(show_spinner = "웹사이트를 로딩 중입니다...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size = 1000, 
                chunk_overlap = 200 
                ) 
    loader = SitemapLoader(
                url,
                parsing_function = parse_page, 
                )
    regex = re.compile(r"(www.)*\w+\.\w{2,3}")
    cache_dir = LocalFileStore(
        f"./.cache/site_embeddings/{re.search(regex, url).group()}")
    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder,
        cache_dir
    )
    loader.requests_per_second = 2 # slower down
    docs = loader.load_and_split(text_splitter = splitter)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


# -----------------------------------------------------------------------------------


st.markdown(
    """
    안녕하세요. GPT 친구 **AI Kong**이에요. 

    톺아보고 싶은 웹사이트의 주소를 알려주면, 
    
    GPT 친구와 손잡고 웹사이트의 내용에 대해 답변해 드려요. 
    
    웹사이트 주소는 왼쪽 창에서 입력해 주세요.        
        """
    )

# ------------------------------------------------------------------------------------------------

with st.sidebar: 
    url = st.text_input("웹사이트 주소를 입력해 주세요...", 
                        placeholder = "https://example.com")
    

# !playwright install
# !playwright install-deps
# [Optional] !pip install pytest-playwright

# ------------------------------------------------------------------------------------------------
   
if url:
    if ".xml" not in url:
        with st.sidebar: 
            st.error("Sitemap URL을 입력해 주세요.")
    else: 
        retriever = load_website(url)
        query = st.text_input("웹사이트 내용에 대해서 질문해 주세요.")
        if query:
            chain = ({
                "docs": retriever, 
                "question": RunnablePassthrough()
                    } | RunnableLambda(get_answers) | RunnableLambda(choose_answer))
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))


# ------------------------------------------------------------------------------------------------
 
with st.sidebar:    
    st.image('./images/DataKong.png', width = 100)
    
# ------------------------------------------------------------------------------------------------
 