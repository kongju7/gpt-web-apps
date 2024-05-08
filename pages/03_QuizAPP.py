from langchain.retrievers import WikipediaRetriever 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import streamlit as st 
import json 

# ------------------------------------------------------------------------------------------------


st.set_page_config(
    page_title = "QuizAPP",
    page_icon = "✍️"
)

st.title("QuizAPP")

# -----------------------------------------------------------------------------------

with st.sidebar:
    st.session_state.api_key = st.text_input("당신의 OpenAI API Key를 입력해 주세요.", type="password") 

openai_api_key = st.session_state.api_key
    

# ------------------------------------------------------------------------------------------------

llm = ChatOpenAI(
    temperature = 0.1, 
    model = "gpt-3.5-turbo-1106", 
    streaming = True, 
    callbacks = [
        StreamingStdOutCallbackHandler()
    ]
)


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """
             You are a helpful assistant assuming the role of an educator.
             
             Based ONLY on the following context create 10 questions in Korean to evaluate the user's understanding of the text.
             
             For each question, provide four potential answers, indicating the correct one with an (o).
             
             Use Korean only! 
             
             Question examples: 
             
             Question: 바다의 색깔은 무엇인가요? 
             Answers: 빨간색| 노란색| 초록색| 파란색(o)

             Question: 대한민국의 수도는 어디인가요? 
             Answers: 부산| 서울(o)| 인천| 대구 
             
             Question: 영화 아바타가 개봉한 연도는 언제인가요? 
             Answers: 2007| 2001| 2009(o)| 1998                         
             
             Question: 마이클 잭슨의 직업은 무엇인가요? 
             Answers: 가수(o)| 교수| 화가| 청소원 
              
             Your turn! 
             
             Context: {context}
             """
             )
            
        ]
    )

questions_chain = {
        "context": format_docs
        } | questions_prompt | llm 


formatting_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """
             You are a powerful formatting algorithm. 
             
             You format exam questions into JSON format. 
             Answer with (o) are the correct ones. 
 
             Example Input: 
                              
             Question: 바다의 색깔은 무엇인가요? 
             Answers: 빨간색| 노란색| 초록색| 파란색(o)

             Question: 대한민국의 수도는 어디인가요? 
             Answers: 부산| 서울(o)| 인천| 대구 
             
             Question: 영화 아바타가 개봉한 연도는 언제인가요? 
             Answers: 2007| 2001| 2009(o)| 1998                         
             
             Question: 마이클 잭슨의 직업은 무엇인가요? 
             Answers: 가수(o)| 교수| 화가| 청소원 
             
             
             Example Output: 
             
             ```json
             {{"questions": [
                {{
                    "qeustion": "바다의 색깔은 무엇인가요?", 
                    "answers": [
                        {{
                            "answer": "빨간색", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "노란색", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "초록색", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "파란색",
                            "correct": true 
                        }}
                                ]
                }},
                {{
                    "qeustion": "대한민국의 수도는 어디인가요?", 
                    "answers": [
                        {{
                            "answer": "부산", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "서울", 
                            "correct": true
                        }}, 
                        {{ 
                            "answer": "인천", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "대구",
                            "correct": false                     
                        }}
                                ]
                }},
                {{     
                    "qeustion": "영화 아바타가 개봉한 연도는 언제인가요?", 
                    "answers": [
                        {{
                            "answer": "2007", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "2001", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "2009", 
                            "correct": true
                        }}, 
                        {{ 
                            "answer": "1998",
                            "correct": false       
                        }}
                                ]                        
                }},
                {{     
                    "qeustion": "마이클 잭슨의 직업은 무엇인가요?", 
                    "answers": [
                        {{
                            "answer": "가수", 
                            "correct": true
                        }}, 
                        {{ 
                            "answer": "교수", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "화가", 
                            "correct": false 
                        }}, 
                        {{ 
                            "answer": "청소원",
                            "correct": false       
                        }}
                            ]
                }}
                        ]
            }}
        
             ```
        Your turn! 
        
        Quesions: {context}
             
            """
        )
        ]
)

formatting_chain = formatting_prompt | llm 

# -----------------------------------------------------------------------------------

@st.cache_data(show_spinner = "파일을 불러오는 중...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f: 
        f.write(file_content)
        
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n", 
        chunk_size = 1000, 
        chunk_overlap = 100 
        ) 
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter = splitter)
    return docs

@st.cache_data(show_spinner = "퀴즈 생성 중...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser 
    return chain.invoke(_docs)

@st.cache_data(show_spinner = "위키피디아 검색 중...")
def wiki_search(term):
    retriever = WikipediaRetriever(
                            lang = "ko", 
                            top_k_result = 5)
    docs = retriever.get_relevant_documents(term)
    return docs

# -----------------------------------------------------------------------------------

with st.sidebar:
    docs = None
    choice = st.selectbox("선택해 주세요.", 
                            (
                            "파일", 
                            "위키피디아 문서"
                            ), 
                          )
    if choice == "파일":
        file = st.file_uploader(".txt .pdf .docx 형식의 파일을 업로드하세요.", 
                                type = ["pdf", "txt", "docx"])  
        if file:
            docs = split_file(file)
    else: 
        topic = st.text_input("위키피디아 검색...")
        if topic:
            docs = wiki_search(topic)
            # st.write(docs)

# ------------------------------------------------------------------------------------------------

if not docs: 
    st.markdown(
        """
    안녕하세요. 문제내기를 좋아하는 GPT 친구 **AI Kong**이에요. 

    직접 업로드한 파일에 대해 문제를 내거나, 
    
    업로드할 파일이 없다면 위키피디아에 있는 내용을 검색해 문제를 출제해 맞춰보기를 할 수 있어요. 
    
    파일은 왼쪽 창에서 업로드 해주세요. 
    
    문제를 출제하는 데에 시간이 다소 소요될 수 있습니다.    
        """
    )
else:     
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"]):
            st.write(question["question"])
            value = st.radio(
                "정답을 선택해 주세요.",
                [answer["answer"] for answer in question["answers"]],
                index = None,
                key = idx
            )
            
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("정답입니다!")
            elif value is not None:
                st.error("오답입니다. 다시 한번 생각해 보세요.")
        
        button = st.form_submit_button("정답 제출")


# ------------------------------------------------------------------------------------------------

with st.sidebar:    
    st.image('./images/DataKong.png', width = 100)
    
# ------------------------------------------------------------------------------------------------
 