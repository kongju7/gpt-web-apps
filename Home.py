import streamlit as st 

# ------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title = "Kong's GPT Web Apps",
    page_icon = "🗃️"
)

# ------------------------------------------------------------------------------------------------

with st.sidebar:
    st.session_state.api_key = st.text_input("당신의 OpenAI API Key를 입력해 주세요.", type="password")
openai_api_key = st.session_state.api_key

# ------------------------------------------------------------------------------------------------

st.markdown(
    """
# Kong's GPT Web Apps

이 사이트는 **OpenAI**(**`GPT`** & **`Whisper`**) **API**를 활용하여 몇 가지 애플리케이션(기본형)을 직접 구현해 본 사이트입니다.   
  
  
  
- <a href="/DocumentAPP" target="_self">DocumentAPP</a> : 문서 톺아보기 
- <a href="/SecureDocumentAPP" target="_self">SecureDocumentAPP</a> : 로컬 LLM 활용한 보안문서 톺아보기 
- <a href="/QuizAPP" target="_self">QuizAPP</a> : 문제 출제하고 맞추기 
- <a href="/SiteAPP" target="_self">SiteAPP</a> : 사이트 톺아보기 
- <a href="/MeetingAPP" target="_self">MeetingAPP</a> : **Whisper AI**를 활용한 비디오 음성인식 및 요약   
- <a href="/InvestorAPP" target="_self">InvestorAPP</a> : 투자의견 제공    

-----    
    
##### 참고사항
- 애플리케이션을 웹 상에서 직접 실행해 보기 위해서는 당신의 **API Key**를 직접 입력해야 합니다.  
- **SecureDocumentAPP**은 로컬 LLM 모델을 사용하는 애플리케이션으로, 현재 웹에서는 실행할 수 없습니다.
    
-----   
##### 애플리케이션 실행을 위해 필요한 API Key     
- [ ] [OpenAI API Key](https://platform.openai.com/account/api-keys)  
- [ ] [Alphavantage API Key](https://www.alphavantage.co/support/#api-key)  
- [ ] [Google CSE ID](https://programmablesearchengine.google.com/controlpanel/create) - '검색 엔진 ID'
- [ ] [Google API Key](https://developers.google.com/custom-search/v1/introduction?hl=ko) - '키 가져오기'   

    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------

with st.sidebar:
    st.image('./images/DataKong.png', width = 100)
        
# ------------------------------------------------------------------------------------------------