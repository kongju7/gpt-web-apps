
import streamlit as st 
import os 
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.schema import SystemMessage

# ------------------------------------------------------------------------------------------------
 
st.set_page_config(
    page_title = "InvestorAPP",
    page_icon = "📈"
)

st.title("InvestorAPP")

st.markdown("""
    안녕하세요. 투자를 도와주는 GPT 친구 **AI Kong**이에요. 

    투자에 관심있는 회사의 영문명을 입력하면 회사 재무 정보를 검토하여 
    
    투자에 대한 의견을 제시해 드려요. 
    
    **[참고] 주식 정보를 호출하기 위해서는 Alphavantage와 Google API Key가 추가로 필요합니다.**
    
    - [Alphavantage 사이트](https://www.alphavantage.co/support/#api-key)에서 **API Key**를 발급받고, 
    - [Google 검색 엔진 ID 발급 사이트](https://programmablesearchengine.google.com/controlpanel/create)에서는 **검색 엔진 ID**와
    - [Google API Key 발급 사이트](https://developers.google.com/custom-search/v1/introduction?hl=ko)에서는 **검색 엔진 API Key**를 모두 발급받아 왼쪽 창에 입력해 주세요.   
     

    Alphavantage의 무료 API는 호출이 하루 25번으로 제한된다는 점 주의해 주세요. 
    
            """)

# openai_api_key = st.secrets["OPENAI_API_KEY"]
# alphavantage_api_key = st.secrets["ALPHAVANTAGE_API_KEY"]
# GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# ------------------------------------------------------------------------------------------------


with st.sidebar:
    st.session_state.api_key = st.text_input("당신의 OpenAI API Key를 입력해 주세요.", type="password") 
OPENAI_API_KEY = st.session_state.api_key

# ------------------------------------------------------------------------------------------------

with st.sidebar:
    st.session_state.alphavantage_api_key = st.text_input("당신의 Alphavantage API Key를 입력해 주세요.", type="password")
    
with st.sidebar:
    st.session_state.GOOGLE_CSE_ID = st.text_input("당신의 Google CSE ID를 입력해 주세요.", type="password")

with st.sidebar:
    st.session_state.GOOGLE_API_KEY = st.text_input("당신의 Google API Key를 입력해 주세요.", type="password")

alphavantage_api_key = st.session_state.alphavantage_api_key        
GOOGLE_CSE_ID = st.session_state.GOOGLE_CSE_ID
GOOGLE_API_KEY = st.session_state.GOOGLE_API_KEY

# ------------------------------------------------------------------------------------------------

llm = ChatOpenAI(
    temperature = 0.1,
)

class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description = "The query you will search for.Example query: Stock Market Symbol for Apple Company")

class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company. 
    It takes a query as an argument. 
    """
    args_schema: Type[
        StockMarketSymbolSearchToolArgsSchema
    ] = StockMarketSymbolSearchToolArgsSchema
    
    def _run(self, query): 
        # search = DuckDuckGoSearchAPIWrapper()
        search = GoogleSearchAPIWrapper()
        return search.run(query)
  
class CompanyOverviewArgsSchema(BaseModel):
    symbol : str = Field(description = "Stock symbol of the company.Example: AAPL, TSLA")
  
class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
    Use this to get an overview of the financials of the company. 
    You should enter a stock symbol. 
    """
    args_schema: Type[
        CompanyOverviewArgsSchema
    ] = CompanyOverviewArgsSchema
    
    def _run(self, symbol): 
        r = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alphavantage_api_key}")
        return r.json()

class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this to get the income statement of a company. 
    You should enter a stock symbol. 
    """
    args_schema: Type[
        CompanyOverviewArgsSchema
    ] = CompanyOverviewArgsSchema
    
    def _run(self, symbol): 
        r = requests.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alphavantage_api_key}")
        return r.json()["annualReports"]

class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description = """
    Use this to get the weekly performance of a company stock. 
    You should enter a stock symbol. 
    """
    args_schema: Type[
        CompanyOverviewArgsSchema
    ] = CompanyOverviewArgsSchema
    
    def _run(self, symbol): 
        r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alphavantage_api_key}")
        response = r.json()
        return list(response["Weekly Time Series"].items())[:100] # 100주 정보
        
system_message = SystemMessage(content = """
        You are a hedge fund manager. 
        You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
        Consider the performance of a stock, the company overview and the income statement. 
        Be assertive in your judgement and recommend the stock or advise the user against it.
        USE KOREAN ONLY. 
        """)    
        
agent = initialize_agent(
    llm = llm, 
    verbose = True, 
    agent = AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors = True, 
    tools = [StockMarketSymbolSearchTool(),
             CompanyOverviewTool(), 
             CompanyIncomeStatementTool(),
             CompanyStockPerformanceTool()
    ],
    agent_kwargs = {
        "system_message": system_message
    }
)        

# ------------------------------------------------------------------------------------------------

company = st.text_input("투자에 관심있는 회사의 영문명을 입력해 주세요.(예: NVIDIA, APPLE)")

if company:
    result = agent.invoke(company)
    st.write(result["output"].replace("$", "\$"))

# ------------------------------------------------------------------------------------------------
 
with st.sidebar:    
    st.image('./images/DataKong.png', width = 100)
    
# ------------------------------------------------------------------------------------------------
 