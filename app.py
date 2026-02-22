import os
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import gspread
from datetime import datetime
import pytz

# 1. API 키 설정 (스트림릿 금고에서 가져오기)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Streamlit 웹 페이지 설정
st.set_page_config(page_title="성인간호학 AI 튜터", page_icon="🏥")
st.title("🏥 성인간호학 AI 튜터")
st.markdown(
    "강의록과 실라버스 기반으로 질문에 답변해 드립니다. 모호한 내용은 교수님께 문의하도록 안내합니다."
)


# 2. 구글 스프레드시트 연결 설정 (스트림릿 금고 사용)
@st.cache_resource
def init_google_sheet():
    try:
        # 스트림릿 금고에 숨겨둔 구글 키(JSON)를 파이썬 사전 형태로 변환해서 읽기
        google_secret_str = st.secrets["GOOGLE_SECRET"]
        creds_dict = json.loads(google_secret_str)
        gc = gspread.service_account_from_dict(creds_dict)

        sh = gc.open("챗봇_질문기록")  # 교수님 엑셀 파일명
        worksheet = sh.sheet1

        if len(worksheet.get_all_values()) == 0:
            worksheet.append_row(["시간", "학생 질문", "AI 튜터 답변"])

        return worksheet
    except Exception as e:
        st.warning(f"⚠️ 구글 시트 연결에 실패했습니다. 에러 원인:{e}")
        return None


sheet = init_google_sheet()


# 3. RAG 파이프라인 구축
@st.cache_resource
def init_rag_pipeline():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = """
    [역할 및 페르소나]
    당신은 간호대학생의 '성인간호학(Adult Nursing)' 학습 및 교과목 이수를 돕는 전문적이고 정확한 AI 튜터 챗봇입니다. 학생의 질문에 친절하고 학구적인 태도로 답변합니다.

    [지식 및 정보 제공 원칙 (매우 중요)]
    1. 철저한 자료 기반: 당신은 오직 아래에 제공된 '제공된 자료(Context)' 내에서만 정보를 검색하고 답변해야 합니다.
    2. 검증된 출처 제한: 의학 및 간호학 지식과 관련된 답변은 제공된 자료 중에서도 출처가 분명한 범위의 내용만 바탕으로 제공하십시오.
    3. 임의 추론 금지: 제공된 자료에 명시되어 있지 않은 사실을 스스로 추론하거나 지어내서 답변하지 마십시오.

    [교과목 문의 및 예외/에스컬레이션 처리]
    1. 교과목 운영에 대한 문의는 제공된 자료 내에서만 답변하십시오.
    2. 교수자 이관(Escalation): 제공된 자료에 답이 없거나 모호한 경우 반드시 다음 문구를 출력하십시오.
       "해당 내용은 제공된 강의 자료에서 명확한 확인이 어렵거나, 추가적인 전문적 해석이 필요합니다. 정확한 학습과 임상 적용을 위해 담당 교수님께 직접 문의해 주시기 바랍니다."

    [답변 형식 규정]
    - 전문적인 간호학 용어를 정확하게 사용하되, 문맥을 쉽게 풀어 설명하십시오.
    - 참고한 [출처: 문서명, 페이지]를 대괄호 안에 명시하십시오.

    제공된 자료(Context):
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


rag_chain = init_rag_pipeline()

# 4. Streamlit 채팅 UI 구성 및 데이터 저장 로직
if rag_chain is None:
    st.error("⚠️ 'data' 폴더에 PDF 파일이 없습니다. 자료를 넣고 새로고침 해주세요.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "성인간호학에 대해 질문해 주세요 (예: 폐렴 환자의 간호 중재는?)"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("자료를 검색하고 답변을 작성 중입니다..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)

                with st.expander("참고한 문서 조각 확인하기"):
                    for doc in response["context"]:
                        st.write(
                            f"- {doc.metadata['source']} (Page {doc.metadata['page']})"
                        )

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # 5. 질문과 답변을 구글 시트에 자동 기록!
        if sheet is not None:
            try:
                kst = pytz.timezone("Asia/Seoul")
                now = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_row([now, prompt, answer])
            except Exception as e:
                print(f"시트 저장 에러: {e}")

