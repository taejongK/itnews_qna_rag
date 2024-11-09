import os

import streamlit as st
from PIL import Image
from models.wordcloud import *
from models.service import *

from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
faiss_path = 'data/faiss_itnews'
itnews_df = get_news_df("data/itnews.db")

# 챗봇 애플리케이션 제목
st.title("Streamlit 챗봇 애플리케이션")

# Keywords image 섹션
st.header("Keywords Image")
if st.button("이미지 보기"):
    plt = get_wordcloud(itnews_df)
    st.pyplot(plt)

# Summary 섹션
st.header("Summary")
keyword = st.text_input("키워드를 입력하세요:")
if st.button("뉴스 요약 가져오기"):
    if keyword:
        # 키워드에 대한 요약 내용을 가져오는 로직
        # 요약 내용 생성 예시
        # summary_text = f"'{keyword}'에 대한 뉴스 요약입니다. 자세한 내용은 여기에 표시됩니다."
        selected_summary = itnews_df[itnews_df.keywords.apply(
            lambda x: keyword in x)].summary.tolist()
        summary_text = load_summary(
            selected_summary, api_key=os.getenv("OPENAI_API_KEY"))
        st.write(summary_text)
    else:
        st.warning("키워드를 입력해주세요.")

# QnA 섹션
st.header("QnA")
qna_input = st.text_input("질문을 입력하세요:")
if st.button("질문하기"):
    if qna_input:
        # 질문에 대한 답변 로직
        # response_text = f"'{qna_input}'에 대한 답변을 여기에 표시합니다."  # 답변 예시
        qna_chat = ITNewsQnA(api_key=os.getenv(
            "OPENAI_API_KEY"), faiss_path=faiss_path)  # QnA 객체 생성
        rag_chain = qna_chat.build_rag_chain()  # RAG 모델 생성
        answer, title_list, source_list = qna_chat.get_answer(qna_input, rag_chain)  # 질문에 대한 답변
        st.write(answer)
        for title, source in zip(title_list, source_list):
            st.write(f"Title: {title}, Source: {source}")
    else:
        st.warning("질문을 입력해주세요.")
