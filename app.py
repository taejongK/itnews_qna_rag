import streamlit as st
from PIL import Image

# 챗봇 애플리케이션 제목
st.title("Streamlit 챗봇 애플리케이션")

# Keywords image 섹션
st.header("Keywords Image")
if st.button("이미지 보기"):
    # 이미지 경로를 지정
    path = "/Users/kimtaejong/Personal/Github/news_rag/wordscloud/wordcloud_2024-11-07-2024-11-08.png"
    image = Image.open(path)  # 여기에 이미지 경로를 지정하세요
    st.image(image, caption="키워드 이미지", use_column_width=True)

# Summary 섹션
st.header("Summary")
keyword = st.text_input("키워드를 입력하세요:")
if st.button("뉴스 요약 가져오기"):
    if keyword:
        # 키워드에 대한 요약 내용을 가져오는 로직
        summary_text = f"'{keyword}'에 대한 뉴스 요약입니다. 자세한 내용은 여기에 표시됩니다."  # 요약 내용 생성 예시
        st.write(summary_text)
    else:
        st.warning("키워드를 입력해주세요.")

# QnA 섹션
st.header("QnA")
qna_input = st.text_input("질문을 입력하세요:")
if st.button("질문하기"):
    if qna_input:
        # 질문에 대한 답변 로직
        response_text = f"'{qna_input}'에 대한 답변을 여기에 표시합니다."  # 답변 예시
        st.write(response_text)
    else:
        st.warning("질문을 입력해주세요.")
