import requests
import bs4
from bs4 import BeautifulSoup
import re

import pandas as pd
import sqlite3
from datetime import date

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# 특정 섹션에서 원하는 클래스 내부의 a 태그에서 기사 링크 수집 함수


def get_article_links(section_url, num_articles=10):
    response = requests.get(section_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # "newsct_wrapper _GRID_TEMPLATE_COLUMN _STICKY_CONTENT" 클래스 내부의 a 태그만 선택
    links = []
    wrapper = soup.find(
        "div", class_="newsct_wrapper _GRID_TEMPLATE_COLUMN _STICKY_CONTENT")
    if wrapper:
        for a_tag in wrapper.find_all("a", href=True):
            href = a_tag['href']
            # 기사 링크인 경우에만 links 리스트에 추가
            if re.match(r'^https://n\.news\.naver\.com/mnews/article/\d+/\d+$', href):
                links.append(href)
            if re.match(r'^https://n\.news\.naver\.com/mnews/hotissue/article/', href):
                links.append(href)

    # 중복 제거 후, 상위 num_articles 개의 기사 링크 반환
    unique_links = list(set(links))
    return unique_links[:num_articles]


def get_article_data(link):
    loader = WebBaseLoader(
        web_paths=(link,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["media_end_head_title",
                                 "newsct_article _article_body",
                                 "go_trans _article_content"]},
            )
        ),
    )
    return loader


if __name__ == "__main__":
    # IT/과학 섹션 URL
    section_url = "https://news.naver.com/section/105"
    article_links = get_article_links(section_url, num_articles=100)

    # 각 기사 URL에서 기사 내용을 수집하는 코드
    documents = []
    for link in article_links:
        # loader = WebBaseLoader(link)
        loader = get_article_data(link)
        try:
            # print(link)
            documents.extend(loader.load())
        except Exception as e:
            print(e)
            pass

    # documents 리스트에 각 기사의 내용이 저장됨
    print(f"Collected {len(documents)} articles")

    summary_keywords_prompt = PromptTemplate.from_template(
        """Extract a brief summary and five keywords from the following text. Ensure that the output is Korean. The output should be formatted in JSON as follows:

    {{
    "summary": "Your summary here",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
    }}

    Text: {input_text}"""
    )

    # 문서 요약 추출 봇 모델
    summary_keywords_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0)
    extract_chain3 = (
        {'input_text': RunnablePassthrough()}
        | summary_keywords_prompt
        | summary_keywords_llm
        | StrOutputParser()
    )

    today_date = date.today().strftime("%Y-%m-%d")

    itnews_data = []
    for doc in documents:
        # 요약과 키워드 추출
        sumandkey = eval(extract_chain3.invoke(doc.page_content))

        source = doc.metadata['source']  # primary key
        date = today_date
        title = doc.page_content.split("\n\n\n")[0]
        content = ''.join(doc.page_content.split("\n\n\n")[1:])
        summary = sumandkey['summary']
        keywords = f"{sumandkey['keywords']}"
        itnews_data.append((source, date, title, content, summary, keywords))

    # 1. connect to database
    connection = sqlite3.connect("data/itnews.db", timeout=30)
    cursor = connection.cursor()

    # 2. create table
    cursor.execute("""
                CREATE TABLE IF NOT EXISTS itnews (
                    url TEXT PRIMARY KEY, -- URL은 유일한 값이므로 PRIMARY KEY로 설정
                    date TEXT,
                    title TEXT,
                    content TEXT,
                    summary TEXT,
                    keywords TEXT
                )
                """)

    # 3. insert data
    for data in itnews_data:
        cursor.executemany(
            "INSERT OR IGNORE INTO itnews (url, date, title, content, summary, keywords) VALUES (?, ?, ?, ?, ?, ?)", itnews_data)  # 중복되는 데이터는 무시

    # 4. commit
    connection.commit()

    # 6. close connection
    cursor.close()
    connection.close()

    now_news_df = pd.DataFrame(itnews_data, columns=[
        'url', 'date', 'title', 'content', 'summary', 'keywords'])

    # docs에 넣기
    docs = []
    for idx, row in now_news_df.iterrows():
        docs.append(Document(page_content=row.content, metadata={
                    "source": row.url, "title": row.title}))

    # text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # vetorstore
    try:
        vectorstore = FAISS.load_local(
            'data/faiss_itnews', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        print("Loaded existing vectorstore")
    except:
        vectorstore = FAISS.from_documents(documents=splits,
                                           docstore=InMemoryDocstore(),
                                           embedding=OpenAIEmbeddings())  # 벡터화 해서 저장
        print("Created new vectorstore")

    vectorstore.add_documents(splits)

    # vector store 저장
    vectorstore.save_local('data/faiss_itnews')
