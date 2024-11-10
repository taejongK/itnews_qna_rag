from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.retrievers.multi_query import MultiQueryRetriever

import os

import streamlit as st
from PIL import Image

from dotenv import load_dotenv


def load_summary(selected_summary, api_key):
    # Summary prompt
    sumsum_prompt = PromptTemplate.from_template(
        """Summarize the information provided by the user into bullet points.
        The response should be concise, accurate, and presented in Korean.

        Content: {content}"""
    )

    # 문서 요약 추출 봇 모델
    sumsum_llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                            temperature=0, api_key=api_key)
    sumsum_chain = (
        {'content': RunnablePassthrough()}
        | sumsum_prompt
        | sumsum_llm
        | StrOutputParser()
    )

    sum_result = sumsum_chain.invoke('\n'.join(selected_summary))
    return sum_result


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


class ITNewsQnA:
    def __init__(self, api_key, faiss_path):
        # QnA prompt
        self.api_key = api_key
        self.faiss_path = faiss_path
        self.multiquery_retriever = self.vectorstore_retriever()
        self.qna_prompt = PromptTemplate.from_template(
            """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
        검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
        한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

        #Question:
        {question}

        #Context:
        {context}

        #Answer:"""
        )

    def vectorstore_retriever(self):
        # load vectorstore
        vectorstore = FAISS.load_local(
            self.faiss_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

        # for multiquery retriever
        mqr_llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                             temperature=0, api_key=self.api_key)

        # 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
        retriever = vectorstore.as_retriever(search_type="similarity", k=5)
        multiquery_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=mqr_llm
        )
        return multiquery_retriever

    def build_rag_chain(self):
        # qna_llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=self.api_key)
        qna_llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                             temperature=0, api_key=self.api_key)  # 실험용

        rag_chain = (
            {'context': self.multiquery_retriever | format_docs,
                'question': RunnablePassthrough()}
            | self.qna_prompt
            | qna_llm
            | StrOutputParser()
        )
        return rag_chain

    def get_answer(self, query, rag_chain):
        answer = rag_chain.invoke(query)

        retrieved_docs = self.multiquery_retriever.get_relevant_documents(
            query)

        title_list = []
        source_list = []
        for doc in retrieved_docs:
            title = doc.metadata['title']
            source = doc.metadata['source']
            title_list.append(title)
            source_list.append(source)
        return answer, title_list, source_list
