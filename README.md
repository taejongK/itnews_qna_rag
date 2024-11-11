# 뉴스기사 클롤링 RAG 프로젝트

## Abstract

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image.png)

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%201.png)

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%202.png)

# Introduction

### 전처리 과정

1. WebBaseLoad를 사용해 IT news 기사 수집 (Load)
2. GPT-3.5-turbo를 사용해 수집된 기사에서 Summary, Keywords를 추출하여 URL, Title, Content, Summary, Keywords 값을 SQLite에 저장
3. RecursiveCharacterTextSplitter를 사용해 텍스트의 의미가 손상되지 않도록 분할
4. 수집된 기사를 OpenAI 모델로 Embedding해 VectorDB(Faiss)에 저장

### 서비스 과정

1. SQLite에서 최근 3일의 Keywords를 추출하여 word cloud 생성
2. 사용자가 입력한 keyword를 데이터베이스에서 검색하여 관련된 summary를 제공
3. 사용자가 세부적인 질문을 하면 Ensemble Retriever로 질문과 관련 내용, 키워드가 포함된 문서를 검색하여 prompt에 넘기고 LLM이 답변 생성

# Methods

## 뉴스 기사 요약 및 키워드 추출

### GPT model 비교_비용과 시간 측면에서

- 뉴스기사를 요약하고 키워드를 추출함에 있어서 과도하게 좋은 모델이 필요한 것인가에 대한 검증이 필요
- 퀄리티를 비교하기 전에 우선 비용과 소요시간 측면에서 비교
- gpt-4와 gpt-3.5-turbo 모델 비교

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%203.png)

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%204.png)

- 어떤 모델을 사용하던지 input token의 수는 같지만 output token의 수는 3배 가까이 차이

https://openai.com/api/pricing/

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%205.png)

openai의 홈페이지에 나온 가격 표에 따르면 gpt-4모델은 gpt-3.5-turbo 모델에 비해 input에서는 약 60배, ouput에서는 약 108배 비쌈

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%206.png)

### 품질 평가

[naver_news_summary_evaluation.csv](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/naver_news_summary_evaluation.csv)

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%207.png)

- 구글 스프래드시트를 사용하여 QA를 진행
- gpt-3.5-turbo모델을 사용하여 원본과 각 모델로 요약한 결과를 입력하여 요약 결과를 평가하고 10점 만점의 점수로 출력하는 프롬프트를 사용

```jsx
=GPT("Compare the original text with the summaries and evaluate how well the summaries are written on a scale of 1 to 10, outputting only the scores and no other text." & $L2)
```

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%208.png)

- ㅌ두 모델의 점수 차는 약 1.3 점 정도, 요금차이와 속도 차이를 생각했을 때, gpt-3.5-turbo를 사용하는 것이 합리적이라고 판단했습니다.

### WordsCloud 결과

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%209.png)

## VectorDB(Faiss)

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%2010.png)

출처: [https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/)

- 수집된 기사를 RecursiveCharacterTextSplitter를 사용해 내용을 작은 단위로 분할
- RecursiveCharacterTextSplitter는 뉴스 기사의 내용을 **단락->문장->단어** 순서로 재귀적으로 분할하며 지정한 chunk 크기에 들어오면 분할을 중지
- 온전한 의미단위로 분할되어 가장 많이 사용되는 splitter
- 분할된 context들을 vector embedding해 vectorDB(Faiss)에 저장

## Ensemble Retriever

### 문제점

- 질문에 대한 답변 속도가 너무 느림
- 잘못된 질문을 하면 키워드도 전혀 상관없는 답변을 출력
  
    ![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%2011.png)
    
    ```jsx
    query = "LG유플러스가 익시오랑 뭘 했나?"
    
    네이버는 상반기에 '네이버플러스 스토어'를 별도 애플리케이션으로 출시할 계획이며, 
    이 서비스는 AI 기반 맞춤 쇼핑 추천 기능을 고도화한 것입니다. 
    또한, 네이버는 생성형 인공지능(AI) 검색 서비스의 모바일 버전을 내년에 출시할 예정입니다. 
    이 서비스는 최신 데이터를 기반으로 이용자 의도와 맥락을 더 잘 이해하고 
    검색에 대한 직접적인 답을 쉽게 요약하는 AI 브리핑 기능을 제공할 것입니다. 이외에도, 네이버는 이미지와 음성을 이해하고 처리하는 멀티모달 기술도 제공할 계획입니다.
    ['\n또 최대실적 쓴 네이버…“AI·커머스 경쟁력 키울 것”', '\n네이버 최수연 "생성AI 검색기능 모바일 버전 내년 출시"', '\nKT, 데이터브릭스와 국내 AX 전환 협력…데이터·AI 플랫폼 개발', '\n또 최대실적 쓴 네이버…“AI·커머스 경쟁력 키울 것”', '\n네이버-인텔 공동개발 ‘가우디’, 연말 외부 공개…AI 원가 절감 대안', '\n[일문일답] LGU+, 구글과 \'홈 에이전트\' 개발…"AI 투자, 2028년까지 3조"']
    ```
    
    - 유플러스와 익시오 관련 질문을 했는데 전혀 상관없는 네이버플러스 이야기를 함

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%2012.png)

### MultiQuery Retriever

LLM 모델을 사용해 사용자의 질문을 여러 질문으로 나누어 생성

Q: "LG유플러스가 익시오랑 뭘 했나?”

- 1. 어떤 협력사와 LG유플러스가 협력했나요?
- 2. LG유플러스와 익시오가 어떤 협업을 진행했나요?
- 3. LG유플러스와 익시오 간의 협력 사항은 무엇인가요?

- MultiQuery Retriever는 Dense Retriever로 질문과 검색할 내용을 모두 vector화 하여 유사도를 기반으로 검색. 때문에 맥락을 고려한 검색이 가능
- 하지만 명확한 키워드를 포함하지 않은 유사한 문서도 주요 내용으로 인식할 수 있다는 문제가 있음

### BM 25 Retriever

https://blog.naver.com/shino1025/222240603533

TF-IDF기반의 검색 모델인 BM25의 특징을 사용자가 검색한 주요 키워드를 최대한 포함하도록 한다는 점이다. 

![image.png](%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3%E1%84%80%E1%85%B5%E1%84%89%E1%85%A1%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A9%E1%86%AF%E1%84%85%E1%85%B5%E1%86%BC%20RAG%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20132b81594c638073aa36f1cbc5604736/image%2013.png)

- 간단히 설명하자면 저장된 문서들의 주요 키워드 랭킹을 매기고 사용자의 질문에서도 핵심 키워드를 추출하여 문서를 검색

# Conclusion

- RAG 서비스를 처음 만들어 보면서 서비스 전반에 걸쳐서 비용과 처리 시간, 프로젝트 목표에 맞춰서 고려할 요소들을 이해
- LLM을 사용해 LLM의 성능을 평가하는 방법을 적용하여 평가를 자동화 하는 방법을 도입
- Ensemble Retriever를 사용해 embedding을 사용한 검색 방법의 단점을 보완하여 답변 품질 상승
- 이번 프로젝트에서는 기사를 저장할 때 image 데이터는 크게 고려하지 못함. 하지만 image 역시 내용을 이해하는데 중요한 정보이기 때문에 기사 분할 할 때, 이미지를 인식하고 포함할 수 있는 방법 개발 필요