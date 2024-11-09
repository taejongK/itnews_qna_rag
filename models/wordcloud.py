from wordcloud import WordCloud
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def get_news_df(path):
    # db에서 데이터 불러오기
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    # 최근 3일 이내 데이터만 불러오기
    query = '''SELECT * FROM itnews WHERE date >= DATE('now', '-3 day')'''
    df = pd.read_sql_query(query, connection)
    df['keywords'] = df.keywords.apply(lambda x: eval(x))

    cursor.close()
    connection.close()
    return df


def get_wordcloud(df):
    all_keywords = []  # 최근 3일간의 모든 키워드를 저장할 리스트
    for keyword in df.keywords:
        # print(keyword)
        all_keywords.extend(keyword)

    # 리스트를 공백으로 연결해 문자열로 변환
    text = ' '.join(all_keywords)

    # 워드 클라우드 생성
    wordcloud = WordCloud(font_path='/System/Library/Fonts/AppleGothic.ttf',
                          width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt
