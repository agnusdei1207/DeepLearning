# -*- coding: utf-8 -*-

- 텍스트 : 길이나 문자의 종류가 다른 비정형 데이터
- 텍스트 마이닝 : 텍스트를 분석해서 텍스트에 포함된 의미를 찾아내는 작업
    - 응용분야 
        1. 감성분석: 텍스트에서 감성을 분석(긍정, 부정, 기쁨, 슬픔)
        2. Topic 모델링: 텍스트에서 대표적인 단어를 추출하는 작업
        3. 요약: 긴 문장을 짧은 문장으로 압축
        4. 문장생성: 짧은 문장을 이용해서 긴 문장으로 만드는 작업
        5. 번역: 다른 언어들끼리 변환하는 작업

- 한글을 이용한 텍스트 마이닝의 문제점
    - 한글은 띄쓰기가 잘 지켜지지 않는다는 점
        1. 형태소 분리
        2. 형용사의 표현이 다양함 (의성,의태어, 같은 단어이나 다른 뜻, 다른 단어인데 같은 뜻

- 텍스트 마이닝 과정
    1. 오류제거(Cleaning): 잘못된 문자 변경
    2. 토큰화: 문장을 단어, 문자, 자소로 분리하는 작업
    3. 텍스트 전처리
        - 불용어 제거(Stopword): 학습에 필요없는 단어 제거하는 작업
        - 어간, 표제어 처리 (대표어)
    4. 라벨 인코딩: 문자를 숫자로 변경
        - 빈도수 분석: 단어들이 문장에서 등장한 횟수 분석
        - 빈도순 정렬
        - 빈도순 인덱스 번호 부여 및 할당
    5. 문장을 같은 길이로 만들어 주는 작업
    6. 학습

### 환경구축
"""

# 자연어 처리 라이브러리 설치
!pip install nltk # 영어 기준

import nltk

nltk.download()

#punkt(토큰화), stopword(불용어처리), wordnet(표제어 추출)

!pip install tweepy==3.10.0

# 한국어 자연어 처리 라이브러리 설치
!pip install konlpy

"""https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype 버젼이 일치하는 버전 다운로드 Jpype (Help-about을 통해 버전 확인)

JPype1‑1.3.0‑cp38‑cp38‑win_amd64.whl
"""

!pip install C:\JPype1-1.3.0-cp38-cp38-win_amd64.whl

# jpype 라이브러리 설치
!pip install jpype1==1.3.0

# konlpy 삭제 (아나콘다 프롬프트 가서 삭제)
# 다시 설치하기

!pip install konlpy

from konlpy.tag import Kkma

"""java 패스 에러 (Illegal char <*> C:\Users\smhrd\anaconda3\Lib\site-packages\konlpy\java)가 나는 경우

C:\Users\smhrd\anaconda3\Lib\site-packages\konlpy\jvm.py 파일을 열어서 *을 제거한다

###  토큰화

- 단어에서 문자, 단어, 형태소, 자소를 분리하는 작업
"""

import nltk
from nltk.tokenize import word_tokenize
# 문장
corpus = "나는 상우입니다. 나이는 31세입니다. 지금은 머신러닝 수업 중입니다."

print(word_tokenize(corpus))

from konlpy.tag import Kkma, Okt

kkma = Kkma()
okt = Okt()

#명사 분리: 일반적으로 잘 분류하지 않는 방법
print(kkma.nouns(corpus))
print(okt.nouns(corpus))

#형태소 분리
print(kkma.morphs(corpus)) #꼬꼬마는 속도가 느리지만 정확도가 높다
print(okt.morphs(corpus))

#품사 확인
import pandas as pd
kkma_pos = kkma.pos(corpus)
okt_pos = okt.pos(corpus)

print(kkma.pos(corpus))
print(okt.pos(corpus))

kkma_pos = pd.DataFrame(kkma_pos)
okt_pos = pd.DataFrame(okt_pos)
print(kkma_pos.shape)
print(okt_pos.shape)

#명사만 추출해보자
okt_pos[okt_pos.iloc[:,1]=="Noun"]

kkma_pos[kkma_pos.iloc[:,1]=="NNG"]

"""### 정규화 / 정제 (Cleaning) 오타수정
- 문장에서 필요한 텍스트만 추출하는 작업
"""

import re
#문장에서 한글만 추출 - 정규식을 활용
corpus = "안녕하세요^^ hahahaha 김정식입니다. 나이는 30세입니다!! ~!! ^^"
# ^ : not (한글과 공백이 아닌 문자)
word = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]") #공백있음!!! 띄어쓰기도 추출하려는 의도

# 함수 안에 입력된 문자아닌 문자제거
#word = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9a-zA-Z]")

#필요없는 문자 제거
result = word.sub("",corpus)

token = okt.morphs(result)
token

"""### 불용어 처리 Stop word
- 학습에 사용되지 않는 단어나 문자를 삭제하는 작업
"""

# 제거할 단어 리스트를 저장
stop_words = ["입니다", "는", "세입", "니", "다"]

result2 = []

#분리된 형태소를 하나씩 가져온다
#영어는 공백을 기준으로 구분하지만 한글은 형태소로 구분하는 게 더 효율적이다.
for w in token:
    if w not in stop_words:
        result2.append(w)
result2

"""## 텍스트 마이닝에서는 특성이 곧 글자이기에 글자수가 같아야 한다

##  Encoding : 학습을 위한 데이터로 바꾸어주자!
- 단어들을 숫자로 변경하는 작업
    - 라벨인코딩: 단어들을 정수로 변경
    - 원핫인코딩: 단어들을 2진수로 변경
        - 텍스트마이닝에서는 단어들이 많기 때문에 비트수가 커져서 과대적합 발생확률이 높아진다
- word embedding: 단어들을 거리와 방향값으로 변환(실수값)
    - 3차원의 공간에 단어들을 뿌려준다
    - 스칼라를 벡터화 한 후 거리와 방향을 설정한다
    - 실수값
- word embedding 방법:
    - 카운트 기반 방법: LSA 등 문장생성
    - 예측 기반 방법: Word2Vec, Word2Doc, FastTest 임베딩 된 모델
    - 카운트와 예측 기반 동시에 하는 방법: Golve 등
- BoW(Bag of words): 단어사전
    - 단어의 순서는 고려하지 않고 빈도만 고려하고 수치화 한 것
    - 순서
        - 빈도수 분석
        - 내림차순 정렬
        - 인덱스 부여
        - 단어들을 해당 인덱스로 변환

# Bow 만들기

- Tokenizer 함수 이용법
"""

from sklearn.feature_extraction.text import CountVectorizer
# 리스트
corpus = ["이수환은 남자입니다. 하지만 남자는 역시 주먹입니다.\
          하지만 이수환은 가위를 좋아합니다"]

vec = CountVectorizer()
#문자에서 단어들의 빈도수를 분석한 후에 가나다순으로 정렬하여 표시
result = vec.fit_transform(corpus).toarray()

result

#분리된 단어를 표시
print(vec.get_feature_names())

#단어와 인덱스 출력 (가나다 순)
print(vec.vocabulary_)

# 불용어 처리 후 다시 넣어보자!
okt = Okt()
result = okt.morphs(corpus[0])
print(result)

# 불용어 처리

result2 = []

stop_word = ["은","입니다", "를", "하지만", "는","."]

for m in result:
    if m not in stop_word:
        result2.append(m)
result2

# 분리된 단어를 문장으로 다시 변환
corpus2 = ""

for m in result2:
    corpus2 = corpus2+" "+m
    
corpus2

result3 = vec.fit_transform([corpus2]).toarray()
result3

print(vec.get_feature_names())
print(vec.vocabulary_)

"""- CountVectorizer 함수 이용법

- TfidVectorizer 함수 이용법

- TF_IDF: 단어의 중요도는 빈도수에 비례, 등장하는 문서의 수에 반비례
-TF (Text Frequency): 해당 단어가 몇 번 등장했는지? 빈도수
-DF (Document Frequency): 해당 단어가 등장하는 문서 혹은 문장의 수
- IDF(Inverse DF): 단어의 중요도는 등장하는 문장 혹은 문서 수에 반비례하기 때문에 주로 사용한다.

![KakaoTalk_20211228_110019481.png](attachment:KakaoTalk_20211228_110019481.png)
"""

corpus = ["수환은 남자입니다",
         "남자는 주먹입니다.",
        "하지만 수환 남자는 가위를 좋아합니다"]

# 빈도수가 높으면 중요도가 올라가는 인코딩
c_result = vec.fit_transform(corpus).toarray()

print(c_result)
print(vec.vocabulary_)

# TF-IDF를 이용한 인코딩
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

result4 = tfidf.fit_transform(corpus).toarray()

print(result4) 

#중요도를 분석해서 올린다.(위의 공식 적용)

#크롤링
X_train = ["제품을 잘 쓰고 있어요",
              "제품의 디자인이 우수해요",
              "성능이 우수해요",
              "제품에 손상이 있어요",
              "제품 가격이 비싸요",
             " 정말 필요한 제품입니다"]
#라벨링
y_train = [1,1,1,0,0,1]

#데이터 분석
vec.fit(X_train)
#분석된 결과를 반영

#수치화 
result = vec.fit_transform(X_train).toarray()

print(result)

#모델 선택 및 학습
from sklearn.linear_model import LogisticRegression as lr

lr = lr(C=1)

lr.fit(result,y_train)

lr.score(result,y_train)

X_new = ["제품이 너무 손상이 있어요"]

X_new_result = vec.transform(X_new).toarray()

X_new

#pred = lr.predict(result4)
#pred.score()

"""### 연습! Term Frequency - Inverse Document Frequency"""

X_train = ["제품을 잘 쓰고 있어요",
                   "제품의 디자인이 우수해요",
                   "성능이 우수해요",
                   "제품에 손상이 있어요",
                   "제품 가격이 비싸요",
                   "정말 필요한 제품입니다",
                   "가격이 비싸고 서비스도 엉망이예요",
                   "가격대비 성능이 우수해요",
                   "디자인이 좋아요",
                   "필요없는 기능이 많은듯 해요",
                   "서비스 대응이 나빠요",
                   "제품에 손상이 심해요"]
y_train = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0]

# TF-IDF를 이용한 인코딩
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

result = tfidf.fit_transform(corpus).toarray()

print(result) 

#중요도를 분석해서 올린다.(위의 공식 적용)

X_new = ["제품이 나빠요"]

X_new_result = vec.transform(X_new).toarray()

X_new

#pre = lr.predict(X_new_result)
#pre.score()

"""### IMDB 데이터셋을 이용한 감성분석 (Embedding)
- 인코딩이 되어 있는 데이터셋
- 댓글 데이터 셋
- 부정0, 긍정1 라벨링
"""

import pandas as pd

imdb = pd.read_csv("./imdb_master.csv")
# unsup 제거
imdb = imdb.drop(imdb[imdb["label"]=="unsup"].index) # .index *****
imdb.head()

"""### 전처리 """

from sklearn.model_selection import train_test_split

X = imdb.loc[:,"review"]
y = imdb.loc[:,"label"]

X = X.replace("[^a-zA-z ]","")
y = y.replace("[^a-zA-z ]","")

X = X.astype(str)
y = y.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=3, test_size=0.3)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

"""### 인코딩"""

from sklearn.feature_extraction.text import CountVectorizer

# max_features=10000 : 단어수를 10000개만 인코딩
cv = CountVectorizer(max_features=1000)

cv.fit(X_train)

X_train_en = cv.transform(X_train).toarray()
X_test_en = cv.transform(X_test).toarray()

X_train_en[0], cv.vocabulary_

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()

model_lr.fit(X_train_en,y_train)
X_train_en.shape

print("훈련 점수 : ", model_lr.score(X_train_en, y_train))
print("테스트 점수 : ", model_lr.score(X_test_en, y_test))

pred = model_lr.predict(X_test_en)

for i in range(10):
    print(pred[i]," ===> ", y_test.iloc[i])

from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense


seed=0 # 
np.random.seed(seed) # 
tf.random.set_seed(100) #

model = Sequential() 
model.add(Dense(1, activation='sigmoid', input_dim=1))  
model.add(Dense(1, activation='sigmoid'))
model.summary() 

data = X_train_en
labels = np.array(y_train)

model.compile(optimizer='adam',  #컴파일 방식
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(data, labels, epochs=10000, batch_size=(1000,1000))

print("\n Accuracy: %.4f"%(model.evaluate(X_test_en,y_test_en)[1]))



