#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib # 시각화')


# In[2]:


get_ipython().system('pip install flask # 플라스크 서버')


# In[3]:


get_ipython().system('pip install opencv-python # 영상관련 처리')


# In[ ]:


pip install -U flask-cors # cross domain issue 비동기 통신 시 차단을 허가


# In[4]:


from flask import Flask, request # Flask 서버 구축
import cv2 # 영상처리용 opencv
import io # 파일 io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from flask_cors import CORS 


# In[7]:


app = Flask(__name__) # 플라스크 서버 객체 생성
# 사용자 요청을 처리하기 위한 라우터 설정
CORS(app)

# 모델 로딩
model = load_model("my_fashion_mode.h5") 

@app.route('/fileUpload', methods = ['GET','POST']) # 허용하는 url 경로 및 요청방식
# 해당 서버 접속 시 호출되는 메소드
def fileUpload():
    if request.method == "POST":
        print("print")
        f = request.files["image"] # 요청객체 name =  ""
        bytes_file = io.BytesIO() # 전송을 위해 byte단위로 저장하는 객체 생성
        f.save(bytes_file) # f를 byte 로 변형 및 저장
        data = np.fromstring(bytes_file.getvalue(), dtype = np.uint8) # byte 타입 >> 핸들링을 위해 np 타입으로 변경 / unit8 : 8비트 양수화 0~255
        # 색상데이터는 255개뿐이니 크기를 딱 맞게 할당한 것
        print(data.shape)
        
        #################################커스텀##############################################
        
        img = cv2.imdecode(data, 1) # 1 : 컬러사진 / 0 : 흑백
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # openCV 색상 체계를 변환 BGR >> RGB
        display(plt.imshow(img_rgb)) # matplotlib 을 이용하여 사진을 그린다.
        plt.show() # 그림을 보여준다
        
        # 학습 할 때 사용한 전처리 방법을 사용자의 업로드 파일에 그대로 적용해야 한다 *****
        # 티쳐블머신에서 진행한 스케일링에 발맞춰 스케일링
        img_scaled = (np.array(img_rgb, dtype = np.float32) / 127.0) - 1
        
        
        # 모델 예측
        pre = model.predict(img_scaled.reshape(1,224,224,3)) # 스케일링 된 사진데이터 shapq(224,224,3)   3 : RGB /  1 : 1장만~
        # 딥러닝은 동시에 여러 샘플을 작업하기 때문에 샘플 수를 입력해야 한다.
        i = np.argmax(pre) # 3개의 확률 정보 중 최대값의 인덱스 번호를 리턴
        
        if i == 0:
            result = "마이크"
        elif i  == 1:
            result = "과자"
        elif i == 2:
            result = "리모컨"
        
        
    return "당신이 업로드한 사진은 {}입니다".format(result)

app.run(host="222.102.104.28", port=7000) # 서버 구동 시작! 

# 주의사항 : * 서버가 열려있는 지 체크


# In[ ]:





# In[ ]:




