#!/usr/bin/env python
# coding: utf-8

# In[36]:


#pip install chromedriver_autoinstaller


# In[37]:


#pip install pyperclip


# In[39]:


#pip install selenium


# In[1]:


#pip install beautifulsoup4


# In[2]:


#pip install js2py


# In[5]:


import chromedriver_autoinstaller
import time
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import urllib.request
from selenium import webdriver as wb
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pyperclip


# In[24]:


driver = webdriver.Chrome("chromedriver.exe")
driver.get("https://cafe.naver.com/ArticleList.nhn?search.clubid=21820768&search.menuid=218&search.boardtype=I")

driver.find_element_by_class_name('gnb_txt').click()

time.sleep(1)

# id, pw 입력할 곳을 찾습니다.
tag_id = driver.find_element_by_name('id')
tag_pw = driver.find_element_by_name('pw')
tag_id.clear()

# id 입력
time.sleep(0.5)
tag_id.click()
pyperclip.copy('agnusdei1207')
tag_id.send_keys(Keys.CONTROL, 'v')


# pw 입력
time.sleep(0.5)
tag_pw.click()
pyperclip.copy('Tmfqls12@')
tag_pw.send_keys(Keys.CONTROL, 'v')


# 로그인 버튼을 클릭합니다
driver.find_element_by_id('log.login').click()
time.sleep(0.5)

#iframe 접속 전에 자바스크립트로 호출 함수 선언 후 호출 -> iframe 불러오기 
print(driver.execute_script('return navigator.userAgent'))


#iframe 접속
driver.switch_to.frame('cafe_main')
driver.find_element_by_css_selector('span.ellipsis').click()

print(soup.text)
time.sleep(1)
#휠 내리기
body = driver.find_element_by_css_selector("body")
for i in range(7):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    
#그림
#그림클릭
time.sleep(1)
driver.switch_to.frame('cafe_main')

soup = driver.find_element_by_xpath('/html/body/div/div/div/div[2]/div[2]/div[1]/div[2]/div[1]/div/div/div[3]/div/div/div/a/img')
soup.click()

#soup = soup.find("img",class_ ="se-image-resource")




# In[ ]:




