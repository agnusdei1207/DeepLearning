#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install selenium


# In[2]:


from selenium import webdriver as wb
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bs
from pandas import pd


# In[3]:


#chromedirver 설치
#선생님 제자리는 chromedriver 설치가 안 돼서 시험이니까 일단 설치했다고 하고 작성할게요!


# In[4]:


#1번 문제
driver = wb.Chrome()
driver.get("https://www.naver.com/")

input_area = driver.find_element_by_css_selector("input#query")
input_area.send_keys("크롤링")

input_area.send_keys("Keys.ENTER")           


# In[5]:


#2번 문제
driver = wb.Chrome()
driver.get("https://www.gmarket.co.kr/Bestsellers?viewType=G&groupCode=G01")

time.sleep(1)

button = driver.find_element_by_css_selector("span.text_rank")

time.sleep(0.5)
try:
    button.click()
except:
    print("end")

body = driver.find_element_by_css_selector("body")
for i in range(10):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)

soup = bs(driver.page_source,"lxml")

item_name = driver.find_elements_by_css_selector("a.itemname")
itme_price = driver.find_elements_by_css_selector("div.s-price>strong>span>span")

list1 = []
list2 = []

for i in range(20):
    list1.append(item_name[i].text)
    list2.append(tiem_price[i].text)

dic = {"상품이름":list1, "가격":list2}
df = pd.DataFrame(dic)
df


# In[6]:


#3번 문제
driver = wb.Chrome()
driver.get("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36")

button = driver.find_element_by_css_selector("span.menu_bg.menu01")
time.sleep(1)
try:
    button.click()
except:
    print("end")

body = driver.find_element_by_css_selector("body")
for i in range(10):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)

soup = bs(driver.page_source,"lxml")

song_list = []
singer_list = []

song_list = driver.find_elements_by_css_selector("div.ellipsis.rank01")
singer_list = driver.find_element_by_css_selector("div.ellipsis.rank02>a")

list1 = []
list2 = []

for i in range(100):
    list1.append(song_list[i].text)
    list2.append(singer_list[i].text)

dic = {"곡명":list1, "가수":list2}
df = pd.DataFrame(dic)
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




