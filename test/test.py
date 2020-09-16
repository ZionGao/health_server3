#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/8/30 5:33 下午
# @Author : 郜志强
# @Version：V 0.1
# @File : test.py
# @desc :
import requests
from flask import json
import pandas as pd

url = "http://47.115.62.243:9003/readFile"
file = open("data/user01.csv", 'rb')
files = {'file': file}
response = requests.post(url, files=files)
print(response.json())
file.close()

print('*'*100)

url = "http://47.115.62.243:9003/single"
# url = "http://0.0.0.0:5000/single"

input = pd.read_csv('data/health_evaluation.csv')
js = json.dumps(input.iloc[0,:].to_dict())
print(input.iloc[0,:].to_dict())
response = requests.post(url, data=js)
print(response.json())

print('*'*100)

url = "http://47.115.62.243:9003/batch"
# url = "http://0.0.0.0:5000/batch"
file = open("data/健康预测批量测试样本.csv", 'rb')
files = {'file': file}
response = requests.post(url, files=files)
file.close()
print(response.json())
