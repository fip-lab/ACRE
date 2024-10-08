#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : random_utils.py
@Author  : huanggj
@Time    : 2023/6/16 13:48
"""
import random
def generate_random_str(randomlength=16):
  """
  生成一个指定长度的随机字符串
  """
  random_str =''
  base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
  length =len(base_str) -1
  for i in range(randomlength):
    random_str +=base_str[random.randint(0, length)]
  return random_str
