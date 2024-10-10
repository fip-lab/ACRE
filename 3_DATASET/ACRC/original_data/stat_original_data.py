#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : chaifen.py
@Author  : huanggj
@Time    : 2024/3/1 21:13
"""
import os,json


# 写入json数据
def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    if os.path.exists(des_json_path):
        os.remove(des_json_path)

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)


def read_json(json_path):
    data_list = json.load(open(json_path, 'r', encoding='UTF-8'))['data']
    dict_ = dict()
    options_dict = dict()
    for d in data_list:
        dict_[int(d['cid'])] = d['context']
        options_dict[int(d['cid'])] = d['qas'][0]['options']
    return data_list,dict_,options_dict



#data_list, datas_map, options_dict = read_json("./new_context_data.json")

data_list = json.load(open("./ancient.json", 'r', encoding='UTF-8'))
print("aaa")



