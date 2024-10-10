#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : top1_sim1.py
@Author  : huanggj
@Time    : 2023/5/4 9:52
"""
#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : topk_sim.py

top1加上下句， 剩下的句子（对于每一句，跟当前选项相似度最高，跟当前3句相似度最低）
@Author  : huanggj
@Time    : 2023/5/2 19:15
"""
import numpy as np
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from collections import defaultdict


class Input_Preparation:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.option_embeddings = {}
        self.top_sentences = {}

    # 对所有句子进行编码
    def sentences_embeddings(self, sentences):
        embeddings = []
        for sentence in sentences:
            with torch.no_grad():
                inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to('cuda')  # Move the input tensors to the GPU
                outputs = self.model(**inputs)
                sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                embeddings.append(sentence_embedding)
        return embeddings

    def compute_option_embeddings(self, options):
        with torch.no_grad():
            for option in options:
                inputs = self.tokenizer(option, return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to('cuda')  # Move the input tensors to the GPU
                outputs = self.model(**inputs)
                option_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                self.option_embeddings[option] = option_embedding

    def cosine_similarity(self, emb1, emb2):
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        similarity_matrix = cosine_similarity(emb1, emb2)
        return similarity_matrix

    def select_top_1(self, option_embedding, all_sentences_embeddings):
        cos_similarities = cosine_similarity(option_embedding.reshape(1, -1), all_sentences_embeddings)
        top_1_index = cos_similarities.argsort()[0][-1]

        if top_1_index == 0:
            selected_indices = [0, 1, 2]
        elif top_1_index == len(all_sentences_embeddings) - 1:
            selected_indices = [top_1_index - 2, top_1_index - 1, top_1_index]
        else:
            selected_indices = [top_1_index - 1, top_1_index, top_1_index + 1]

        return selected_indices

    def assign_remaining_sentences(self, options, all_sentences, remaining_sentences_indices):
        assigned_remaining_sentences = defaultdict(list)
        sentence_embeddings = self.sentences_embeddings(all_sentences)

        for remaining_sentence_index in remaining_sentences_indices:
            sentence_embedding = sentence_embeddings[remaining_sentence_index]

            min_dissimilarity = float('inf')
            best_option = None

            for option in options:
                option_embedding = self.option_embeddings[option]

                # 计算与选项的相似度
                similarity_to_option = cosine_similarity([option_embedding], [sentence_embedding])[0][0]

                # 计算与分支内已有的topk句子的最小相似度
                min_similarity_to_topk = min(
                    cosine_similarity([sentence_embedding], [sentence_embeddings[top_sentence]])
                    for top_sentence in self.top_sentences[option]
                )[0][0]

                dissimilarity = min_similarity_to_topk - similarity_to_option

                if dissimilarity < min_dissimilarity:
                    min_dissimilarity = dissimilarity
                    best_option = option

            assigned_remaining_sentences[best_option].append(remaining_sentence_index)

        return assigned_remaining_sentences

    def prepare_inputs(self, options, all_sentences, question):
        inputs = {}
        self.compute_option_embeddings(options)
        all_sentences_embeddings = self.sentences_embeddings(all_sentences)

        # Step 1: Compute the top k sentences for each option
        for option in options:
            top_indices = self.select_top_1(option_embedding=self.option_embeddings[option],
                                            all_sentences_embeddings=all_sentences_embeddings)
            self.top_sentences[option] = sorted(top_indices)

        # Step 2: Remove used sentences from all_sentences
        used_sentences_indices = set()
        for option in options:
            used_sentences_indices.update(self.top_sentences[option])
        remaining_sentences_indices = [idx for idx in range(len(all_sentences)) if idx not in used_sentences_indices]

        # Step 3: Assign remaining sentences to each option
        assigned_remaining_sentences = self.assign_remaining_sentences(options, all_sentences,
                                                                       remaining_sentences_indices)

        input_list = []
        for option in options:
            all_sentence_indices = self.top_sentences[option] + assigned_remaining_sentences[option]

            # 对每个选项分支中的句子索引按照文章中的顺序进行排序
            all_sentence_indices.sort()
            text = None
            try:
                text = question + option + '<SEP>' + '。'.join([all_sentences[idx] for idx in all_sentence_indices])
            except:
                print(all_sentence_indices)
                print(all_sentences)
                #print(all_sentence_indices)
                #print("出错了！！ {}  !!  {}",format(",".join(all_sentence_indices), str(len(all_sentences))))
                #print(options)
            inputs[option] = text
            input_list.append(text)
        return input_list

def write_json(data, des_json_path):

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=3, sort_keys=False)


import json,os
from concurrent.futures import ProcessPoolExecutor


def process_batch(batch_data):
    # 加载模型和分词器
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to('cuda')
    tokenizer.add_tokens(['<选正确>', '<选错误>'])
    model.resize_token_embeddings(len(tokenizer))

    input_preparation = Input_Preparation(tokenizer, model)

    batch_prepared_inputs = []

    print("当前进程号 : {}, 处理数据条数: {}".format(os.getpid(), len(batch_data)))


    curr_data = 1
    for data in batch_data:
        context = data['fwe_context']
        cid = data['cid']
        options = data['qas'][0]['options']
        question = data['qas'][0]['question']
        question_type = 1
        if '不' in question or '误' in question:
            question_type = 0
        question = '<选正确>' if question_type == 1 else '<选错误>'
        # 切分句子
        sentences = context.split('。')
        print("当前进程号 : {}, 处理数据cid: {}".format(os.getpid(), cid))
        # 使用 Input_Preparation 类准备输入
        prepared_inputs = input_preparation.prepare_inputs(options, sentences, question)
        #print(prepared_inputs)
        data['qas'][0]['options_evidence'] = prepared_inputs
        batch_prepared_inputs.append(data)
        print("当前进程号 : {}, 处理数据进度: {}/{}".format(os.getpid(), curr_data, len(batch_data)))
        curr_data += 1
    return batch_prepared_inputs


def chunks(lst, n):
    """将列表 lst 分成大小为 n 的块。"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

set_ = 'test'

#source_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/new_data_0504/acrc_{}_rebalance.json'.format(set_)
source_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/data_aug_0612/acrc_{}.json'.format(set_)
des_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/data_aug_0612/acrc_top1_sim_{}.json'.format(set_)





# 加载数据集
with open(source_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 将数据集划分为小批量
max_workers=8

batch_size = (len(dataset['data']) // max_workers) + 1
batches = list(chunks(dataset['data'], batch_size))

# 使用多进程处理数据
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    prepared_inputs_list = list(executor.map(process_batch, batches))

# 将结果扁平化
prepared_inputs_list = [item for sublist in prepared_inputs_list for item in sublist]

if os.path.exists(des_path):
    os.remove(des_path)


write_json(prepared_inputs_list, des_path)
