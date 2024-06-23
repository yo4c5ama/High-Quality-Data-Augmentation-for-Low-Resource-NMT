#!/usr/bin/env python3
# coding: utf-8
# @Time    : 2023/6/20
# @Author  : Liu_Hengjie
# @Software: PyCharm
# @File    : retrieval.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import random

# ex = 'top6'
# val = 'top6'
de = '../data/transformer_baseline/train.de_hsb.de'
hsb = '../data/transformer_baseline/train.de_hsb.hsb'
val_de = '../data/transformer_baseline/val.de_hsb.de'
val_hsb = '../data/transformer_baseline/val.de_hsb.hsb'
test_de = '../data/transformer_baseline/test.de_hsb.de'
test_hsb = '../data/transformer_baseline/test.de_hsb.hsb'
mono_de = "../data/original_data/de_hsb/news.2007.de.shuffled.deduped"
mono_de_domain = '../data/transformer_baseline/mono.de_hsb.de'

train_src = f'./data/de_hsb/L_.9/train.dehsb_hsb.dehsb'
train_tgt = f'./data/de_hsb/L_.9/train.dehsb_hsb.hsb'
val_src = f'./data/de_hsb/L_.9/val.dehsb_hsb.dehsb'
val_tgt = f'./data/de_hsb/L_.9/val.dehsb_hsb.hsb'
test_src = f'./data/de_hsb/L_.9/test.dehsb_hsb.dehsb'
test_tgt = f'./data/de_hsb/L_.9/test.dehsb_hsb.hsb'
sim_path = f'./data/de_hsb/L_.9/sim.txt'
aug_src = f'../data/transformer_L2_.8/aug.dehsb_hsb.dehsb'


sep = " | "
S = " S： "
St = " St： "
Tt = " Tt： "
new = []



def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text.split('\n')



def get_k_similar(k, a_lines, b_lines):
    sent_pair = []
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    embeddings_a = model.encode(a_lines)
    embeddings_b = model.encode(b_lines)
    faissIndex = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings_a.shape[1]))
    # norm_embeddings_a = faiss.normalize_L2(embeddings_a)
    # norm_embeddings_b = faiss.normalize_L2(embeddings_b)
    faissIndex.add_with_ids(embeddings_a, np.array(range(0, len(a_lines))))
    dot_product, indexes = faissIndex.search(embeddings_b, k)
    return dot_product, indexes


def compair_sentence(a_lines, b_lines, c_lines, d_lines, dot_product, indexes):
    src = []
    tgt = []
    sim = []
    for ind, val in enumerate(indexes):
        if ind not in val:
            val = val[:-1]
        n = 0
        for k, s_idx in enumerate(val):
            if s_idx != ind:
                n += 1
                if n > 8:
                    break
                if n == 1 and dot_product[ind][k] > 0.8:
                    src.append(S + b_lines[ind]+ St + a_lines[s_idx] + Tt + c_lines[s_idx] + '\n')
                    tgt.append(d_lines[ind] + '\n')
                    sim.append(str(dot_product[ind][k]) + '\n')
                if dot_product[ind][k] <= 0.8:
                    src.append(S + b_lines[ind]+ St + a_lines[s_idx] + Tt + c_lines[s_idx] + '\n')
                    tgt.append(d_lines[ind] + '\n')
                    sim.append(str(dot_product[ind][k]) + '\n')
                else:
                    break
    write_file(sim_path, sim)
    return src, tgt


def compair_aug_sentence(a_lines, b_lines, c_lines, indexes):
    src = []

    for ind, val in enumerate(indexes):
        if ind not in val:
            val = val[:-1]
        for s_idx in val:
            if s_idx != ind:
                src.append(S + b_lines[ind] + St + a_lines[s_idx] + Tt + c_lines[s_idx] + '\n')
    return src


def create_same_domain_sentence(a_lines, indexes):
    domain_sentences = []

    for ind, val in enumerate(indexes):
        if ind not in val:
            val = val[:-1]
        for s_idx in val:
            if s_idx != ind:
                domain_sentences.append(a_lines[s_idx] + '\n')

    return domain_sentences


def write_file(path, sentences):
    with open(path, 'w', encoding="utf-8") as f:
        for i in sentences:
            f.write(i)


def generate_new(top_k_similar, retrieve_dataset, query_lines, retrieve_hsb_tran_dataset, query_hsb_tran_dataset, new_src_path, new_tgt_path):
    dot_product, indexes = get_k_similar(top_k_similar, retrieve_dataset, query_lines)
    src, tgt = compair_sentence(retrieve_dataset, query_lines, retrieve_hsb_tran_dataset, query_hsb_tran_dataset, dot_product, indexes)
    bi_list = list(zip(src, tgt))
    random.shuffle(bi_list)
    new_src, new_tgt = list(zip(*bi_list))
    print(int(len(new_src)))
    print(int(len(new_tgt)))
    write_file(new_src_path,  new_src)
    write_file(new_tgt_path, new_tgt)

def generate_aug(top_k_similar, retrieve_dataset, query_lines, retrieve_hsb_tran_dataset, new_src_path):
    dot_product, indexes = get_k_similar(top_k_similar, retrieve_dataset, query_lines)
    src = compair_aug_sentence(retrieve_dataset, query_lines, retrieve_hsb_tran_dataset, indexes)
    # new_src = src[0]
    write_file(new_src_path, src)


def same_domain(top_k_similar, retrieve_dataset, query_lines, new_src_path):
    dot_product, indexes = get_k_similar(top_k_similar, retrieve_dataset, query_lines)
    domain_sentences = create_same_domain_sentence(retrieve_dataset, indexes)
    write_file(new_src_path, domain_sentences)

if __name__ == '__main__':
    de_lines = read_data(de)
    hsb_lines = read_data(hsb)
    val_de_lines = read_data(val_de)
    val_hsb_lines = read_data(val_hsb)
    test_de_lines = read_data(test_de)
    test_hsb_lines = read_data(test_hsb)
    aug_de_lines = read_data(mono_de_domain)
    mono_de_lines = read_data(mono_de)[:-1]

    # same_domain(2, mono_de_lines, de_lines, mono_de_domain)
    generate_aug(2, de_lines, aug_de_lines, hsb_lines, aug_src)
    # generate_new(20, de_lines, de_lines, hsb_lines, hsb_lines, train_src, train_tgt)
    # generate_new(20, de_lines, val_de_lines, hsb_lines, val_hsb_lines, val_src, val_tgt)
    # generate_new(2, de_lines, test_de_lines, hsb_lines, test_hsb_lines,  test_src, test_tgt)


