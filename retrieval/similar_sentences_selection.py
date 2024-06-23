#!/usr/bin/env python3
# coding: utf-8
# @Time    : 2023/6/20
# @Author  : Liu_Hengjie
# @Software: PyCharm
# @File    : most_sim.py
#!/usr/bin/env python3
# coding: utf-8
# @Time    : 2023/6/20
# @Author  : Liu_Hengjie
# @Software: PyCharm
# @File    : most_sim.py
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import faiss
import random

# ex = 'top6'
# val = 'top6'
de = '../data/original_data/de_hsb/train.hsb-de.de'
hsb = '../data/original_data/de_hsb/train.hsb-de.hsb'
val_de = '../data/original_data/de_hsb/devel.hsb-de.de'
val_hsb = '../data/original_data/de_hsb/devel.hsb-de.hsb'
test_de = '../data/original_data/de_hsb/devel_test.hsb-de.de'
test_hsb = '../data/original_data/de_hsb/devel_test.hsb-de.hsb'
created_de = '../data/GAN_TM/created.de_hsb.de'
created_hsb = '../data/GAN_TM/created.de_hsb.hsb'
# mono_de = "../data/monolingual/de/mono_de_500000.de"
# mono_de_domain = '../data/perplexity/orginal_bilingual_corpus/'

train_src = f'../data/transformer_L2_.5/train.dehsb_hsb.dehsb'
train_tgt = f'../data/transformer_L2_.5/train.dehsb_hsb.hsb'
val_src = f'../data/transformer_L2_.5/val.dehsb_hsb.dehsb'
val_tgt = f'../data/transformer_L2_.5/val.dehsb_hsb.hsb'
test_src = f'../data/transformer_L2_.5/test.dehsb_hsb.dehsb'
test_tgt = f'../data/transformer_L2_.5/test.dehsb_hsb.hsb'
created_src = '../data/GAN_TM/created.de_hsb.de'
created_tgt = '../data/GAN_TM/created.de_hsb.hsb'
sim_path = f'../data/transformer_L2_.5/sim.txt'
aug_src = f'../data/faiss/mono/src_aug_21000.3.dehsb'


de_sep = " [DE] "
hsb_sep = " [HSB] "
sep = " | "
S = " [SEP] "
St = " [SEP] "
Tt = " [SEP] "
new = []



def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text.split('\n')



def get_k_similar(k, a_lines, b_lines):
    sent_pair = []
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2', device='cpu')
    embeddings_a = model.encode(a_lines)
    embeddings_b = model.encode(b_lines)
    faissIndex = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings_a.shape[1]))
    # norm_embeddings_a = faiss.normalize_L2(embeddings_a)
    # norm_embeddings_b = faiss.normalize_L2(embeddings_b)
    faissIndex.add_with_ids(embeddings_a, np.array(range(0, len(a_lines))))
    dot_product, indexes = faissIndex.search(embeddings_b, k)
    return dot_product, indexes


def compair_sentence(a_lines, b_lines, c_lines, d_lines, dot_product, indexes, distance):
    src = []
    tgt = []
    sim = []
    indexes_iterator = tqdm(indexes)
    for ind, val in enumerate(indexes_iterator):
        if ind not in val:
            val = val[:-1]
        n = 0
        for k, s_idx in enumerate(val):
            if s_idx != ind:
                n += 1
                if n > 5:
                    break
                if n == 1 and dot_product[ind][k] > distance:
                    src.append(S + b_lines[ind]+ St + a_lines[s_idx] + Tt + c_lines[s_idx] + '\n')
                    tgt.append(d_lines[ind] + '\n')
                    sim.append(str(dot_product[ind][k]) + '\n')
                if dot_product[ind][k] <= distance:
                    src.append(S + b_lines[ind]+ St + a_lines[s_idx] + Tt + c_lines[s_idx] + '\n')
                    tgt.append(d_lines[ind] + '\n')
                    sim.append(str(dot_product[ind][k]) + '\n')
                else:
                    break
    # write_file(sim_path, sim)
    return src, tgt


def compair_aug_sentence(a_lines, b_lines, c_lines, indexes):
    src = []

    for ind, val in enumerate(indexes):
        if ind not in val:
            val = val[:-1]
        for s_idx in val:
            if s_idx != ind:
                src.append(sep + a_lines[s_idx] + sep + b_lines[ind] + sep + c_lines[s_idx] + '\n')
    return src


def create_same_domain_sentence(a_lines, b_lines, c_lines, indexes):
    domain_sentences = []
    mono = []
    indexes_iterator = tqdm(indexes)
    for ind, val in enumerate(indexes_iterator):
        if ind not in val:
            val = val[:-1]
        for s_idx in val:
            if s_idx != ind:
                domain_sentences.append(S + a_lines[s_idx] + St + b_lines[ind] + Tt + c_lines[ind] + '\n')
                mono.append(a_lines[s_idx]+ '\n'    )
    return domain_sentences, mono


def write_file(path, sentences):
    with open(path, 'w', encoding="utf-8") as f:
        for i in sentences:
            f.write(i)


def generate_new(top_k_similar, distance, retrieve_dataset, query_lines, retrieve_tar_sen_dataset, query_tar_sen_dataset, new_src_path, new_tgt_path):
    dot_product, indexes = get_k_similar(top_k_similar, retrieve_dataset, query_lines)
    src, tgt = compair_sentence(retrieve_dataset, query_lines, retrieve_tar_sen_dataset, query_tar_sen_dataset, dot_product, indexes, distance)
    bi_list = list(zip(src, tgt))
    random.shuffle(bi_list)
    new_src, new_tgt = list(zip(*bi_list))
    print("the nuber of new generate sentence pairs: ", int(len(new_src)))
    # print(int(len(new_tgt)))
    write_file(new_src_path,  new_src)
    write_file(new_tgt_path, new_tgt)

def generate_aug(top_k_similar, retrieve_dataset, query_lines, retrieve_hsb_tran_dataset, new_src_path):
    indexes = get_k_similar(top_k_similar, retrieve_dataset, query_lines)
    src = compair_aug_sentence(retrieve_dataset, query_lines, retrieve_hsb_tran_dataset, indexes)
    new_src = src[0]
    print(int(len(new_src)))
    write_file(new_src_path, new_src)


def similar_domain_selection(top_k_similar, retrieve_dataset, query_lines, retrieve_tar_sen_dataset, new_src_path, mono_src):
    dot_product, indexes = get_k_similar(top_k_similar, retrieve_dataset, query_lines)
    domain_sentences, mono = create_same_domain_sentence(retrieve_dataset, query_lines, retrieve_tar_sen_dataset, indexes)
    print("the nuber of sentences in monolingual corpus after similar domain selection: ", int(len(domain_sentences)))
    write_file(new_src_path, domain_sentences)
    write_file(mono_src, mono)

if __name__ == '__main__':
    de_lines = read_data(de)
    hsb_lines = read_data(hsb)
    val_de_lines = read_data(val_de)
    val_hsb_lines = read_data(val_hsb)
    test_de_lines = read_data(created_de)
    test_hsb_lines = read_data(created_hsb)
    # mono_de_lines = read_data(mono_de)[:-1]

    # same_domain(2, mono_de_lines, de_lines, mono_de_domain)
    # generate_new(10, 0.5, de_lines, de_lines, hsb_lines, hsb_lines, train_src, train_tgt)
    # generate_new(10, 0.5, de_lines, val_de_lines, hsb_lines, val_hsb_lines, val_src, val_tgt)
    # generate_new(2, 0.5, de_lines, test_de_lines, hsb_lines, test_hsb_lines,  test_src, test_tgt)
    # generate_new(2, 0.5, de_lines, test_de_lines, hsb_lines, test_hsb_lines, created_src, created_tgt)

