#!/usr/bin/env python3
# coding: utf-8
# @Time    : 2023/12/11
# @Author  : Liu_Hengjie
# @Software: PyCharm
# @File    : evaluation.py
import sys
from sacrebleu.metrics import BLEU, CHRF, TER
from model_config import Config
import os
import re

def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text.split('\n')


def write_file(path, sentences):
    with open(path, 'w', encoding="utf-8") as f:
        for i in sentences:
            f.write(i + '\n')


def evaluation(config):
    label_path = config.label_path
    trans_path = config.translation_path
    eval_path = config.eval_result_path
    command = "sacrebleu {0} -i {1} -m bleu chrf ter --confidence -f text --short > {2}".format(label_path, trans_path,
                                                                                                eval_path)
    f = os.popen(command)
    print(f.read())

def cal_bleu(config = Config()):
    label = []
    label.append(read_data(config.label_path))
    trans = read_data(config.translation_path)
    bleu = BLEU()
    bleu_score = bleu.corpus_score(trans, label).score
    chrF = CHRF()
    chrF2 = chrF.corpus_score(trans, label).score
    ter = TER()
    ter_score = ter.corpus_score(trans, label).score
    
    # print(bleu_score, chrF2)
    return bleu_score, chrF2, ter_score


def main(config = Config()):
    cal_bleu(config)
    evaluation(config)
    for i in read_data(config.eval_result_path):

        print(i)


if __name__ == "__main__":
    config = Config()
    cal_bleu(config)
    evaluation(config)
