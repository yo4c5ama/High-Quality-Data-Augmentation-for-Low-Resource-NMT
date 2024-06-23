#!/usr/bin/env python3
# coding: utf-8
# @Time    : 2023/9/19
# @Author  : Liu_Hengjie
# @Software: PyCharm
# @File    : readFile.py
import random
de_path = 'data/monolingual/de/news.2007.de.shuffled.deduped'
aug_de_paths = 'data/monolingual/de/mono_de_500000.de'

def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text.split('\n')


def write_file(path, sentences):
    with open(path, 'w', encoding="utf-8") as f:
        for i in sentences:
            f.write(i + '\n')


if __name__ == '__main__':
    de = read_data(de_path)
    print(len(de))
    sample_de = random.sample(de, 500000)
    write_file(aug_de_paths, sample_de)
    # for i in range(1, 4):
    #
    #     write_file(aug_de_paths.format(str(i)), sample_de)
    # print(sample_de)
