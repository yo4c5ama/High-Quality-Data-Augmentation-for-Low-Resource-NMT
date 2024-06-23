#!/usr/bin/env python3
# coding: utf-8
# @Time    : 2023/12/9
# @Author  : Liu_Hengjie
# @Software: PyCharm
# @File    : Dataset_local.py
import torch
from pathlib import Path
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader



def cut_pad(output_with_id, tokenizer_tgt):
    SOS_id = tokenizer_tgt.encode('[SOS]')[-1]
    EOS_id = tokenizer_tgt.encode("[EOS]")[-1]
    sentences = []
    for each in output_with_id:
        sentence = []
        if type(each) is not list:
            if each != SOS_id:
                if each != EOS_id:
                    sentences.append(each)
                else:
                    break
        else:
            for word in each:
                if each != SOS_id:
                    if word != EOS_id:
                        sentence.append(word)
                    else:
                        break
            sentences.append(sentence)
    return sentences


def fix_pad(config, output_with_id, tokenizer_tgt):
    SOS_id = tokenizer_tgt.encode('[SOS]')[-1]
    EOS_id = tokenizer_tgt.encode("[EOS]")[-1]
    PAD_id = tokenizer_tgt.encode("[PAD]")[-1]
    sentences = []
    for each in output_with_id:
        sentence = []
        for word in each:
            if each != SOS_id:
                if word != EOS_id:
                    sentence.append(word)
                else:
                    break
        if len(sentence) == 100:
            sentence = sentence[:-2]
        label = torch.cat(
            [
                torch.tensor(sentence, dtype=torch.int64),
                torch.tensor([EOS_id], dtype=torch.int64),
                torch.tensor([torch.tensor([PAD_id], dtype=torch.int64)] * (config.seq_len - len(sentence) - 1), dtype=torch.int64),
            ],
            dim=0,
        )
        if label.size(0) != config.seq_len:
            print(len(label),label,sentence)
        sentences.append(label.unsqueeze(0))
    return torch.cat(sentences)


def get_or_build_tokenizer(config, lang):
    if config.use_translation_memory:
        tokenizer_path = config.tokenizer.format(config.lang_src)
        model_path = config.tokenizer.format(config.lang_src) + ".model"
        vocab_path = config.datasource.format('train', "vocab")
        if not Path.exists(Path(vocab_path)):
            with open(vocab_path, 'w') as vocab:
                with open('./data/transformer_baseline/train.de_hsb.de', 'r') as src:
                    for line in src:
                        vocab.write(line)
                with open('./data/transformer_baseline/train.de_hsb.hsb', 'r') as tgt:
                    for line in tgt:
                        vocab.write(line)
    else:
        tokenizer_path = config.tokenizer.format(lang)
        model_path = config.tokenizer.format(lang) + ".model"
        vocab_path = config.datasource.format('train', lang)
    if not Path.exists(Path(model_path)):
        spm.SentencePieceTrainer.train(input=vocab_path, model_prefix=tokenizer_path, model_type="bpe",
                                       vocab_size=config.vocab_size,
                                       character_coverage=1.0, user_defined_symbols=["[UNK]", "[PAD]", "[SOS]", "[EOS]",
                                                                                     "[SEP]", "S:", "St:", "Tt:"])
        tokenizer = spm.SentencePieceProcessor(model_file=model_path)
    else:
        tokenizer = spm.SentencePieceProcessor(model_file=model_path)
    return tokenizer


def read_data_file(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read().split("\n")
    if text[-1] == "":
        return text[:-1]
    else:
        return text


def get_ds(config, dataset_name, tokenizer_src, tokenizer_tgt, batch_size, mono_flag=False):

    if mono_flag:
        dataset_src = read_data_file(config.datasource.format(dataset_name, config.lang_src))
        dataset_tgt = read_data_file(config.datasource.format("train", config.lang_tgt))
    else:
        # get train and val data
        dataset_src = read_data_file(config.datasource.format(dataset_name, config.lang_src))
        dataset_tgt = read_data_file(config.datasource.format(dataset_name, config.lang_tgt))
    dataset_ds = BilingualDataset(dataset_src, dataset_tgt, tokenizer_src, tokenizer_tgt, config.lang_src,
                                  config.lang_tgt, config.seq_len)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for src, tgt in zip(dataset_src, dataset_tgt):
        max_len_src = max(max_len_src, len(tokenizer_src.encode(src)))
        max_len_tgt = max(max_len_tgt, len(tokenizer_tgt.encode(tgt)))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    dataset_dataloader = DataLoader(dataset_ds, batch_size=batch_size, shuffle=True)

    return dataset_dataloader


class BilingualDataset(Dataset):

    def __init__(self, src, tgt, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.src_text = src
        self.tgt_text = tgt
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sep_token = torch.tensor([tokenizer_tgt.encode("[SEP]")[-1]], dtype=torch.int64)
        self.sos_token = torch.tensor([tokenizer_tgt.encode("[SOS]")[-1]], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.encode("[EOS]")[-1]], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.encode("[PAD]")[-1]], dtype=torch.int64)

    def __len__(self):
        return len(self.src_text)

    def __getitem__(self, idx):
        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(self.src_text[idx])
        dec_input_tokens = self.tokenizer_tgt.encode(self.tgt_text[idx])

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0:
            # Add <s> and </s> token
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens[0:self.seq_len-2], dtype=torch.int64),
                    self.eos_token
                ],
                dim=0,
            )
        else:
            # Add <s> and </s> token
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )
        if dec_num_padding_tokens < 0:
            # Add only <s> token
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(dec_input_tokens[0:self.seq_len-1], dtype=torch.int64)
                ],
                dim=0,
            )

            # Add only </s> token
            label = torch.cat(
                [
                    torch.tensor(dec_input_tokens[0:self.seq_len-1], dtype=torch.int64),
                    self.eos_token
                ],
                dim=0,
            )
        else:
            # Add only <s> token
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

            # Add only </s> token
            label = torch.cat(
                [
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

        # Double-check the size of the tensors to make sure they are all seq_len long
        # if encoder_input.size(0) != self.seq_len:
        #     print(len(enc_input_tokens), self.sos_token, self.eos_token, encoder_input.size(0), decoder_input.size(0), label.size(0))
        # if decoder_input.size(0) != self.seq_len:
        #     print(len(enc_input_tokens), self.sos_token, self.eos_token, encoder_input.size(0), decoder_input.size(0), label.size(0))
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": self.src_text,
            "tgt_text": self.tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
