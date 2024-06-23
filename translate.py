from pathlib import Path
from model_config import Config, latest_weights_file_path
from generator_model import build_transformer
from Dataset import get_or_build_tokenizer, get_ds, causal_mask, cut_pad
import torch
from tqdm import tqdm
import random
import sys


def translate(config, test=True):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    tokenizer_src = get_or_build_tokenizer(config, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, config.lang_tgt)
    if test:
        test_dataloader = get_ds(config, "test", tokenizer_src, tokenizer_tgt, batch_size=config.test_batch_size)
    else:
        test_dataloader = get_ds(config, "mono", tokenizer_src, tokenizer_tgt, batch_size=config.test_batch_size,
                                 mono_flag=True)

    # Load the model
    model = build_transformer(config.vocab_size, config.vocab_size, config.seq_len, config.seq_len,
                              d_model=config.d_model).to(device)
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # translate the sentence
    sources = []
    label_sentences = []
    translation_sentences = []
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(test_dataloader)
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)
            label = batch['label'].to(device)  # (B, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            prob = model.project(decoder_output)  # (B, seq_len, vocab_size)
            _, output_with_id = torch.max(prob, dim=2)
            sentence_with_id = cut_pad(output_with_id.tolist(), tokenizer_tgt)
            source_with_id = cut_pad(encoder_input.squeeze(0).tolist(), tokenizer_src)
            label_with_id = cut_pad(label.tolist(), tokenizer_tgt)
            batch_source = tokenizer_src.decode(source_with_id)
            batch_translation = tokenizer_tgt.decode(sentence_with_id)
            batch_label = tokenizer_tgt.decode(label_with_id)
            for each_src in batch_source:
                sources.append(each_src)
            for each_label in batch_label:
                label_sentences.append(each_label)
            for each_tran in batch_translation:
                translation_sentences.append(each_tran)

    return sources, label_sentences, translation_sentences


def write_file(translation_sentences, path):
    with open(path, "w", encoding="utf-8") as f:
        for each_sentence in translation_sentences:
            f.write(each_sentence + "\n")


def inference(config, model, test=True, scratch=False):
    # Define the device, tokenizers, and model

    tokenizer_src = get_or_build_tokenizer(config, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, config.lang_tgt)
    if test:
        test_dataloader = get_ds(config, "test", tokenizer_src, tokenizer_tgt, batch_size=config.test_batch_size)

    else:
        test_dataloader = get_ds(config, "created", tokenizer_src, tokenizer_tgt, batch_size=config.test_batch_size,
                                 mono_flag=True)


    model.eval()
    count = 0
    sources = []
    expected = []
    predicted = []

    with torch.no_grad():
        batch_iterator = tqdm(test_dataloader)
        for batch in batch_iterator:
            count += 1
            encoder_input = batch['encoder_input'].to(config.device)  # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(config.device)  # (B, 1, 1, seq_len)
            label = batch['label'].to(config.device)  # (B, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            if scratch:
                max_len = random.randint(1, config.trans_len)
            else:
                max_len = config.trans_len

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len,
                                       config.device)
            source_with_id = cut_pad(encoder_input.squeeze(0).tolist(), tokenizer_src)
            label_with_id = cut_pad(label.squeeze(0).tolist(), tokenizer_tgt)
            batch_source = tokenizer_src.decode(source_with_id)
            batch_label = tokenizer_tgt.decode(label_with_id)
            model_out_text = tokenizer_tgt.decode(model_out)
            sources.append(batch_source)
            expected.append(batch_label)
            predicted.append(model_out_text)
            # if count > 2000:
            #     break

    return sources, expected, predicted


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.encode('[SOS]')[-1]
    eos_idx = tokenizer_tgt.encode('[EOS]')[-1]

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while decoder_input.size(1) < max_len:

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word.tolist()[0] == eos_idx:
            break
    # if decoder_input.squeeze(0)[0] == sos_idx:
    #     model_out = decoder_input.squeeze(0)[1:]
    # else:
    #     model_out = decoder_input.squeeze(0)
    model_out = decoder_input.squeeze(0)
    model_out = cut_pad(model_out.tolist(), tokenizer_tgt)
    return model_out


def main(test, config = Config()):

    print("Using device:", config.device)
    # Load the model
    model = build_transformer(config.vocab_size, config.vocab_size, config.seq_len, config.seq_len,
                                  d_model=config.d_model).to(config.device)
    if test == False:
        model_filename = config.translate_model
    else:
        model_filename = latest_weights_file_path(config)
    
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    source_sentences, label_sentences, translation_sentences = inference(config, model, test)
    # source_sentences, label_sentences, translation_sentences = translate(config)

    # write_file()
    # write_file(source_sentences, "./result/{0}/aug_shuffle.dehsb_hsb.de".format(config.experiment))
    # write_file(translation_sentences, "./result/{0}/aug_shuffle.dehsb_hsb.hsb".format(config.experiment))
    write_file(source_sentences, config.mono_path)
    write_file(label_sentences, config.label_path)
    write_file(translation_sentences, config.translation_path)




if __name__ == "__main__":
    main(test=True)
