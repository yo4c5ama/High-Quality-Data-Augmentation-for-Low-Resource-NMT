#!/usr/bin/env python3
# coding: utf-8
# @Time    : 2023/12/11
# @Author  : Liu_Hengjie
# @Software: PyCharm
# @File    : train.py

from generator_model import build_transformer
from discriminator_model import DiscriminatorBiLSTM, DiscriminatorCNN, DiscriminatorFusion
from model_config import Config, Config_LSTM, Config_CNN, Config_Fusion, get_weights_file_path, \
    latest_weights_file_path, d_latest_weights_file_path
from Dataset import BilingualDataset, causal_mask, get_ds, get_or_build_tokenizer, cut_pad
import translate
from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
import os
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.contrib import tzip


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config.seq_len, config.seq_len,
                              d_model=config.d_model)
    return model


def init_or_reload_model(config, model_config, device, if_reload):
    g_model = get_model(config, config.vocab_size, config.vocab_size).to(device)
    d_model = DiscriminatorFusion(model_config).to(device)
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=config.lr, eps=1e-9, foreach=False)
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=5e-5, eps=1e-9, foreach=False)
    d_warmup_scheduler = WarmUpLR(d_optimizer, config.d_warmup)
    g_warmup_scheduler = WarmUpLR(g_optimizer, config.warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer,
                                                  lr_lambda=lambda epoch: torch.rsqrt(torch.tensor(epoch + 1)).float())
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda epoch: torch.rsqrt(
        torch.tensor(epoch + 1)).float())
    initial_epoch = 1
    global_step = 0
    g_train_step = 0
    preload = config.preload
    g_model_filename = latest_weights_file_path(config)

    if if_reload:
        print(f'Preloading model {g_model_filename}')
        state = torch.load(g_model_filename)
        g_model.load_state_dict(state['model_state_dict'])
        g_train_step = state['global_step']
        g_optimizer.load_state_dict(state['optimizer_state_dict'])
        g_warmup_scheduler = WarmUpLR(g_optimizer, config.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda epoch: torch.rsqrt(torch.tensor(epoch + 1)).float())
    else:
        print('No model to preload, starting from scratch')
    return initial_epoch, g_train_step, global_step, g_model, g_optimizer, d_model, d_optimizer, scheduler, g_warmup_scheduler, d_scheduler, d_warmup_scheduler


def train_model(config, model_config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    # device = config.device
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Build tokenizers and dataloader
    tokenizer_src = get_or_build_tokenizer(config, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, config.lang_tgt)
    train_dataloader = get_ds(config, "train", tokenizer_src, tokenizer_tgt, config.batch_size)
    d_val_dataloader = get_ds(config, "val", tokenizer_src, tokenizer_tgt, config.batch_size)
    g_val_dataloader = get_ds(config, "val", tokenizer_src, tokenizer_tgt, config.test_batch_size)

    # Build model

    # If the user specified a model to preload before training, load it
    initial_epoch, g_train_step, global_step, G_model, G_optimizer, D_model, D_optimizer, G_scheduler, G_warmup_scheduler, D_scheduler, d_warmup_scheduler = init_or_reload_model(
        config, model_config, device, if_reload=config.continue_train)
    # Tensorboard
    print(D_model)
    print(G_model)
    writer = SummaryWriter(comment=config.tensorboard_comment)

    d_train_step = 0
    g_train_step = g_train_step

    g_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.encode('[PAD]')[-1], label_smoothing=0.1).to(device)
    d_loss_fn = nn.BCELoss().to(device)

    if config.val_eval_method == 'bleu':
        best_global_matrix = 0
    else:
        best_global_matrix = 100
    D_continue = False
    global_stop = 0
    for epoch in range(initial_epoch, config.num_epochs):
        # torch.cuda.empty_cache()

        if D_continue:
            # print("reload D model:\n")
            d_model_filename = d_latest_weights_file_path(config)
            d_state = torch.load(d_model_filename)
            D_model.load_state_dict(d_state['model_state_dict'])
            d_train_step = (d_state['global_step'] // config.d_stop_step + 1) * config.d_stop_step
            D_optimizer.load_state_dict(d_state['optimizer_state_dict'])

            # D_scheduler = torch.optim.lr_scheduler.LambdaLR(D_optimizer, lr_lambda=lambda k: torch.rsqrt(
            #     torch.tensor(k + 1)).float())
            #
            # d_warmup_scheduler = WarmUpLR(D_optimizer, config.d_warmup)
            # D_scheduler.step()
        # if epoch != 0:
        #     _, _, translation_sentences = translate.inference(config, G_model, test=False)
        #     translate.write_file(translation_sentences, config.created_path)
        # else:
        #     _, _, translation_sentences = translate.inference(config, G_model, test=False)
        #     translate.write_file(translation_sentences, config.created_path)
        print("\n\ntranslate monolingual corpus:")
        _, _, translation_sentences = translate.inference(config, G_model, test=False)
        translate.write_file(translation_sentences, config.created_path)
        created_dataloader = get_ds(config, "created", tokenizer_src, tokenizer_tgt, config.batch_size)
        G_model.train()
        d_early_stop = False
        D_continue = True
        d_stop_count = 0
        d_best_loss = 10000
        d_self_step = 0

        for d_epoch in range(initial_epoch, config.num_epochs):
            if epoch == 0:
                break
            print("\n\nD training process:")
            d_batch_iterator = tqdm(tzip(train_dataloader, created_dataloader), desc=f"Processing Epoch {d_epoch:02d}")
            D_model.train()
            for train_batch, created_batch in d_batch_iterator:
                if d_self_step % config.train_step == 0:
                    if d_self_step > config.train_step * config.d_warmup:
                        # lr decay
                        D_scheduler.step()
                    else:
                        d_warmup_scheduler.step()
                train_label = train_batch['label'].to(device)  # (B, seq_len)
                created_tgt = created_batch['decoder_input'].to(device)  # (B, seq_len)

                real_data = train_label
                d_real = D_model(real_data)
                d_fake = D_model(created_tgt)

                d_real_loss = d_loss_fn(d_real, torch.ones_like(d_real))
                d_fake_loss = d_loss_fn(d_fake, torch.zeros_like(d_fake))
                d_loss = d_real_loss + d_fake_loss
                D_optimizer.zero_grad()
                d_loss.backward()
                D_optimizer.step()

                d_train_step += 1
                d_self_step += 1

                d_lr = D_optimizer.state_dict()["param_groups"][0]['lr']

                writer.add_scalar('D Learning rate', d_lr, d_train_step)
                writer.add_scalars('Discriminator_loss',
                                   {'d_real_loss': d_real_loss.item(), 'd_fake_loss': d_fake_loss.item(),
                                    'd_loss': d_loss.item()}, d_train_step)
                writer.flush()
                d_batch_iterator.set_postfix({"d_loss": f"{d_loss.item():6.3f}", "d_lr": "{}".format(d_lr)})

                if d_train_step % config.d_val_step == 0 and d_train_step != 0:

                    d_val_loss = D_validate(D_model, d_loss_fn, G_model, d_val_dataloader, device, d_train_step, writer)
                    D_model.train()
                    G_model.train()
                    if d_val_loss >= d_best_loss:
                        d_stop_count += 1
                    else:
                        print("\n\nsave D model:")
                        d_stop_count = 0
                        model_filename = get_weights_file_path(config, "d_best")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': D_model.state_dict(),
                            'optimizer_state_dict': D_optimizer.state_dict(),
                            'global_step': d_train_step
                        }, model_filename)
                        D_continue = True
                    d_best_loss = min(d_best_loss, d_val_loss)

                    # if d_stop_count >= 5:
                    #     d_early_stop = True
                    #     break
            #         if d_self_step > config.d_stop_step:
            #             d_early_stop = True
            #             break
            # if d_early_stop:
            #     break

        early_stop = False
        no_improve_steps = 0
        if config.val_eval_method == 'bleu':
            best_matrix = 0
        else:
            best_matrix = 100
        for g_epoch in range(initial_epoch, config.num_epochs):
            print("\n\nG training process:")
            g_batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            for train_batch in g_batch_iterator:
                if g_train_step % config.train_step == 0:
                    if g_train_step > config.train_step * config.warmup:
                        # lr decay
                        G_scheduler.step()
                    else:
                        G_warmup_scheduler.step()
                train_encoder_input = train_batch['encoder_input'].to(device)  # (b, seq_len)
                train_decoder_input = train_batch['decoder_input'].to(device)  # (B, seq_len)
                train_encoder_mask = train_batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
                train_decoder_mask = train_batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)
                train_label = train_batch['label'].to(device)  # (B, seq_len)

                # Run the tensors through the encoder, decoder of the transformer and get the generation
                train_encoder_output = G_model.encode(train_encoder_input, train_encoder_mask)  # (B, seq_len, d_model)
                train_decoder_output = G_model.decode(train_encoder_output, train_encoder_mask, train_decoder_input,
                                                      train_decoder_mask)  # (B, seq_len, d_model)
                proj_output = G_model.project(train_decoder_output)  # (B, seq_len, vocab_size)
                _, output_with_id = torch.max(proj_output, dim=2)
                g_cheat = D_model(output_with_id)
                g_cheat_loss = d_loss_fn(g_cheat, torch.ones_like(g_cheat)) * 0.002
                g_translation_loss = g_loss_fn(proj_output.view(-1, config.vocab_size), train_label.view(-1))
                g_loss = g_cheat_loss + g_translation_loss
                G_optimizer.zero_grad()
                g_loss.backward()
                G_optimizer.step()
                g_lr = G_optimizer.state_dict()["param_groups"][0]['lr']
                writer.add_scalar('G Learning rate', g_lr, g_train_step)
                writer.add_scalars('Generator_loss', {'g_cheat_loss': g_cheat_loss.item(),
                                                      'g_translation_loss': g_translation_loss.item(),
                                                      'g_loss': g_loss.item()}, g_train_step)
                writer.flush()
                g_batch_iterator.set_postfix({"g_loss": f"{g_loss.item():6.3f}"})
                g_train_step += 1
                if g_train_step % config.val_step == 0 and g_train_step != 0:
                    # print(g_train_step)
                    print("\n\nvalidate G model:")
                    # Run validation at the end of every epoch
                    val_loss = run_validation(G_model, config, g_loss_fn, g_val_dataloader, device, g_train_step, writer)
                    val_bleu = inference(G_model, config, g_val_dataloader, tokenizer_src, tokenizer_tgt, device, g_train_step,
                                         writer)
                    if config.val_eval_method == 'bleu':

                        if val_bleu < best_matrix:
                            no_improve_steps += 1
                        # Save the model when results on val get better
                        else:
                            no_improve_steps = 0
                            model_filename = get_weights_file_path(config, "best")
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': G_model.state_dict(),
                                'optimizer_state_dict': G_optimizer.state_dict(),
                                'global_step': g_train_step
                            }, model_filename)
                        best_matrix = max(best_matrix, val_bleu)
                        best_global_matrix = max(best_matrix, best_global_matrix)
                    else:

                        if val_loss > best_matrix:
                            no_improve_steps += 1
                        # Save the model when results on val get better
                        else:
                            no_improve_steps = 0
                            model_filename = get_weights_file_path(config, "best")
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': G_model.state_dict(),
                                'optimizer_state_dict': G_optimizer.state_dict(),
                                'global_step': g_train_step
                            }, model_filename)
                        best_matrix = min(best_matrix, val_loss)
                        best_global_matrix = min(best_matrix, best_global_matrix)
                    # if epoch == 0:
                    #     if g_train_step >= config.g_stop_step:
                    #         early_stop = True
                    #         break

                    # if no_improve_steps >= config.steps_early_stop:
                    #     early_stop = True
                    #     break

            # if early_stop:
            #     if best_global_matrix > best_matrix:
            #         global_stop += 1
            #     else:
            #         global_stop = 0
            #     break

        # if g_train_step >= config.g_stop_step:
        #     break


def D_validate(d_model, d_loss_fn, g_model, validation_ds, device, d_train_step, writer):
    g_model.eval()
    d_model.eval()
    count = 0
    D_val_loss_total = 0
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)
            label = batch['label'].to(device)  # (B, seq_len)
            real_data = label
            d_real = d_model(real_data)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = g_model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = g_model.decode(encoder_output, encoder_mask, decoder_input,
                                            decoder_mask)  # (B, seq_len, d_model)
            proj_output = g_model.project(decoder_output)  # (B, seq_len, vocab_size)
            _, output_with_id = torch.max(proj_output, dim=2)
            d_fake = d_model(output_with_id.detach())

            d_real_loss = d_loss_fn(d_real, torch.ones_like(d_real))
            d_fake_loss = d_loss_fn(d_fake, torch.zeros_like(d_fake))
            d_loss = d_real_loss + d_fake_loss
            D_val_loss_total += d_loss
        avg_D_val_loss = D_val_loss_total / count

        if writer:
            writer.add_scalar('Discriminator validation loss', avg_D_val_loss, d_train_step)
            writer.flush()
    return avg_D_val_loss


def run_validation(model, config, loss_fn, validation_ds, device, global_step, writer):
    model.eval()
    count = 0
    val_loss_total = 0
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)
            label = batch['label'].to(device)  # (B, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)
            output_tensor = proj_output.view(-1, config.vocab_size)
            label_tensor = label.view(-1)
            val_loss = loss_fn(output_tensor, label_tensor)
            # Compare the output with the label. Compute the loss using a simple cross entropy
            val_loss_total += val_loss
        avg_val_loss = val_loss_total / count

        if writer:
            writer.add_scalar('Generator validation loss', avg_val_loss, global_step)
            writer.flush()

    return avg_val_loss


def inference(model, config, validation_ds, tokenizer_src, tokenizer_tgt, device, global_step, writer, scratch=False):
    # Define the device, tokenizers, and model
    model.eval()
    sources = []
    expected = []
    reference = []
    predicted = []

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds)
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            label = batch['label'].to(device)  # (B, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            if scratch:
                max_len = random.randint(0, config.trans_len)
            else:
                max_len = config.trans_len
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len,
                                      device)
            source_with_id = cut_pad(encoder_input.squeeze(0).tolist(), tokenizer_src)
            label_with_id = cut_pad(label.squeeze(0).tolist(), tokenizer_tgt)
            batch_source = tokenizer_src.decode(source_with_id)
            batch_label = tokenizer_tgt.decode(label_with_id)
            model_out_text = tokenizer_tgt.decode(model_out)
            sources.append(batch_source)
            expected.append(batch_label)
            predicted.append(model_out_text)

    reference.append(expected)
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predicted, reference).score
    if writer:
        writer.add_scalar('Generator validation bleu', bleu_score, global_step)
        writer.flush()
    return bleu_score


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

def begin_training():
    warnings.filterwarnings("ignore")
    config = Config()
    model_config = Config_Fusion()
    print(config)
    if not os.path.exists(config.model_folder):
        os.makedirs(config.model_folder)
    if not os.path.exists(config.result_folder):
        os.makedirs(config.result_folder)
    train_model(config, model_config)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = Config()
    model_config = Config_Fusion()
    print(config)
    if not os.path.exists(config.model_folder):
        os.makedirs(config.model_folder)
    if not os.path.exists(config.result_folder):
        os.makedirs(config.result_folder)
    train_model(config, model_config)
