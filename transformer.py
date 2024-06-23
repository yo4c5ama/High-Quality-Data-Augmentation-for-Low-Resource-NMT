#!/usr/bin/env python3
# coding: utf-8
# @Time    : 2023/12/9
# @Author  : Liu_Hengjie
# @Software: PyCharm
# @File    : train_my.py

from generator_model import build_transformer
from Dataset import BilingualDataset, causal_mask, get_ds, get_or_build_tokenizer, cut_pad
from model_config import Config, get_weights_file_path, latest_weights_file_path, Transformer_Config
import os
from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from datetime import datetime


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


def init_or_reload_model(config, device, if_reload):
    model = get_model(config, config.vocab_size, config.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.998), eps=1e-9, foreach=False)
    warmup_scheduler = WarmUpLR(optimizer, config.warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda epoch: torch.rsqrt(torch.tensor(epoch + 1)).float())
    initial_epoch = 1
    global_step = 0
    preload = config.preload
    # model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config,preload) if preload else None
    model_filename = latest_weights_file_path(config)
    if if_reload:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        warmup_scheduler = WarmUpLR(optimizer, config.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: torch.rsqrt(torch.tensor(epoch + 1)).float())

    else:
        print('No model to preload, starting from scratch')
    return initial_epoch, global_step, model, optimizer, warmup_scheduler, scheduler


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
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
    val_dataloader = get_ds(config, "val", tokenizer_src, tokenizer_tgt, config.test_batch_size)

    # Build model

    # If the user specified a model to preload before training, load it
    initial_epoch, global_step, model, optimizer, warmup_scheduler, scheduler = init_or_reload_model(config, device,
                                                                                                     if_reload=config.continue_train)
    # Tensorboard
    writer = SummaryWriter(comment=config.tensorboard_comment)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.encode('[PAD]')[-1], label_smoothing=0.1).to(device)
    if config.val_eval_method == 'bleu':
        best_matrix = 0
    else:
        best_matrix = 100
    no_improve_steps = 0
    early_stop = False
    for epoch in range(initial_epoch, config.num_epochs):
        # torch.cuda.empty_cache()
        print("\n\nepoch {} training process:".format(epoch))
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad()
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, config.vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            if global_step % config.train_step == 0:
                if global_step > config.train_step * config.warmup:
                    # lr decay
                    scheduler.step()
                else:
                    warmup_scheduler.step()

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            # Log the lr
            lr = optimizer.state_dict()["param_groups"][0]['lr']
            writer.add_scalar('Learning rate', lr, global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()
            # Update the weights
            optimizer.step()


            if global_step % config.val_step == 0 and global_step != 0:
                # lr decay
                # scheduler.step()
                # Run validation at the end of every epoch
                print("\n\nvalidate G model:")
                val_loss = run_validation(model, config, loss_fn, val_dataloader, device, global_step, writer)
                val_bleu = inference(model, config, val_dataloader, tokenizer_src, tokenizer_tgt, device, global_step, writer)
                if config.val_eval_method == 'bleu':
                    best_matrix = max(best_matrix, val_bleu)
                    if val_bleu < best_matrix:
                        no_improve_steps += 1
                    # Save the model when results on val get better
                    else:
                        no_improve_steps = 0
                        model_filename = get_weights_file_path(config, "best")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step
                        }, model_filename)
                else:
                    best_matrix = min(best_matrix, val_loss)
                    if val_loss > best_matrix:
                        no_improve_steps += 1
                    # Save the model when results on val get better
                    else:
                        no_improve_steps = 0
                        model_filename = get_weights_file_path(config, "best")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step
                        }, model_filename)
            global_step += 1
        #         if global_step >= config.g_stop_step:
        #             early_stop = True
        #             break
        #         # if no_improve_steps >= config.steps_early_stop:
        #         #     early_stop = True
        #         #     break
        #     global_step += 1
        #
        # if early_stop:
        #     break





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
            writer.add_scalar('validation loss', avg_val_loss, global_step)
            writer.flush()

    return avg_val_loss


def inference(model, config, validation_ds, tokenizer_src, tokenizer_tgt, device, global_step, writer):
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

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config.seq_len,
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
        writer.add_scalar('validation bleu', bleu_score, global_step)
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
    config = Transformer_Config()
    print(config)
    if not os.path.exists(config.model_folder):
        os.makedirs(config.model_folder)
    if not os.path.exists(config.result_folder):
        os.makedirs(config.result_folder)
    train_model(config)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = Config()
    if not os.path.exists(config.model_folder):
        os.makedirs(config.model_folder)
    if not os.path.exists(config.result_folder):
        os.makedirs(config.result_folder)
    train_model(config)
