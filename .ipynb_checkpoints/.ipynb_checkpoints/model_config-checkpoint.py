from pathlib import Path
import torch
import os

def get_weights_file_path(config, step: str):
    model_folder = config.model_folder
    model_filename = f"{config.model_basename}{step}.pt"
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = config.model_folder
    model_filename = f"{config.model_basename}best.pt"
    model_files = list(Path(model_folder).glob(model_filename))
    if len(model_files) == 0:
        return None
    model_files.sort()
    return str(model_files[-1])

def d_latest_weights_file_path(config):
    model_folder = config.model_folder
    model_filename = f"{config.model_basename}d_best.pt"
    model_files = list(Path(model_folder).glob(model_filename))
    if len(model_files) == 0:
        return None
    model_files.sort()
    return str(model_files[-1])


class Config:

    def __init__(self):
        self.experiment = 'GAN_TM'
        self.val_eval_method = "bleu"
        self.use_translation_memory = False
        self.lang_src = 'dehsb'
        self.lang_tgt = 'hsb'
        self.bilingual = '{0}_{1}'.format(self.lang_src, self.lang_tgt)
        self.datasource = './data/' + self.experiment + '/{0}.' + self.bilingual + '.{1}'
        self.tokenizer = "./data/tokenizer/SP_bpe_{0}"
        self.model_folder = "./model/{0}/".format(self.experiment)
        self.result_folder = "./result/{0}/".format(self.experiment)
        self.model_basename = "checkpoint_"
        self.continue_train = False
        self.preload = "latest"
        self.mono_path = "./result/{0}/mono_source.txt".format(self.experiment)
        self.label_path = "./result/{0}/label.txt".format(self.experiment)
        self.translation_path = "./result/{0}/translation.txt".format(self.experiment)
        self.eval_result_path = "./result/{0}/eval_result.txt".format(self.experiment)
        self.created_path = './data/' + self.experiment + '/created.' + self.bilingual + '.{0}'.format(self.lang_tgt)
        self.translate_model = '/mango/files/Tools/accessByAuthor/LabMembers/Students/LIU_Hengjie/Low_resource_NMT_TM_GAN/model/checkpoint_best.pt'

        self.batch_size = 1
        self.val_batch_size = 1
        self.test_batch_size = 1
        self.steps_early_stop = 2
        self.d_stop_step = 20
        self.g_stop_step = 20
        self.train_step = 2
        self.val_step = 50
        self.d_val_step = 50
        self.num_epochs = 2
        self.lr = 1.25e-4
        self.warmup = 5
        self.d_warmup = 5
        self.dropout = 0.15
        self.attn_dropout = 0.1
        self.seq_len = 300
        self.trans_len = 150
        self.d_model = 512
        self.vocab_size = 8000
        self.tensorboard_comment = "{0} batch_size={1} lr={2} validate_step={3}".format(self.experiment,
                                                                                        self.batch_size, self.lr,
                                                                                        self.val_step)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __str__(self):
        message=("Please modify the model_config.py file if you use our code.\nThe parameters and path set here are just for toy data.\n"
                 "---------------file path------------------\n"
                 "experiment: {0}\n"
                 "source language: {1}\n"
                 "target language: {2}\n"
                 "save model folder: {3}\n"
                 "save result folder:{4}\n"
                 "---------------Parameters-----------------\n"
                 "batch_size: {5}\n"
                 "the number of epochs: {6}\n"
                 "learning rate: {7}\n"
                 "dropout: {8}\n".format(self.experiment, self.lang_src, self.lang_tgt, self.model_folder, self.result_folder, self.batch_size, self.num_epochs-1, self.lr, self.dropout))

        return message

class Transformer_Config:

    def __init__(self):
        self.experiment = 'aug_double'
        self.val_eval_method = "bleu"
        self.use_translation_memory = False
        self.lang_src = 'de'
        self.lang_tgt = 'hsb'
        self.bilingual = '{0}_{1}'.format(self.lang_src, self.lang_tgt)
        self.datasource = './data/' + self.experiment + '/{0}.' + self.bilingual + '.{1}'
        self.tokenizer = "./data/tokenizer/SP_bpe_{0}"
        self.model_folder = "./model/{0}/".format(self.experiment)
        self.result_folder = "./result/{0}/".format(self.experiment)
        self.model_basename = "checkpoint_"
        self.continue_train = False
        self.preload = "latest"
        self.mono_path = "./result/{0}/mono_source.txt".format(self.experiment)
        self.label_path = "./result/{0}/label.txt".format(self.experiment)
        self.translation_path = "./result/{0}/translation.txt".format(self.experiment)
        self.eval_result_path = "./result/{0}/eval_result.txt".format(self.experiment)
        self.created_path = './data/' + self.experiment + '/created.' + self.bilingual + '.{0}'.format(self.lang_tgt)

        self.batch_size = 1
        self.val_batch_size = 1
        self.test_batch_size = 1
        self.steps_early_stop = 2
        self.d_stop_step = 20
        self.g_stop_step = 20
        self.train_step = 2
        self.val_step = 50
        self.d_val_step = 50
        self.num_epochs = 2
        self.lr = 1.25e-4
        self.warmup = 5
        self.d_warmup = 5
        self.dropout = 0.15
        self.attn_dropout = 0.1
        self.seq_len = 300
        self.trans_len = 150
        self.d_model = 512
        self.vocab_size = 8000
        self.tensorboard_comment = "{0} batch_size={1} lr={2} validate_step={3}".format(self.experiment,
                                                                                        self.batch_size, self.lr,
                                                                                        self.val_step)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __str__(self):
        message=("Please modify the model_config.py file if you use our code.\nThe parameters and path set here are just for toy data.\n"
                 "---------------file path------------------\n"
                 "experiment: {0}\n"
                 "source language: {1}\n"
                 "target language: {2}\n"
                 "save model folder: {3}\n"
                 "save result folder:{4}\n"
                 "---------------Parameters-----------------\n"
                 "batch_size: {5}\n"
                 "the number of epochs:{6}\n"
                 "learning rate: {7}\n"
                 "dropout: {8}\n".format(self.experiment, self.lang_src, self.lang_tgt, self.model_folder, self.result_folder, self.batch_size, self.num_epochs-1, self.lr, self.dropout))

        return message
class Config_CNN:
    """配置参数"""

    def __init__(self):
        self.model_name = "DiscriminatorCNN"
        self.save_path = 'source/saved_dict/' + self.model_name + '.ckpt'
        self.learning_rate = 1e-5
        self.hidden_size = 16
        self.filter_sizes = 1
        self.num_filters = 256
        self.dropout = 0.1


class Config_LSTM:
    """配置参数"""

    def __init__(self):
        self.model_name = "DiscriminatorBiLSTM"
        self.save_path = 'source/saved_dict/' + self.model_name + '.ckpt'
        self.learning_rate = 1e-5
        self.hidden_size = 768
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout = 0.1


class Config_Fusion:
    """配置参数"""

    def __init__(self):
        self.model_name = "DiscriminatorFusion"
        self.save_path = 'source/saved_model/' + self.model_name
        self.learning_rate = 1e-5
        self.hidden_size = 16
        self.filter_sizes = 1
        self.num_filters = 256
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout = 0.1
