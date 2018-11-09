import torch
from torch.autograd import Variable
import torch.nn as nn


class TextCnn(nn.Module):
    def __init__(self, config):
        super(TextCnn, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)
        convs = [nn.Sequential(
            nn.Conv1d(in_channels = config.EMBEDDING_DIM,
                    out_channels = config.HIDDING_DIM,
                    kernel_size = kernel_size),
            nn.BatchNorm1d(config.HIDDING_DIM),
            nn.RELU(inplace=True)
            ) for kernel_size in config.KERNEL_SIZE]
        self.Maxpool = [nn.MaxPool1d(kernel_size=(config.SENTENCE_LEN - kernel_size + 1)) for kernel_size in config.KERNEL_SIZE]
        self.sentence_convs = nn.ModuleList(convs)
        self.fc = nn.Sequential(
                nn.Linear(config.HIDDING_DIM * len(config.KERNEL_SIZE), config.FC_HID_DIM),
                nn.BatchNorm1d(config.FC_HID_DIM),
                nn.RELU(inplace=True),
                nn.Linear(config.FC_HID_DIM, config.LABEL_NUM_1)
                )

    def forward(self, sentence):
        sen_encoder = self.embedding(sentence)
        sub_convs = [conv(sen_encoder.permute(0,2,1)) for conv in self.sentence_convs]
        sub_convs_pooled = [self.Maxpool[i](sub_convs[i]) for i in range(len(config.KERNEL_SIZE))]
        conv_out = torch.cat(sub_convs_pooled, dim = 1)
        sub_fc = conv_out.view(conv_out.size(0),-1)
        logits_sub = self.fc(sub_fc)
        logits_sub_sig = torch.sigmoid(logits_sub.unsqueeze(-1))
        return logits_sub

