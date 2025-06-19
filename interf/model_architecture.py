import torch
import torch.nn as nn
from torch.nn import GRU
from transformers import AutoModel

MODEL_NAME = 'microsoft/codebert-base'

class MultiTaskModel(nn.Module):
    def __init__(self, n_tb_classes, n_bt_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        for layer in self.bert.encoder.layer[:2]:
            for param in layer.parameters():
                param.requires_grad = False

        self.gru = GRU(input_size=768, hidden_size=384, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(768)
        self.traceback_head = nn.Linear(768, n_tb_classes)
        self.bugtype_head = nn.Linear(768, n_bt_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        gru_output, _ = self.gru(sequence_output)
        cls_output = gru_output[:, 0, :]
        x = self.dropout(cls_output)
        x = self.batchnorm(x)
        return self.traceback_head(x), self.bugtype_head(x)
