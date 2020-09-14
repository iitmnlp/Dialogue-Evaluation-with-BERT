#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.7.10

'''
Unreference model with bert embedding
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class BERT_RUBER_unrefer(nn.Module):
    
    def __init__(self,embedding_size, dropout=0.5):
        super(BERT_RUBER_unrefer, self).__init__()
        
        self.M = nn.Parameter(torch.rand(embedding_size, embedding_size))
        self.layer1 = nn.Linear(embedding_size * 2 , 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 128)
        self.opt = nn.Linear(128, 1, bias=False)
        self.drop = nn.Dropout(p=dropout)
        
    def forward(self, query, reply):
        # query / replty: 76-dim bert embedding
        # [B, H]        
        qh = query.unsqueeze(1)    # [B, 1, 768]
        rh = reply.unsqueeze(2)    # [B, 768, 1]
        # score = torch.bmm(torch.matmul(qh, self.M), rh).squeeze(2)  # [B, 1]
        qh = qh.squeeze(1)    # [B, H]
        rh = rh.squeeze(2)    # [B, H]
        linear = torch.cat([qh, rh], 1)    # [B, 2 * H  + 1]
        linear = self.drop(torch.tanh(self.layer1(linear)))
        linear = self.drop(torch.tanh(self.layer2(linear)))
        linear = torch.tanh(self.layer3(linear))
        linear = torch.sigmoid(self.opt(linear).squeeze(1))  # [B]
        return linear


if __name__ == "__main__":
    unrefer = BERT_RUBER_unrefer(200)