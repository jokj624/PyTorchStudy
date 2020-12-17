# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:17:51 2020

@author: gh
"""

import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

vocab_size = 256 #총 아스키 코드 개수

x_ = list(map(ord, "hungry"))
y_ = list(map(ord, "affamata"))
x = torch.LongTensor(x_)
y = torch.LongTensor(y_)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, inputs, targets):
        initial_state = self._init_state()
        embedding = self.embedding(inputs).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([0])
        
        outputs = []
        for i in range(targets.size()[0]):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            projection = self.project(decoder_output)
            outputs.append(projection)
            decoder_input = torch.LongTensor([targets[i]])
            
        outputs = torch.stack(outputs).squeeze()

        return outputs
        
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
        
seq2seq2 = Seq2Seq(vocab_size, 16)
criterion= nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq2.parameters(), lr = 1e-3)

log = []
for i in range(1000):
    prediction = seq2seq2(x, y)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val = loss.data
    log.append(loss_val)
    if i % 100 == 0:
        print("\n 반복:%d 오차: %s" % (i, loss_val.item()))
        _, top1 = prediction.data.topk(1,1)
        print([chr(c) for c in top1.squeeze().numpy().tolist()])
        
plt.plot(log)
plt.ylabel('cross entropy loss')
plt.show()        