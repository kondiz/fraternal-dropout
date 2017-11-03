import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import fixMaskEmbeddedDropout
from weight_drop import WeightDrop

class fixMaskDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(fixMaskDropout, self).__init__()
        self.dropout = dropout
        self.mask = None
    
    def forward(self, draw_mask, input):
        if self.training == False:
            return input
        if self.mask is None or draw_mask==True:
            self.mask =  input.data.new().resize_(input.size()).bernoulli_(1 - self.dropout) / (1 - self.dropout)
        mask = Variable(self.mask)
        masked_input = mask*input
        return masked_input

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, dropout=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.idrop = fixMaskDropout(dropouti)
        self.drop = fixMaskDropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.embedded_dropout = fixMaskEmbeddedDropout(self.encoder, dropoute)
        self.lstm = WeightDrop(torch.nn.LSTM(ninp, ninp), ['weight_hh_l0'], dropout=wdrop)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight_raw

        self.init_weights()

        self.ninp = ninp
        self.dropoute = dropoute

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight_raw.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, draw_mask_e=True, draw_mask_i=True, draw_mask_w=True, draw_mask_o=True):
        emb = self.embedded_dropout(draw_mask_e, input)
        
        emb_i = self.idrop(draw_mask_i, emb)

        raw_output, hidden = self.lstm(draw_mask_w, emb_i, hidden)
        output = self.drop(draw_mask_o, raw_output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        if return_h:
            return result, hidden, raw_output, output
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, self.ninp).zero_()),
                Variable(weight.new(1, bsz, self.ninp).zero_()))
