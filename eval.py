import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='/u/zolnakon/repos/awd-lstm-lm/data/penn/',
                    help='location of the data corpus')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--model_path', type=str,  default='PTB.pt', #'/data/lisa/exp/zolnakon/last/40738.pt',
                    help='path to save the final model')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, eval_batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Evaluating code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

# Load the best saved model.
with open(args.model_path, 'rb') as f:
    model = torch.load(f)

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Model total parameters:', total_params)

criterion = nn.CrossEntropyLoss()
    
# Run on test data.
train_loss = evaluate(train_data)
val_loss = evaluate(val_data)
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| Evaluation | train ppl {:8.4f} | val ppl {:8.4f} | test ppl {:8.4f}'.format(
    math.exp(train_loss), math.exp(val_loss), math.exp(test_loss)))
print('=' * 89)