import argparse
import time
import math
import numpy as np
np.random.seed(331)
import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='custom',
                    help='FD|ELD|PM for prepared models or anything else for custom settings')
parser.add_argument('--emsize', type=int, default=655,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to the RNN output (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=321,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='not use CUDA')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str,  default='output/',
                    help='dir path to save the log and the final model')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--name', type=str,  default=randomhash,
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--kappa', type=float, default=0,
                    help='kappa penalty for hidden states discrepancy (kappa = 0 means no penalty)')
parser.add_argument('--double_target', action='store_true',
                    help='use target for the auxiliary network as well')
parser.add_argument('--eval_auxiliary', action='store_true',
                    help='forward auxiliary network in evaluation mode (without dropout)')
parser.add_argument('--same_mask_e', action='store_true',
                    help='use the same dropout mask for removing words from embedding layer in both networks')
parser.add_argument('--same_mask_i', action='store_true',
                    help='use the same dropout mask for input embedding layers in both networks')
parser.add_argument('--same_mask_w', action='store_true',
                    help='use the same dropout mask for the RNN hidden to hidden matrix in both networks')
parser.add_argument('--same_mask_o', action='store_true',
                    help='use the same dropout mask for the RNN output in both networks')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
args = parser.parse_args()

if args.model == 'FD':
    args.double_target = True
    args.eval_auxiliary = False
    if args.kappa <= 0:
        args.kappa = 0.15
elif args.model == 'ELD':
    args.double_target = False
    args.eval_auxiliary = True
    if args.kappa <= 0:
        args.kappa = 0.25
elif args.model == 'PM':
    args.double_target = False
    args.eval_auxiliary = False
    if args.kappa <= 0:
        args.kappa = 0.15
else:
    print("Warning! Custom model is used, you may want to try FD|ELD|PM options before")

args.model_file_name = '/' + args.name + '.pt'
args.log_file_name = '/' + args.name + '.log'

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run without --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(ntokens, args.emsize, args.dropout, args.dropouti, args.dropoute, args.wdrop)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

string_args = ''
for name in sorted(vars(args)):
    string_args += name + '=' + str(getattr(args, name)) + ', '
string_args += 'total_params=' + str(total_params)
    
with open(args.save_dir + args.log_file_name, 'a') as f:
    f.write(string_args + '\n')
    f.write('epoch time training_running_ppl validation_pll pat lr\n')

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
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


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    train_running_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if np.random.random() < 0.01:
            hidden = model.init_hidden(args.batch_size)
        else:
            hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        model.train()
        output, new_hidden, rnn_h, dropped_rnn_h = model(data, hidden, return_h=True)
        raw_loss = criterion(output.view(-1, ntokens), targets)
        
        total_loss += raw_loss.data
        train_running_loss += raw_loss.data

        loss = raw_loss
        
        # Kappa penalty
        if args.kappa > 0:
            dm_e = not args.same_mask_e
            dm_i = not args.same_mask_i
            dm_w = not args.same_mask_w
            dm_o = not args.same_mask_o

            if args.eval_auxiliary:
                model.eval()

            kappa_output, _, _, _ = model(data, hidden, return_h=True,
                                          draw_mask_e=dm_e, draw_mask_i=dm_i, draw_mask_w=dm_w, draw_mask_o=dm_o)
            
            if args.double_target:
                loss = loss + criterion(kappa_output.view(-1, ntokens), targets)
                loss = loss/2
            
            l2_kappa = (output - kappa_output).pow(2).mean()
            loss = loss + args.kappa * l2_kappa
        
        # Activiation Regularization
        l2_alpha = dropped_rnn_h.pow(2).mean()
        loss = loss + args.alpha * l2_alpha
        
        # Temporal Activation Regularization (slowness)
        loss = loss + args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()

        loss.backward()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        hidden = new_hidden

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)),
                flush=True)
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    return train_running_loss[0] / batch

# Loop over epochs.
lr = args.lr
best_val_loss = None
patience = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_running_loss = train()
        val_loss = evaluate(val_data)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save_dir + args.model_file_name, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if patience > 20:
                patience = 0
                best_val_loss = None
                lr /= 3
                if lr < 0.1:
                    print('Learning rate is too small to continue. This is the end.')
                    break
                with open(args.save_dir + args.model_file_name, 'rb') as f:
                    model = torch.load(f)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wdecay)
        print('-' * 89)
        print('end of epoch {:3d} | time: {:5.2f}s | t loss {:3.2f} | '
                't ppl {:5.2f} | v loss {:3.2f} | v ppl {:5.2f} | pat {:2d}'
                .format(epoch, (time.time() - epoch_start_time),
                train_running_loss, math.exp(train_running_loss),
                val_loss, math.exp(val_loss), patience), flush=True)
        print('-' * 89)
        with open(args.save_dir + args.log_file_name, 'a') as f:
            f.write(str(epoch) + ' ' + str(time.time() - epoch_start_time) + ' ' +
            str(math.exp(train_running_loss)) + ' ' + str(math.exp(val_loss)) + ' ' +
            str(patience) + ' ' + str(lr) + '\n')

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save_dir + args.model_file_name, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

with open(args.save_dir + args.log_file_name, 'a') as f:
    f.write(str(0) + ' ' + str(0) + ' ' +
    str(math.exp(test_loss)) + ' ' + str(math.exp(test_loss)) + ' ' +
    str(-1) + ' ' + str(-1) + '\n')
