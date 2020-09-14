import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from prettytable import PrettyTable

import argparse
import random
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from unreference_score import *
from utils import *
import time
from scipy.stats import pointbiserialr
from sklearn.metrics import accuracy_score
    
    
parser = argparse.ArgumentParser(description='RUBER training script')
parser.add_argument('--seed', type=int, default=123, help='seed for random init')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay ratio')
parser.add_argument('--batch-size', type=int, default=64, help='train and eval batch size')

parser.add_argument('--hidden-size', type=int, default=128, help='GRU hidden state')
parser.add_argument('--save-steps', type=int, default=100, help='number of steps per save')
parser.add_argument('--num-layers', type=int, default=1, help='num of layers in GRU')
parser.add_argument("--mode", type=str, help="Type of negative response to use", choices=['random', 'adversarial', 'both'])
parser.add_argument("--test-only", action='store_true', help='whether to only run test or both train and test')
parser.add_argument('--exp-dir', type=str, default='./experiments', help='path to the experiment dir')
parser.add_argument('--init-checkpoint', type=str, default=None, help='path to the initial checkpoint file, else weights are randomly initialized')


args = parser.parse_args()

# set the random seed for the model
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train(data_iter, net, optimizer, 
          delta=0.5, grad_clip=10, epoch=0):
    net.train()
    batch_num, losses = 0, 0
    # criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    criterion = nn.BCELoss()
    start = time.time() 
    for batch_idx, batch in enumerate((data_iter)):
        qbatch, rbatch, qlength, rlength, label = batch
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        qlength = torch.from_numpy(qlength)
        rlength = torch.from_numpy(rlength)
        label = torch.from_numpy(label).float()
        batch_size = len(qlength)
                
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            qlength, rlength = qlength.cuda(), rlength.cuda()
            label = label.cuda()
            
        qbatch = qbatch.transpose(0, 1)
        rbatch = rbatch.transpose(0, 1)
        optimizer.zero_grad()
        
        scores = net(qbatch, qlength, rbatch, rlength)   
        loss = criterion(scores, label)
        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        
        optimizer.step()
        losses += loss.item()
        total_loss += loss.item()
        batch_num = batch_idx + 1
        
        if (batch_idx+1) % 100== 0:
            total_loss = total_loss/100
            time_taken = time.time() - start
            print(f'Epoch: {epoch}, Iter: {batch_idx+1}, Loss: {total_loss}, Time: {time_taken}s')
            sys.stdout.flush()
            total_loss = 0
            start = time.time()

        if (batch_idx+1) %args.save_steps==0:
            state = {'net': net.state_dict(), 
                     'optimizer': optimizer.state_dict(), 
                     'epoch': epoch}
            steps = batch_idx+1
            torch.save(state,
                   f'{args.exp_dir}/steps_{steps}_epoch_{epoch}.pt')
            print ('Saved model after {} steps'.format(steps))
        
    return round(losses / batch_num, 4)


# validation
def validation(data_iter, net, save_scores=False, delta=0.8):
    ''' 
    calculate the Acc
    '''
    score_list=[]
    label_list=[]

    net.eval()
    losses, batch_num, acc, acc_num = 0, 0, 0, 0
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    for batch_idx, batch in enumerate((data_iter)):
        qbatch, rbatch, qlength, rlength, label = batch
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        qlength = torch.from_numpy(qlength)
        rlength = torch.from_numpy(rlength)
        label = torch.from_numpy(label).float()
        batch_size = len(qlength)
                
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            qlength, rlength = qlength.cuda(), rlength.cuda()
            label = label.cuda()
            
        qbatch = qbatch.transpose(0, 1)
        rbatch = rbatch.transpose(0, 1)
                    
        scores = net(qbatch, qlength, rbatch, rlength)    # [2 * B]
        loss = criterion(scores, label)
        
        score_list.extend(scores.cpu().data.numpy().tolist())
        label_list.extend(label.cpu().data.numpy().tolist())

        s = scores >= 0.5
        acc += torch.sum(s.float() == label).item()
        
        acc_num += batch_size
        
        batch_num += 1
        losses += loss.item()

    score_list = np.array(score_list)
    label_list = np.array(label_list)

    pbc, pval = pointbiserialr(label_list, score_list)
    acc = accuracy_score(label_list, score_list >=0.5)
    print ('PBC: {}, pval: {}'.format(pbc, pval))

    if save_scores:
        np.savetxt( args.exp_dir + '/test_' + args.mode +'_scores.txt' ,score_list)
        np.savetxt( args.exp_dir + '/test_' + args.mode +'_labels.txt' ,label_list)

        predicted = (score_list >=0.5).astype(np.int32)
        c_matrix = confusion_matrix(label_list,predicted)
        print ('confusion_matrix = ',c_matrix)
    
    return round(losses / (batch_num), 4), acc


def test(net, test_data, save_scores=False):

    test_loss, test_acc = validation(test_data, net, save_scores)
    print('Acc =', test_acc)
    print('Loss =', test_loss)
    
    
def main(args, trainqpath, trainrpath, trainlpath, devqpath, devrpath, devlpath,
         testqpath, testrpath, testlpath, weight_decay=1e-4, lr=1e-3):
    
    with open(f'data/src-vocab.pkl', 'rb') as f:
        srcv = pickle.load(f)
        
    with open(f'data/tgt-vocab.pkl', 'rb') as f:
        tgtv = pickle.load(f)

    src_embed = pickle.load(open('./data/src-embed.pkl', 'rb'))
    tgt_embed = pickle.load(open('./data/tgt-embed.pkl', 'rb'))
    
    net = RUBER_unrefer(srcv.get_vocab_size(), tgtv.get_vocab_size(),
        300, args.hidden_size, SOURCE=srcv, TARGET=tgtv, src_embed=src_embed, tgt_embed=tgt_embed, num_layers=args.num_layers)
    
    if torch.cuda.is_available():
        net.cuda()
    
    print('[!] Finish init the vocab and net')
    loc = 'cuda' if torch.cuda.is_available() else 'cpu'    
    load_best_model(net, args.exp_dir, load_file=args.init_checkpoint, map_location=loc)
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    epochs = 100
    grad_clip = 10
    early_stop_patience = 5
    pbar = (range(1, epochs + 1))
    training_losses, validation_losses, validation_metrices = [], [], []
    min_loss = np.inf
    best_metric = -1
        
    patience = 0
    begin_time = time.time()
    idxx = 1
    for epoch in pbar:

        train_iter = get_batch(trainqpath, trainrpath, trainlpath, args.batch_size, seed=args.seed)
        dev_iter = get_batch(devqpath, devrpath, devlpath, args.batch_size, args.seed, shuffle=False)

        training_loss = train(train_iter, net, optimizer,epoch=epoch)
        validation_loss, validation_metric = validation(dev_iter, net)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        validation_metrices.append(validation_metric)
    
        if best_metric < validation_metric:
            patience = 0
            best_metric = validation_metric
            min_loss = validation_loss
        else:
            patience += 1
            
        # Save the model
        state = {'net': net.state_dict(), 
                 'optimizer': optimizer.state_dict(), 
                 'epoch': epoch}
        torch.save(state,
            f'{args.exp_dir}/Acc_{validation_metric}_vloss_{validation_loss}_epoch_{epoch}.pt')
        
        print(f"Epoch: {epoch}, loss(train-dev): {training_loss}-{validation_loss}, Acc: {validation_metric}, patience: {patience}")
        idxx += 1
        sys.stdout.flush()
        
        if patience > early_stop_patience:
            print('[!] early stop')
            break
    
    # calculate costing time
    end_time = time.time()
    hour = math.floor((end_time - begin_time) / 3600)
    minute = math.floor(((end_time - begin_time) - 3600 * hour) / 60)
    second = (end_time - begin_time) - hour * 3600 - minute * 60
    print(f"Cost {hour}h, {minute}m, {round(second, 2)}s")
    
    evaluate(args, testqpath, testrpath, testlpath)


def evaluate(args, testqpath, testrpath, testlpath):

    with open(f'data/src-vocab.pkl', 'rb') as f:
        srcv = pickle.load(f)
        
    with open(f'data/tgt-vocab.pkl', 'rb') as f:
        tgtv = pickle.load(f)

    src_embed = pickle.load(open('./data/src-embed.pkl', 'rb'))
    tgt_embed = pickle.load(open('./data/tgt-embed.pkl', 'rb'))
    
    net = RUBER_unrefer(srcv.get_vocab_size(), tgtv.get_vocab_size(),
        300, args.hidden_size, SOURCE=srcv, TARGET=tgtv, src_embed=src_embed, tgt_embed=tgt_embed, num_layers=args.num_layers)
    
    if torch.cuda.is_available():
        net.cuda()
    print('[!] Finish init the vocab and net')

    test_iter = get_batch(testqpath, testrpath, testlpath, args.batch_size, args.seed, shuffle=False)
    loc = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_best_model(net, args.exp_dir, load_file=None, map_location=loc)
    count_parameters(net)
    test(net, test_iter, save_scores=True)

if __name__ == "__main__":

    if args.test_only:
        evaluate(args, f'./data/dailydialog-{args.mode}/src-test-id.pkl',
        f'./data/dailydialog-{args.mode}/tgt-test-id.pkl',
        f'./data/dailydialog-{args.mode}/label-test.txt')

    else:
        main(args, 
        f'./data/dailydialog-{args.mode}/src-train-id.pkl',
         f'./data/dailydialog-{args.mode}/tgt-train-id.pkl',
         f'./data/dailydialog-{args.mode}/label-train.txt',
         f'./data/dailydialog-{args.mode}/src-dev-id.pkl',
         f'./data/dailydialog-{args.mode}/tgt-dev-id.pkl',
         f'./data/dailydialog-{args.mode}/label-dev.txt',
         f'./data/dailydialog-{args.mode}/src-test-id.pkl',
         f'./data/dailydialog-{args.mode}/tgt-test-id.pkl',
         f'./data/dailydialog-{args.mode}/label-test.txt',
         lr=args.lr, weight_decay=args.weight_decay)