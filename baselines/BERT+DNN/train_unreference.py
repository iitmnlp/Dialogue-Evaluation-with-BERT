import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import argparse
import random
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm

from unreference_score import *
from utils import *
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--seed', type=int, default=123, help='seed for random init')
parser.add_argument('--batch-size', type=int, default=64, help='train and eval batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay ratio')
parser.add_argument("--mode", type=str, help="Type of negative response to use", choices=['random', 'adversarial', 'both'])
parser.add_argument('--exp-dir', type=str, default='./experiments', help='path to the experiment dir')
parser.add_argument("--test-only", action='store_true', help='whether to only run test or both train and test')

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

        
def train(data_iter, net, optimizer, grad_clip=10):
    net.train()
    batch_num, losses = 0, 0
    criterion = nn.BCELoss()
    
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, label = batch
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        label = torch.from_numpy(label).float()
        batch_size = qbatch.shape[0]
        
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            label = label.cuda()
            
        optimizer.zero_grad()
        
        scores = net(qbatch, rbatch)
        loss = criterion(scores, label)
        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        
        losses += loss.item()
        batch_num = batch_idx + 1
    return round(losses / batch_num, 4)

def validation(data_iter, net):
    net.eval()
    losses, batch_num, acc, acc_num = 0, 0, 0, 0
    criterion = nn.BCELoss()
    
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, label = batch 
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        label = torch.from_numpy(label).float()
        batch_size = qbatch.shape[0]
                
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            label = label.cuda()
            
        scores = net(qbatch, rbatch)
        loss = criterion(scores, label)
        
        s = scores >= 0.5
        acc += torch.sum(s.float() == label).item()
        acc_num += batch_size
        
        batch_num += 1
        losses += loss.item()
        
    return round(losses / batch_num, 4), round(acc / acc_num, 4)

def main(args, trainqpath, trainrpath, trainlpath, devqpath,\
         devrpath, devlpath, testqpath, testrpath, testlpath,\
         weight_decay=1e-4, lr=1e-5):

    net = BERT_RUBER_unrefer(768, dropout=0.1)
    if torch.cuda.is_available():
        net.cuda()

    print('[!] Finish init the model')
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    epoches, grad_clip = 100, 10
    pbar = tqdm(range(1, epoches + 1))
    training_losses, validation_losses, validation_metrices = [], [], []
    min_loss = np.inf
    best_metric = -1
        
    patience = 0
    begin_time = time.time()
    
    for epoch in pbar:
        train_iter = get_batch(trainqpath, trainrpath, trainlpath, args.batch_size, shuffle=True)
        dev_iter = get_batch(devqpath, devrpath, devlpath, args.batch_size)
        
        training_loss = train(train_iter, net, optimizer)
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
            
        state = {'net': net.state_dict(), 
                 'optimizer': optimizer.state_dict(), 
                 'epoch': epoch}
        
        save_path = "{}/Acc_{}_vloss_{}_epoch_{}.pt".format(args.exp_dir, validation_metric, validation_loss, epoch)
        torch.save(state, save_path)
        
        pbar.set_description("loss(train-dev): {}-{}, Acc: {}, patience: {}".format(training_loss, validation_loss, validation_metric, patience))
    pbar.close()
    
    # calculate costing time
    end_time = time.time()
    hour = math.floor((end_time - begin_time) / 3600)
    minute = math.floor(((end_time - begin_time) - 3600 * hour) / 60)
    second = (end_time - begin_time) - hour * 3600 - minute * 60
    print("Cost {}h, {}m, {}s".format(hour, minute, round(second, 2)))
    test(testqpath, testrpath, testlpath)

def test(testqpath, testrpath, testlpath):

    net = BERT_RUBER_unrefer(768, dropout=0.1)
    load_best_model(net, args.exp_dir)

    if torch.cuda.is_available():
        net.cuda()
    
    test_iter = get_batch(testqpath, testrpath, testlpath, args.batch_size)

    net.eval()
    losses, batch_num, acc, acc_num = 0, 0, 0, 0
    criterion = nn.BCELoss()

    score_list=[]
    label_list=[]

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_iter):
            qbatch, rbatch, label = batch 
            qbatch = torch.from_numpy(qbatch)
            rbatch = torch.from_numpy(rbatch)
            label = torch.from_numpy(label).float()
            batch_size = qbatch.shape[0]
                    
            if torch.cuda.is_available():
                qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
                label = label.cuda()
                
            scores = net(qbatch, rbatch)
            
            loss = criterion(scores, label)

            score_list.extend(scores.cpu().numpy().tolist())
            label_list.extend(label.cpu().numpy().tolist())
            
            s = scores >= 0.5
            acc += torch.sum(s.float() == label).item()
            acc_num += batch_size
            
            batch_num += 1
            losses += loss.item()
        
    test_loss = losses / batch_num
    test_acc = acc / acc_num

    print('Acc =', test_acc)
    print('Loss =', test_loss)

    score_list = np.array(score_list)
    label_list = np.array(label_list)

    np.savetxt( args.exp_dir + '/test_' + args.mode +'_scores.txt' ,score_list)
    np.savetxt( args.exp_dir + '/test_' + args.mode +'_labels.txt' ,label_list)

    predicted = (score_list >=0.5).astype(np.int32)
    c_matrix = confusion_matrix(label_list,predicted)
    print ('confusion_matrix = ',c_matrix)

    return 

if __name__ == "__main__":

    # show the parameters
    print('[!] Parameters:')
    print(args)

    if args.test_only:
        test(f'./data/dailydialog-{args.mode}/src-test-id.embed',
         f'./data/dailydialog-{args.mode}/tgt-test-id.embed',
         f'./data/dailydialog-{args.mode}/label-test.txt')

    else:
        main(args, 
            f'./data/dailydialog-{args.mode}/src-train-id.embed',
            f'./data/dailydialog-{args.mode}/tgt-train-id.embed',
            f'./data/dailydialog-{args.mode}/label-train.txt',
            f'./data/dailydialog-{args.mode}/src-dev-id.embed',
            f'./data/dailydialog-{args.mode}/tgt-dev-id.embed',
            f'./data/dailydialog-{args.mode}/label-dev.txt',
            f'./data/dailydialog-{args.mode}/src-test-id.embed',
            f'./data/dailydialog-{args.mode}/tgt-test-id.embed',
            f'./data/dailydialog-{args.mode}/label-test.txt',
            lr=args.lr, weight_decay=args.weight_decay)