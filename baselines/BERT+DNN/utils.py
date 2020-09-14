import pickle
import torch
import numpy as np
import os
import re
from transformers import BertModel, BertTokenizer
import torch
import argparse
import sys
import codecs
import random
from glob import glob
import json
import os

def load_best_model(net, path):
    
    best_acc, best_file = -1, None
    best_epoch = -1
    for file in os.listdir(path):
        try:
            if file[-3:] == '.pt':
                _, acc, _, loss, _, epoch = file.split("_")
                epoch = epoch.split('.')[0]
            else:
                continue
        except:
            continue
        acc = float(acc)
        epoch = int(epoch)
        if acc > best_acc:
            best_file = file
            best_acc = acc

    if best_file:
        file_path = os.path.join(path,best_file)
        print(f'[!] Load the model from {file_path}')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception(f"[!] No saved model")

def get_batch(qpath, rpath, lpath, batch_size, shuffle=False):
    # bert embedding matrix, [dataset_size, 768]
    # return batch shape: [B, 768]
    with open(qpath, 'rb') as f:
        qdataset = pickle.load(f)
    
    with open(rpath, 'rb') as f:
        rdataset = pickle.load(f)

    labels=[]
    for label in open(lpath, 'r').readlines():
        labels.append(float(label.rstrip()))
    labels=np.array(labels) 

    size = len(qdataset)

    if shuffle:
        pureidx = np.arange(size)
        np.random.shuffle(pureidx)

        qdataset = qdataset[pureidx]
        rdataset = rdataset[pureidx]
        labels = labels[pureidx]

    size = len(qdataset)
    idx = 0

    while True:
        qbatch = qdataset[idx:idx+batch_size]
        rbatch = rdataset[idx:idx+batch_size]
        label = labels[idx:idx+batch_size]
        idx += batch_size
        yield qbatch, rbatch, label
        
        if idx > size:
            break
    return None


def preprocess(lines):
    lines = ' <s> '.join(lines)
    return lines.strip()

def create_data(raw_data, src_fname, tgt_fname, label_fname, mode='random'):

    f = open(raw_data, 'r', encoding='utf-8')
    src_f = open(src_fname, 'w', encoding='utf-8')
    tgt_f = open(tgt_fname, 'w', encoding='utf-8')
    label_f = open(label_fname, 'w', encoding='utf-8')

    for idx,line in enumerate(f.readlines()):
        
        df = json.loads(line.strip())        
        context = preprocess(df['context'])
        positive_responses = [preprocess([resp]) for resp in df['positive_responses']]

        if mode == 'random' or mode == 'both':
            for positive_response in positive_responses:
                src_f.write(context+'\n')
                tgt_f.write(positive_response+'\n')
                label_f.write('1\n')
            
            for random_negative_response in df['random_negative_responses']:
                random_negative_response = preprocess([random_negative_response])
                src_f.write(context+'\n')
                tgt_f.write(random_negative_response+'\n')
                label_f.write('0\n')

        if mode == 'adversarial' or mode == 'both':
            for positive_response in positive_responses:
                src_f.write(context+'\n')
                tgt_f.write(positive_response+'\n')
                label_f.write('1\n')
            
            for adversarial_negative_response in df['adversarial_negative_responses']:
                adversarial_negative_response = preprocess([adversarial_negative_response])
                src_f.write(context+'\n')
                tgt_f.write(adversarial_negative_response+'\n')
                label_f.write('0\n')

    f.close()
    src_f.close()
    tgt_f.close()
    label_f.close()

def process_file(path, embed_path, batch_size=128, max_len=128):

    # batch_size: batch for bert to feedforward
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    input_ids_list = []
    lengths_list = []

    i=0
    with codecs.open(path, encoding='utf-8') as f:
        for line in f.readlines():
            # if i%5 == 0:
            data = torch.tensor(tokenizer.encode(line.strip(), add_special_tokens=True))
            if data.shape[0] > max_len:
                data = data[-max_len:]
            
            lengths_list.append(data.shape[0])
            input_ids_list.append(data)    
            i+=1

    data = list(zip(input_ids_list, lengths_list))
    data = sorted(data, key=lambda x: x[1])
    input_ids_list, lengths_list = zip(*data)
    num_data = len(input_ids_list)

    embed = []
    idx = 0
    print ('done loading data')

    with torch.no_grad():
        for idx in range(0, num_data, batch_size):
            input_ids = input_ids_list[idx:idx+batch_size]
            lengths = lengths_list[idx:idx+batch_size]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
            output_embedings = model(input_ids)[0]  #(B, L_max, H)

            lengths = torch.tensor(lengths).unsqueeze(1) #(B, 1)
            mask = torch.arange(output_embedings.shape[1]).unsqueeze(0) #(1, lmax)
            mask = (mask < lengths)
            mask = mask.unsqueeze(2).to(lengths).float()   #(B, lmax, 1)

            output_embedings = mask * (output_embedings) + (1-mask)*(-1e6)
            output_embedings = torch.max(output_embedings, 1)[0].cpu().numpy()

            embed.append(output_embedings)
            
            print(f'{path}: {idx} / {len(input_ids_list)}', end='\r')
            sys.stdout.flush()

    embed = np.concatenate(embed)
    print(f'embed shape: {embed.shape}')
    with open(embed_path, 'wb') as f:
        pickle.dump(embed, f)
        
    print(f'Done Writing the Bert embeddings into {embed_path}!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RUBER utils script')
    parser.add_argument("--train-data", type=str, help="Path to the train file", default=None)
    parser.add_argument("--test-data", type=str, help="Path to the test file", default=None)
    parser.add_argument("--dev-data", type=str, help="Path to the dev file", default=None)
    parser.add_argument("--mode", type=str, help="Type of negative response to use", default='random', choices=['random', 'adversarial', 'both'])    
    parser.add_argument('--batch-size', type=int, default=64, help='bert batch size')

    args = parser.parse_args()

    folder = './data/dailydialog-{}'.format(args.mode)

    if not os.path.exists(folder):
        os.makedirs(folder)

    create_data(args.train_data, folder + '/src-train.txt', folder + '/tgt-train.txt', folder + '/label-train.txt', mode=args.mode)
    create_data(args.dev_data, folder + '/src-dev.txt', folder + '/tgt-dev.txt', folder + '/label-dev.txt', mode=args.mode)
    create_data(args.test_data, folder + '/src-test.txt', folder + '/tgt-test.txt', folder + '/label-test.txt', mode=args.mode)

    for split in ['train', 'dev', 'test']:
        process_file(folder + '/src-{}.txt'.format(split), folder + '/src-{}-id.embed'.format(split), batch_size=args.batch_size)
        process_file(folder + '/tgt-{}.txt'.format(split), folder + '/tgt-{}-id.embed'.format(split), max_len=30, batch_size=args.batch_size)
    