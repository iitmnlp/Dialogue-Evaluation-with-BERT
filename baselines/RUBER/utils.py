#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import argparse
import os
import sys
import re
import time
import csv
import math
import random
import pickle
import json
from nltk import word_tokenize


def tokenizer(iterator):
    for value in iterator:
        yield value.split()
        

def load_embedding(VOCAB, path, embedding_dim=300):
    with open(path) as f:
        weights = np.random.rand(VOCAB.get_vocab_size(), embedding_dim)
        counter = 0
        for line in f.readlines():
            try:
                line = line.strip().split()
                v = list(map(float, line[1:]))
                word = line[0]
                wid = VOCAB.get_index(word)
                if wid != VOCAB.get_index("<unk>"):
                    counter += 1
                    weights[wid] = np.array(v)
            except Exception as e:
                print(e)
                ipdb.set_trace()
        print(f"[!] Loading the weights {round(counter / VOCAB.get_vocab_size(), 4)}")
    return weights

    
def load_best_model(net, path, load_file=None, map_location='cuda'):

    if load_file is None:

        best_acc, load_file = -1, None
        best_epoch = -1
        
        for fname in os.listdir(path):
            try:
                _, acc, _, loss, _, epoch = fname.split("_")
                epoch = epoch.split('.')[0]
            except:
                continue
            acc = float(acc)
            epoch = int(epoch)
            if acc > best_acc:
                load_file = os.path.join(path, fname)
                best_acc = acc

    if load_file:
        print(f'[!] Load the model from {load_file}',map_location)
        net.load_state_dict(torch.load(load_file, map_location=map_location)['net'])
    else:
        print ('No saved model found')
        
def load_special_model(net, path):
    try:
        net.load_state_dict(torch.load(path)['net'])
    except:
        raise Exception(f"[!] {path} load error")
        
        
class Vocab():
    
    '''
    The vocab instance for the dataset (volumn)
    '''
    
    def __init__(self, special_tokens, lower=True):
        self.special_tokens = special_tokens.copy()
        self.freq = {}
        self.lower = lower
     
    def add_token(self, token):
        token = token.lower()
        if token not in self.freq:
            self.freq[token] = 1
        else:
            self.freq[token] += 1
            
    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)
            
    def get_vocab_size(self):
        try:
            return len(self.itos)
        except:
            raise Exception("Not init, try to call .squeeze func")
            
    def get_index(self, token):
        if token in self.stoi:
            return self.stoi[token]
        else:
            return self.stoi["<unk>"]
            
    def get_token(self, index):
        if index > len(self.itos):
            raise Exception("Bigger than vocab size")
        else:
            return self.itos[index]
        
    def squeeze(self, threshold=1, max_size=None, debug=False):
        '''
        threshold for cutoff
        max_size for constraint the size of the vocab
        threshold first, max_size next
        
        this function must be called to get tokens list
        '''
        words = list(self.freq.items())
        words.sort(key=lambda x: x[1])
        new_words = []
        for word in words:
            if word[1] >= threshold:
                new_words.append(word)
        words = list(reversed(new_words))
        if max_size and len(words) > max_size:
            words = words[:max_size]
        self.itos = [word for word, freq in words]
        
        # add the special tokens
        if self.special_tokens:
            self.itos.extend(self.special_tokens)
        
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}
        
        del self.freq
        if debug:
            print(f"Vocab size: {len(self.stoi)}")


def get_batch(qpath, rpath, lpath, batch_size, seed=100, shuffle=True):
    
    np.random.seed(seed)
    random.seed(seed)

    with open(qpath, 'rb') as f:
        qlen, qdataset = pickle.load(f)
    f.close()
        
    with open(rpath, 'rb') as f:
        rlen, rdataset = pickle.load(f)
    f.close()
    
    labels=[]
    for label in open(lpath, 'r').readlines():
        labels.append(float(label.rstrip()))
    labels=np.array(labels)

    size = len(qdataset)    # dataset size
    idx = 0 

    if shuffle:
        pureidx = np.arange(size)
        np.random.shuffle(pureidx)

        qlen = qlen[pureidx]
        qdataset = qdataset[pureidx]
        rlen = rlen[pureidx]
        rdataset = rdataset[pureidx]
        labels = labels[pureidx]
        
    while True:

        qbatch = qdataset[idx:idx+batch_size]
        qll = qlen[idx:idx+batch_size]
        rbatch = rdataset[idx:idx+batch_size]
        rll = rlen[idx:idx+batch_size]
        label = labels[idx:idx+batch_size]
        
        idx += batch_size
        yield qbatch, rbatch, qll, rll, label

        if idx > size:
            return None
    return None

        
def make_embedding_matrix(fname, word2vec, vec_dim, fvocab):
    if os.path.exists(fname):
        print('Loading embedding matrix from %s'%fname)
        return pickle.load(open(fname, 'rb'))

    with open(fvocab, 'rb') as f:
        vocab = pickle.load(f)
    print('Saving embedding matrix in %s'%fname)

    matrix=[]
    vocab_size = vocab.get_vocab_size()
    
    for i in range(vocab_size):
        v = vocab.itos[i]
        if v in word2vec:
            vec = word2vec[v] 
        else:
            print (f'word {v} not found in word2vec')
            vec = [0.0 for _ in range(vec_dim)]
        matrix.append(vec)    
    pickle.dump(matrix, open(fname, 'wb'), protocol=2)
    return matrix

def load_word2vec(fword2vec):
    """
    Return:
        word2vec dict
        vector dimension
        dict size
    """
    print('Loading word2vec dict from %s'%fword2vec)
    vecs = {}
    vec_dim=0
    with open(fword2vec) as fin:
        # size, vec_dim = list(map(int, fin.readline().split()))
        vec_dim = 300
        size = 0
        for line in fin:
            ps = line.rstrip().split()
            try:
                vecs[ps[0]] = list(map(float, ps[1:]))
                size += 1
            except:
                pass
    return vecs, vec_dim, size


def process_file(path, vocabpath, idpath, max_length=20, create_vocab=False, vocab_size=30000):
    
    # create the vocab instance for models
    # create the id file for training dataset
    
    if create_vocab:
        vocab = Vocab(['<unk>', "<pad>", "<sos>", "<eos>"])
        with open(path) as f:
            for line in f.readlines():
                words = line.strip().split()
                vocab.add_tokens(words)
        vocab.squeeze(max_size=vocab_size, debug=True)
        
        # save the vocab into file
        with open(vocabpath, 'wb') as f:
            pickle.dump(vocab, f)
    else:
        with open(vocabpath, 'rb') as f:
            vocab = pickle.load(f)
        
    # creat the dataset

    with open(path) as f:
        dataset = []
        ll = []
        for line in f.readlines():
            words = line.strip().split()[:max_length]
            words = ["<sos>"] + words + ["<eos>"]
            length = len(words)
            if len(words) < max_length + 2:
                words.extend(['<pad>'] * (max_length + 2 - len(words)))
            dataset.append(np.array([vocab.get_index(word) for word in words]))
            ll.append(length)
        dataset = np.stack(dataset)    # [B, Max_length]
        length  = np.array(ll)
    
    # save the id training dataset into the file
    with open(idpath, 'wb') as f:
        pickle.dump((length, dataset), f, protocol=4)         

def preprocess(lines):
    lines = [' '.join(word_tokenize(line)).lower() for line in lines]
    lines = ' <s> '.join(lines)
    return lines.strip()


def create_data(raw_data, src_fname, tgt_fname, label_fname, mode='random'):

    f = open(raw_data, 'r', encoding='utf-8')
    src_f = open(src_fname, 'w', encoding='utf-8')
    tgt_f = open(tgt_fname, 'w', encoding='utf-8')
    label_f = open(label_fname, 'w', encoding='utf-8')

    for line in f.readlines():
        
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RUBER utils script')
    parser.add_argument("--train-data", type=str, help="Path to the train file", default=None)
    parser.add_argument("--test-data", type=str, help="Path to the test file", default=None)
    parser.add_argument("--dev-data", type=str, help="Path to the dev file", default=None)
    parser.add_argument("--mode", type=str, help="Type of negative response to use", default='random', choices=['random', 'adversarial', 'both'])
    args = parser.parse_args()
    
    folder = './data/dailydialog-{}'.format(args.mode)

    if not os.path.exists(folder):
        os.makedirs(folder)

    create_data(args.train_data, folder + '/src-train.txt', folder + '/tgt-train.txt', folder + '/label-train.txt', mode=args.mode)
    create_data(args.dev_data, folder + '/src-dev.txt', folder + '/tgt-dev.txt', folder + '/label-dev.txt', mode=args.mode)
    create_data(args.test_data, folder + '/src-test.txt', folder + '/tgt-test.txt', folder + '/label-test.txt', mode=args.mode)


    for split in ['train', 'dev', 'test']:
        process_file(folder + '/src-{}.txt'.format(split), './data/src-vocab.pkl', folder + '/src-{}-id.pkl'.format(split), max_length=100)
        process_file(folder + '/tgt-{}.txt'.format(split), './data/tgt-vocab.pkl', folder + '/tgt-{}-id.pkl'.format(split), max_length=30)