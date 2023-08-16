import os
import argparse
import torch
import torch.nn as nn
import _pickle as cPickle
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time
from data_prep import DictionaryAll,VQAFeatureDatasetAll
import main_model as model
from train import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--int_hid', type=int, default=1024)
    parser.add_argument('--nblock', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--model', type=int, default='model')
    parser.add_argument('--output', type=str, default='result')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dictionary = DictionaryAll.load_from_file('data/dictionary.pkl')
    
    question_type= ['absurd','activity_recognition','attribute','color','counting', 'object_presence','object_recognition','positional_reasoning',
    'scene_recognition','sentiment_understanding','sport_recognition','utility_affordance']

    train_dset = VQAFeatureDatasetAll('train', dictionary)
    eval_dset = VQAFeatureDatasetAll('val', dictionary)
    
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid, args.int_hid, args.nhead, args.nblock, args.batch_size).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=4)
    eval_loader =  DataLoader(eval_dset, args.batch_size, shuffle=True, num_workers=4)

    train(model, train_loader, eval_loader, args.epochs, args.output)
