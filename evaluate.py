import math
import logging
from tqdm import tqdm
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from evaluation.evaluation import eval_edge_prediction
from model.tgnh import TGN
from model.Transformer import *
from utils.utils import EarlyStopMonitor, get_neighbor_finder
from utils.dataset import loadTree, loadUdData, loadFNNData
from utils.rand5fold import load5foldData
from tqdm import tqdm
import os
from torch import nn
from sklearn import metrics
import random
from random import shuffle
from Transformer_utils import *
import matplotlib.pyplot as plt


class hawkes(nn.Module):
    def __init__(self, d_model=768, num_types=1):
        super(hawkes, self).__init__()
        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        pass


class Net(nn.Module):
    def __init__(self, args, device):
        super(Net, self).__init__()
        self.device = device
        self.Transformer = TransformerModel( ninp=768*2, nhead=8, nhid=768, nlayers=4, dropout=0.2)
        self.hawkes_module = hawkes(d_model=768*2, num_types=1)
        self.fc1 = torch.nn.Linear(768*2, 2)  # Linear for forward Dynamic Interaction
        self.act_tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data):
        source_features = torch.from_numpy(data.unique_features[data.sources]).to(self.device)
        destination_features = torch.from_numpy(data.unique_features[data.destinations]).to(self.device)
        updated_embedding = torch.cat((source_features, destination_features), dim=1).to(self.device)

        updated_embedding = updated_embedding.unsqueeze(dim=1).float()#[:128]
        timestamps = data.timestamps# if len(data.timestamps) <=128 else data.timestamps[:128]
        timestamps = torch.Tensor(timestamps).to(self.device)

        updated_embedding = self.Transformer.pos_encoder(updated_embedding, timestamps)
        updated_embedding = self.Transformer.transformer_encoder(updated_embedding)
        # updated_embedding: [seq_length, dim=1, 768]
        #hawkes_data = updated_embedding.permute((1, 0, 2))
        #event_ll, non_event_ll = log_likelihood(model=self.hawkes_module, data=hawkes_data, time=timestamps.unsqueeze(dim=0))
        # event_loss = -torch.sum(event_ll - non_event_ll)
        updated_embedding = updated_embedding.squeeze(dim=1)

        out_feature = updated_embedding[-1]
        out_feature = self.fc1(self.act_tanh(out_feature))
        class_outputs = self.softmax(out_feature.view(1, -1))
        return class_outputs#, event_ll, non_event_ll


torch.manual_seed(0)
np.random.seed(0)
np.set_printoptions(suppress=True)
### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--data', type=str, help='Dataset name',
                    default='Twitter')
parser.add_argument('--bs', type=int, default=10, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tgn-attn-weibo-ma', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=2, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true', default=True,
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function_type', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="TGNF_with_memory", help='Type of message '
                                                                               'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=768, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')

parser.add_argument('--use_destination_embedding_in_message', action='store_false',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_false',
                    help='Whether to use the embedding of the source node as part of the message')
# Beacause I'm not sure which one I used (use_destination_embedding_in_message and use_source_embedding_in_message), I suggest you also can try the following setting.
# parser.add_argument('--use_destination_embedding_in_message', action='store_false',
#                    help='Whether to use the embedding of the destination node as part of the message')
# parser.add_argument('--use_source_embedding_in_message', action='store_false',
#                    help='Whether to use the embedding of the source node as part of the message')


parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--use_gcn', action='store_true',
                    help='Whether to run the GCN model')
parser.add_argument('--opt', type=str, default="RMSprop", choices=[
    "RMSprop", "Adam"], help='Type of optimizer')
parser.add_argument('--fd', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help='fold index')  # 5 denotes all data

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

print("Now using aggregator function is ", args.aggregator)
weight_decay = 1e-4
patience = 10

Path("./saved_models/{}/{}_fold/".format(args.aggregator, args.fd)).mkdir(parents=True, exist_ok=True)

get_model_path = lambda \
        epoch, train_accuracy,test_accuracy: f'./saved_models/{args.aggregator}/{args.fd}_fold/{epoch}-{train_accuracy}-{test_accuracy}.pth'

# Set device
device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

treeDic = loadTree(args.data)
print("len(treeDic)", len(treeDic))

# random.shuffle(treeDic)
fold5_x_train = []  # all data
fold5_x_test = []  # all data

for i in treeDic:
    fold5_x_train.append(i)
    fold5_x_test.append(i)


def to_np(x):
    return x.cpu().detach().numpy()


def evaluate(model, fold_train, fold_test):
    traindata_list, testdata_list = loadFNNData(fold_train, fold_test)
    model.eval()
    num_item = 0
    total_test_loss = 0.0
    avg_test_loss = 0.0
    ok = 0
    correct_id = []
    wrong_id = []
    test_accuracy = 0.000
    test_pred = []
    test_true = []
    for item in tqdm(testdata_list):
        num_item += 1
        index = item.id
        label = np.array([item.labels])
        label = torch.from_numpy(label).to(device)
        #test_ngh_finder = get_neighbor_finder(item, uniform=False)
        #model.tgn.set_neighbor_finder(test_ngh_finder)
        class_outputs = model(item)
        pred = torch.argmax(class_outputs, dim=1)
        if pred[0] == label[0]:
            ok += 1
            correct_id.append(index)
        else:
            wrong_id.append(index)

        if num_item - 1 == 0:
            test_pred = to_np(pred)
            test_true = to_np(label)
        else:
            test_pred = np.concatenate((test_pred, to_np(pred)), axis=0)
            test_true = np.concatenate((test_true, to_np(label)), axis=0)

    test_accuracy = round(ok / num_item, 3)
    print(test_accuracy)
    print(metrics.classification_report(test_true, test_pred, digits=4))
    return correct_id, wrong_id

if __name__ == '__main__':
    # 5-fold cross validation.

    model = Net(args, device)
    model = model.to(device)
    model.load_state_dict(torch.load('saved_models/last/4_fold/49-0.926-0.868.pth'))

    fold0_x_test = np.load('fnn_5_fold_ids/fold0_x_test.npy')
    fold0_x_train = np.load('fnn_5_fold_ids/fold0_x_train.npy')

    fold1_x_test = np.load('fnn_5_fold_ids/fold1_x_test.npy')
    fold1_x_train = np.load('fnn_5_fold_ids/fold1_x_train.npy')

    fold2_x_test = np.load('fnn_5_fold_ids/fold2_x_test.npy')
    fold2_x_train = np.load('fnn_5_fold_ids/fold2_x_train.npy')
    fold3_x_test = np.load('fnn_5_fold_ids/fold3_x_test.npy')
    fold3_x_train = np.load('fnn_5_fold_ids/fold3_x_train.npy')
    fold4_x_train = np.load('fnn_5_fold_ids/fold4_x_train.npy')
    fold4_x_test = np.load('fnn_5_fold_ids/fold4_x_test.npy')
    wrong_id = np.load('wrong_id.npz.npy')
    correct_id, wrong_id = evaluate(model, fold4_x_train, fold4_x_test)
    #np.save('wrong_id.npz', wrong_id)


    





