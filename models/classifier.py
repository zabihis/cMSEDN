import dgl

import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
import numpy as np
import torch as t
import os
from utils.FocalLoss import *
from models.MLP import MLP
import random
from sklearn.metrics import matthews_corrcoef


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False


class BaseClassifier:
    def __init__(self):
        pass

    def valid_epoch(self, dataloader, loss_fn, device, params=None):
        with t.no_grad():
            valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            self.to_eval_mode()
            pred_list = []
            label_list = []
            for batched_graph, labels in dataloader:  # Removed unused '_'
                batched_graph, labels = batched_graph.to(device), labels.to(device)
                feats = batched_graph.ndata['attr']
                output = self.calculate_y(batched_graph, feats)
                output, labels = output.to(t.float32), labels.to(t.int64)
                
                pred_list.extend(output.detach().cpu().numpy())
                label_list.extend(labels.cpu().numpy())
            with t.no_grad():
                pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
                valid_loss = loss_fn(pred_tensor, label_tensor)
            pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()
            valid_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
            valid_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro', zero_division=0)
            valid_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro',
                                        zero_division=0)
            valid_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro', zero_division=0)
            valid_roc = roc_auc_score(F.one_hot(t.tensor(label_list), num_classes=params.n_classes).cpu().numpy(),
                                      pred_array, average='micro')
            valid_mcc = matthews_corrcoef(np.array(label_list), np.argmax(pred_array, axis=1))
        return valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc, valid_mcc


    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()

    def save(self, path, epochs, bestMtc=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc}
        for idx, module in enumerate(self.moduleList):
            stateDict[idx] = module.state_dict()
        t.save(stateDict, path)

    def load(self, path, map_location=None):
        parameters = t.load(path, map_location=map_location)
        for idx, module in enumerate(self.moduleList):
            module.load_state_dict(parameters[idx])
        print("%d epochs and %.3lf val Score 's model load finished." % (parameters['epochs'], parameters['bestMtc']))


    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()


    def test_model(self, test_loader,
                weightPath='',
                device=t.device('cpu'),
                params=None): 
        if os.path.exists(weightPath):
            self.load(weightPath)
            print("Weight loaded!")
        else:
            raise ValueError(f"Weight not found: {weightPath}!")

        lossfn = FocalLoss(gamma=2)

        test_loss, test_acc, test_f, test_pre, test_rec, test_roc, test_mcc = self.valid_epoch(
                test_loader, lossfn, device, params=params)
        
        print("\n=============== Experiment Result ===============")
        print(f"{'Split':<8}\t{'Acc':<12}\t{'F1':<8}\t{'MCC':<8}\t{'ROC':<8}")
        print(f"{'Test':<8}\t{test_acc:.3f}\t\t{test_f:.3f}\t\t{test_mcc:.3f}\t\t{test_roc:.3f}")
        return 0



class GCN_model(BaseClassifier):
    def __init__(self, in_dim, hidden_dim, n_classes, device=t.device("cpu")):
        super(GCN_model, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        self.classify = MLP(inSize = hidden_dim, outSize = n_classes).to(device)
        self.moduleList = nn.ModuleList([self.conv1, self.conv2, self.classify])
        self.device = device

    def calculate_y(self, g, h):
        h = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
        h = F.relu(self.conv2(g, h, edge_weight=g.edata['weight']))
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)
        
        

class GIN_model(BaseClassifier):
    def __init__(self, in_dim, hidden_dim, n_classes, num_layers=3, dropout=0.2,
                 device=t.device("cpu")):
        super(GIN_model, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.device = device

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(num_layers):
            in_size = in_dim if i == 0 else hidden_dim
            gin_mlp = nn.Sequential(
                nn.Linear(in_size, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(dglnn.GINConv(gin_mlp, aggregator_type='sum'))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(p=dropout))

        self.readout_dim = hidden_dim * 3

        self.classify = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

        self.moduleList = nn.ModuleList(list(self.convs) + list(self.bns) + list(self.dropouts) + [self.classify])

    def calculate_y(self, g, h):
        for i, conv in enumerate(self.convs):
            h_new = conv(g, h, edge_weight=g.edata['weight'])
            h_new = self.bns[i](h_new)
            h_new = F.relu(h_new)
            h_new = self.dropouts[i](h_new)
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        with g.local_scope():
            g.ndata['h'] = h
            mean_pool = dgl.mean_nodes(g, 'h')
            max_pool = dgl.max_nodes(g, 'h')
            sum_pool = dgl.sum_nodes(g, 'h')
            hg = t.cat([mean_pool, max_pool, sum_pool], dim=1)

        return self.classify(hg)