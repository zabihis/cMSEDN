import dgl
import utils.config 
import importlib
import torch as t
from models.classifier import *
from data.seqProcessing import *
importlib.reload(utils.config)
params = utils.config.config()


params.node_embedding_method = "cMSEDN"     # Options: "sMSEDN", "cMSEDN" , "onehot",
                                                        # "DNABERT2"  "NTv2", "word2vec",

params.gnn_model_name = 'GCN'    # Options:'GIN' 'GCN'


# Dimension of node feature
if params.node_embedding_method == "sMSEDN":
    params.d=  params.k *4*4*1
elif params.node_embedding_method=='onehot':
    params.d=4**params.k
elif params.node_embedding_method=='word2vec':
    params.d = 128 
elif params.node_embedding_method == "cMSEDN":
    params.d = 4 * params.k * 4
elif params.node_embedding_method == "DNABERT2":
    params.d = 256 
    params.llmDim=768
elif params.node_embedding_method == "NTv2":
    params.d = 256 
    params.llmDim=512
else:
    raise ValueError(f"{params.node_embedding_method} embedding method is not defined!!") 

if params.gnn_model_name not in ['GIN', 'GCN']:
    raise ValueError(f"{params.node_embedding_method} model is not defined!!") 

trainsetPath = f'data/mouse_tf3/train.csv'
valsetPath = f'data/mouse_tf3/dev.csv'
testsetPath = f'data/mouse_tf3/test.csv'
weightPath = f"checkpoints/dglmodel/model_{params.gnn_model_name}_{params.node_embedding_method}_mouse_tf3_4mer.pkl"

params.savePath = f"checkpoints/dglmodel/model_{params.gnn_model_name}_"+\
                        f"{params.node_embedding_method}_mouse_tf3_{params.k}mer/"
params.graph_cache_dir = f'checkpoints/dglgraph/k{params.k}_d{params.d}_emb{params.node_embedding_method}'
params.seed=45

print(f"--- model selection: {params.gnn_model_name} ---")
print(f"--- Node embedding: {params.node_embedding_method} ---")


#----prepare data-----
print('\n========= Processing train_dataset... =========')
train_dataset = seqProcessing(raw_dir=trainsetPath, save_dir=f'checkpoints/dglgraph/k{params.k}_d{params.d}',\
                               force_reload=True, shared_lab2id=None, params=params )
print('\n========= Processing test_dataset... =========')
val_dataset = seqProcessing(raw_dir=valsetPath, save_dir=f'checkpoints/dglgraph/k{params.k}_d{params.d}',\
                             force_reload=True, shared_lab2id=train_dataset.lab2id, params=params)
print('\n========= Processing test_dataset... =========')
test_dataset = seqProcessing(raw_dir=testsetPath, save_dir=f'checkpoints/dglgraph/k{params.k}_d{params.d}',\
                              force_reload=True, shared_lab2id=train_dataset.lab2id, params=params)

#----init model-----
if params.gnn_model_name=="GCN":
    from models.classifier import *
    model = GCN_model(in_dim=params.d, hidden_dim=params.hidden_dim, \
                            n_classes=params.n_classes, device=params.device)
elif params.gnn_model_name=="GIN":
    from models.classifier import *
    model = GIN_model(in_dim=params.d, hidden_dim=params.hidden_dim, n_classes=params.n_classes,
                            num_layers=3, dropout=0.2, device=params.device)


train_loader = GraphDataLoader(train_dataset, batch_size=params.batchSize, shuffle=True, drop_last=True )
val_loader  = GraphDataLoader(val_dataset, batch_size=params.batchSize, shuffle=False)
test_loader  = GraphDataLoader(test_dataset, batch_size=params.batchSize, shuffle=False)


best = model.test_model(test_loader,
            weightPath=weightPath,
            device=params.device,
            params=params)


print(f'method: {params.gnn_model_name} + {params.node_embedding_method}_d{params.d}_{params.k}mer')

