"""
MolCLR Finetuning Script for SA Prediction
==========================================
Uses pretrained MolCLR weights with AttentiveFP backbone for SA prediction.
Compatible with SynFrag's evaluation pipeline.

This script mirrors synfrag_finetune.py exactly, only loading MolCLR pretrained weights.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import logging
from datetime import datetime

import dgl
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
import torch.nn.functional as F
import deepchem as dc

import rdkit
rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL)

# =============================================================================
# Hyperparameters (IDENTICAL to SynFrag finetune)
# =============================================================================
EPOCHS = 200
BATCH_SIZE = 16
GNN_LR = 0.01
POOL_LR = 0.001
GNN_DECAY = 0
POOL_DECAY = 0.001
LOG_INTERVAL = 2000
PATIENCE = 5

# Model parameters (IDENTICAL to SynFrag)
NODE_FEAT = 30
EDGE_FEAT = 11
NUM_LAYERS = 4
NUM_TIMESTEPS = 1
GRAPH_FEAT = 400
N_TASKS = 1
DROPOUT = 0.5

# =============================================================================
# AttentiveFP Backbone (IDENTICAL to SynFrag and MolCLR pretrain)
# =============================================================================
class AttentiveGRU1(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()
        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()
        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)
        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()
        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats
        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])
        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()
        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])
        return self.attentive_gru(g, logits, node_feats)


class AttentiveFPGNN(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers=2, 
                 graph_feat_size=200, dropout=0.):
        super(AttentiveFPGNN, self).__init__()
        self.init_context = GetContext(node_feat_size, edge_feat_size, 
                                       graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats
    
    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))


class GlobalPool(nn.Module):
    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()
        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )
        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats):
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)
            g_repr = dgl.sum_nodes(g, 'hv', 'a')
            context = F.elu(g_repr)
            return self.gru(context, g_feats)


class AttentiveFPReadout(nn.Module):
    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()
        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats):
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')
        for readout in self.readouts:
            g_feats = readout(g, node_feats, g_feats)
        return g_feats


class AttentiveFPPredictor(nn.Module):
    """AttentiveFP Predictor - IDENTICAL to SynFrag."""
    def __init__(self, node_feat_size, edge_feat_size, num_layers=2,
                 num_timesteps=2, graph_feat_size=200, n_tasks=1, dropout=0.):
        super(AttentiveFPPredictor, self).__init__()
        
        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats)
        return self.predict(g_feats)


# =============================================================================
# Dataset (IDENTICAL to SynFrag)
# =============================================================================
class MoleculeDatasetB:
    def __init__(self, data_file, smiles_field, label_field):
        df = pd.read_csv(data_file)
        self.smiles = df[smiles_field].to_list()
        self.labels = df[label_field].to_list()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]


# =============================================================================
# Utility Functions
# =============================================================================
def get_paths():
    """Get project paths."""
    base_dir = '/local-house/zhangxiang/MolCLR-master'
    return {
        'checkpoints': os.path.join(base_dir, 'ckpt_attentivefp'),
        'logs': os.path.join(base_dir, 'logs_finetune'),
        'data_train': '/local-house/zhangxiang/attrmasking_attentivefp/data/train_dataset/finetune',
        'data_test': '/local-house/zhangxiang/attrmasking_attentivefp/data/test_dataset'
    }


def resolve_path(file_input, default_dir):
    """Resolve file path."""
    if os.path.isabs(file_input) and os.path.exists(file_input):
        return file_input
    if os.path.exists(file_input):
        return os.path.abspath(file_input)
    default_path = os.path.join(default_dir, file_input)
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError(f"File not found: {file_input}")


def create_model(device, model_input, checkpoints_dir):
    """Create model and load pretrained weights."""
    model = AttentiveFPPredictor(
        NODE_FEAT, EDGE_FEAT, NUM_LAYERS, NUM_TIMESTEPS, 
        GRAPH_FEAT, N_TASKS, DROPOUT
    ).to(device)
    
    try:
        model_path = resolve_path(model_input, checkpoints_dir)
        model.gnn.from_pretrained(model_path)
        print(f"Loaded pretrained model: {model_path}")
    except Exception as e:
        print(f"Pretrained model not found: {model_input}, training from scratch")
        print(f"Error: {e}")
    
    return model


def evaluate(model, loader, device, name):
    """Evaluate model on test set."""
    if not loader:
        return 0.0
    
    model.eval()
    y_list, pred_list = [], []
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Testing {name}'):
            try:
                graphs = featurizer.featurize(batch[0])
                dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
                batch_graph = dgl.batch(dgl_graphs).to(device)
                
                pred = model(batch_graph, batch_graph.ndata['x'], batch_graph.edata['edge_attr'])
                y = batch[1].view(pred.shape).to(torch.float).to(device)
                
                y_list.extend(y.flatten().tolist())
                pred_list.extend(pred.detach().cpu().flatten().tolist())
            except Exception as e:
                continue
    
    return roc_auc_score(y_list, pred_list)


def setup_logging(log_dir, experiment_name):
    """Setup logging."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def main():
    parser = argparse.ArgumentParser(description='MolCLR Finetuning for SA')
    parser.add_argument('--input_model_file', required=True, help='Pretrained GNN weights')
    parser.add_argument('--dataset', required=True, help='Training dataset')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--experiment_name', default='molclr', help='Experiment name')
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(731)
    np.random.seed(731)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(731)
    
    paths = get_paths()
    os.makedirs(paths['checkpoints'], exist_ok=True)
    os.makedirs(paths['logs'], exist_ok=True)
    
    log_file = setup_logging(paths['logs'], args.experiment_name)
    
    logging.info("=" * 60)
    logging.info("MolCLR Finetuning for SA Prediction")
    logging.info("=" * 60)
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Model: {args.input_model_file}")
    logging.info(f"Device: {device}")
    
    # Load datasets
    train_path = resolve_path(args.dataset, paths['data_train'])
    ts2_path = os.path.join(paths['data_test'], "TS2.csv")
    ts3_path = os.path.join(paths['data_test'], "TS3.csv")
    
    train_dataset = MoleculeDatasetB(train_path, 'smiles', 'labels')
    test_dataset_ts2 = MoleculeDatasetB(ts2_path, 'smiles', 'labels') if os.path.exists(ts2_path) else None
    test_dataset_ts3 = MoleculeDatasetB(ts3_path, 'smiles', 'labels') if os.path.exists(ts3_path) else None
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    test_loader_ts2 = DataLoader(test_dataset_ts2, BATCH_SIZE, shuffle=False, num_workers=4) if test_dataset_ts2 else None
    test_loader_ts3 = DataLoader(test_dataset_ts3, BATCH_SIZE, shuffle=False, num_workers=4) if test_dataset_ts3 else None
    
    logging.info(f"Train dataset: {len(train_dataset)} | Batches: {len(train_loader)}")
    if test_dataset_ts2:
        logging.info(f"TS2: {len(test_dataset_ts2)}")
    if test_dataset_ts3:
        logging.info(f"TS3: {len(test_dataset_ts3)}")
    
    # Create model
    model = create_model(device, args.input_model_file, paths['checkpoints'])
    
    optimizer = optim.Adam([
        {'params': model.gnn.parameters(), 'lr': GNN_LR, 'weight_decay': GNN_DECAY},
        {'params': model.readout.parameters(), 'lr': POOL_LR, 'weight_decay': POOL_DECAY},
        {'params': model.predict.parameters(), 'lr': POOL_LR, 'weight_decay': POOL_DECAY}
    ])
    loss_fn = nn.MSELoss()
    
    logging.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    best_auroc_ts2 = best_auroc_ts3 = 0
    patience_counter = 0
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    
    # Results file
    results_file = os.path.join(paths['checkpoints'], f'{args.experiment_name}_auroc.txt')
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = num_batches = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            try:
                graphs = featurizer.featurize(batch[0])
                dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
                batch_graph = dgl.batch(dgl_graphs).to(device)
                
                preds = model(batch_graph, batch_graph.ndata['x'], batch_graph.edata['edge_attr'])
                labels = batch[1].view(preds.shape).to(torch.float).to(device)
                loss = loss_fn(preds, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                continue
        
        # Evaluate
        auroc_ts2 = evaluate(model, test_loader_ts2, device, "TS2")
        auroc_ts3 = evaluate(model, test_loader_ts3, device, "TS3")
        
        # Log results
        avg_loss = total_loss / max(num_batches, 1)
        result = f"Epoch {epoch} | Loss: {avg_loss:.4f} | TS2: {auroc_ts2:.4f} | TS3: {auroc_ts3:.4f}"
        logging.info(result)
        
        with open(results_file, 'a') as f:
            f.write(result + '\n')
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(paths['checkpoints'], f'{args.experiment_name}_epoch_{epoch}.pth'))
        
        # Early stopping
        if test_loader_ts2 and test_loader_ts3:
            if auroc_ts2 > best_auroc_ts2 and auroc_ts3 > best_auroc_ts3:
                best_auroc_ts2, best_auroc_ts3 = auroc_ts2, auroc_ts3
                patience_counter = 0
                logging.info(f"New best: TS2={auroc_ts2:.4f}, TS3={auroc_ts3:.4f}")
                torch.save(model.state_dict(), 
                          os.path.join(paths['checkpoints'], f'{args.experiment_name}_best.pth'))
            elif auroc_ts2 < best_auroc_ts2 and auroc_ts3 < best_auroc_ts3:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logging.info("Early stopping!")
                    break
            else:
                patience_counter = 0
    
    logging.info("Finetuning completed!")
    logging.info(f"Best TS2: {best_auroc_ts2:.4f}, Best TS3: {best_auroc_ts3:.4f}")


if __name__ == "__main__":
    main()
