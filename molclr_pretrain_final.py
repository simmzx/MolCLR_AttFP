"""
MolCLR Pretraining with AttentiveFP - FINAL STABLE VERSION
===========================================================
Fully compatible with DGL 1.x/2.x API
Fixed all known issues:
1. DGL API compatibility (copy_edge -> copy_e, etc.)
2. NT-Xent loss computation
3. Memory management
"""

import os
import sys
import random
import argparse
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

import dgl
from dgl.nn.pytorch import edge_softmax

# Suppress logging
import logging
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import deepchem as dc

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    'node_feat_size': 30,
    'edge_feat_size': 11,
    'num_layers': 4,
    'graph_feat_size': 400,
    'dropout': 0.3,
    'proj_dim': 256,
    'temperature': 0.1,
    'epochs': 20,
    'batch_size': 256,
    'lr': 0.0005,
    'weight_decay': 1e-5,
    'warm_up': 2,
    'mask_rate': 0.25,
    'num_workers': 4,
    'valid_size': 0.01,
    'seed': 731,
}


# =============================================================================
# AttentiveFP Backbone - DGL 1.x/2.x Compatible
# =============================================================================
class AttentiveGRU1(nn.Module):
    """First AttentiveGRU layer with edge features."""
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super().__init__()
        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        # Compute attention-weighted edge messages
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        # Aggregate messages: use copy_e (DGL 1.x+) or copy_edge (older)
        g.update_all(
            dgl.function.copy_e('e', 'm'),  # copy edge feature to message
            dgl.function.sum('m', 'c')       # sum messages
        )
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    """Second AttentiveGRU layer without edge features."""
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super().__init__()
        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)
        # Message passing: source node feature * edge attention
        g.update_all(
            dgl.function.u_mul_e('hv', 'a', 'm'),  # src_mul_edge in old API
            dgl.function.sum('m', 'c')
        )
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    """Get initial context for AttentiveFP."""
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super().__init__()
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
        self.attentive_gru = AttentiveGRU1(
            graph_feat_size, graph_feat_size, graph_feat_size, dropout
        )

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats
        
        # Compute edge features from source nodes and edges
        g.apply_edges(
            lambda edges: {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}
        )
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        
        # Compute attention logits
        g.apply_edges(
            lambda edges: {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}
        )
        logits = self.project_edge2(g.edata['he2'])
        
        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    """Single GNN layer for AttentiveFP."""
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super().__init__()
        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(
            lambda edges: {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}
        )
        logits = self.project_edge(g.edata['he'])
        return self.attentive_gru(g, logits, node_feats)


class AttentiveFPGNN(nn.Module):
    """AttentiveFP GNN encoder."""
    def __init__(self, node_feat_size, edge_feat_size, num_layers=2, 
                 graph_feat_size=200, dropout=0.):
        super().__init__()
        self.init_context = GetContext(
            node_feat_size, edge_feat_size, graph_feat_size, dropout
        )
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats


class GlobalPool(nn.Module):
    """Global attention pooling."""
    def __init__(self, feat_size, dropout):
        super().__init__()
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
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1)
            )
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)
            g_repr = dgl.sum_nodes(g, 'hv', 'a')
            return self.gru(F.elu(g_repr), g_feats)


class AttentiveFPReadout(nn.Module):
    """AttentiveFP readout layer."""
    def __init__(self, feat_size, num_timesteps=1, dropout=0.):
        super().__init__()
        self.readouts = nn.ModuleList([
            GlobalPool(feat_size, dropout) for _ in range(num_timesteps)
        ])

    def forward(self, g, node_feats):
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')
        for readout in self.readouts:
            g_feats = readout(g, node_feats, g_feats)
        return g_feats


class MolCLRAttentiveFP(nn.Module):
    """MolCLR model with AttentiveFP backbone."""
    def __init__(self, node_feat_size, edge_feat_size, num_layers, 
                 graph_feat_size, proj_dim, dropout):
        super().__init__()
        self.gnn = AttentiveFPGNN(
            node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout
        )
        self.readout = AttentiveFPReadout(graph_feat_size, num_timesteps=1, dropout=dropout)
        self.projection_head = nn.Sequential(
            nn.Linear(graph_feat_size, graph_feat_size),
            nn.ReLU(inplace=True),
            nn.Linear(graph_feat_size, proj_dim)
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        h = self.readout(g, node_feats)
        z = self.projection_head(h)
        return h, z

    def get_gnn_state_dict(self):
        return self.gnn.state_dict()


# =============================================================================
# NT-Xent Loss
# =============================================================================
class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        device = z_i.device
        
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ])
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# =============================================================================
# Dataset
# =============================================================================
class MoleculeDatasetCLR(Dataset):
    """Dataset for contrastive learning."""
    def __init__(self, smiles_list, mask_rate=0.25):
        self.smiles_list = smiles_list
        self.mask_rate = mask_rate
        self.featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        
    def __len__(self):
        return len(self.smiles_list)
    
    def augment_graph(self, g):
        g = g.clone()
        num_nodes = g.num_nodes()
        num_mask = max(1, int(self.mask_rate * num_nodes))
        perm = torch.randperm(num_nodes)
        mask_idx = perm[:num_mask]
        node_feats = g.ndata['x'].clone()
        node_feats[mask_idx] = 0
        g.ndata['x'] = node_feats
        return g
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        try:
            mol_graph = self.featurizer.featurize([smiles])[0]
            g = mol_graph.to_dgl_graph(self_loop=True)
            g_i = self.augment_graph(g)
            g_j = self.augment_graph(g)
            return g_i, g_j
        except:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None
    g_is, g_js = zip(*batch)
    return dgl.batch(g_is), dgl.batch(g_js)


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch:2d}', ncols=100)
    
    for g_i, g_j in pbar:
        if g_i is None:
            continue
        
        g_i = g_i.to(device)
        g_j = g_j.to(device)
        
        _, z_i = model(g_i, g_i.ndata['x'], g_i.edata['edge_attr'])
        _, z_j = model(g_j, g_j.ndata['x'], g_j.edata['edge_attr'])
        
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        loss = criterion(z_i, z_j)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./ckpt_molclr')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=CONFIG['num_workers'])
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'molclr_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    
    print("=" * 70)
    print("MolCLR Pretraining with AttentiveFP (FINAL STABLE VERSION)")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print("=" * 70)
    
    print("\nLoading SMILES...")
    with open(args.data_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(smiles_list)} molecules")
    
    dataset = MoleculeDatasetCLR(smiles_list, mask_rate=CONFIG['mask_rate'])
    
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(CONFIG['valid_size'] * len(dataset))
    train_idx = indices[split:]
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=args.num_workers, 
        collate_fn=collate_fn,
        drop_last=True, 
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    
    model = MolCLRAttentiveFP(
        node_feat_size=CONFIG['node_feat_size'],
        edge_feat_size=CONFIG['edge_feat_size'],
        num_layers=CONFIG['num_layers'],
        graph_feat_size=CONFIG['graph_feat_size'],
        proj_dim=CONFIG['proj_dim'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - CONFIG['warm_up'], eta_min=1e-6)
    criterion = NTXentLoss(temperature=CONFIG['temperature'])
    
    print("\nVerifying loss computation...")
    test_z = torch.randn(32, CONFIG['proj_dim']).to(device)
    test_z = F.normalize(test_z, p=2, dim=1)
    test_loss = criterion(test_z[:16], test_z[16:])
    print(f"Test loss (should be ~4-5): {test_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        print(f"Epoch {epoch:2d} | Loss: {loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch > CONFIG['warm_up']:
            scheduler.step()
        
        if loss < best_loss and loss > 0.01:
            best_loss = loss
            torch.save(model.get_gnn_state_dict(), os.path.join(output_dir, 'gnn_pretrained.pth'))
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            print(f"  ★ Best model saved (loss: {best_loss:.4f})")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch}.pth'))
    
    torch.save(model.get_gnn_state_dict(), os.path.join(output_dir, 'gnn_final.pth'))
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_final.pth'))
    
    print("\n" + "=" * 70)
    print(f"Training completed! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
