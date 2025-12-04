import torch 
from torch import nn
import numpy as np
from utils.utils import lap_eig, topological_sort

class NodeEmbedding(nn.Module):
    def __init__(self, adj_mx, node_emb_dim, k=16, dropout=0):
        super(NodeEmbedding, self).__init__()
        N, _ = adj_mx.shape
        self.k = k
        self.set_adj(adj_mx)
        self.fc = nn.Linear(in_features=k, out_features=node_emb_dim)

    def forward(self):
        node_embedding = self.fc(self.lap_eigvec)

        return node_embedding
    
    def set_adj(self, adj_mx):
        N, _ = adj_mx.shape
        
        self.adj_mx = adj_mx
        eig_vec, eig_val = lap_eig(adj_mx)

        k = self.k
        if k > N:
            eig_vec = np.concatenate([eig_vec, np.zeros((N, k - N))], dim=-1)
            eig_val = np.concatenate([eig_val, np.zeros(k - N)], dim=-1)
        
        ind = np.abs(eig_val).argsort(axis=0)[::-1][:k]

        eig_vec = eig_vec[:, ind]

        if hasattr(self, 'lap_eigvec'):
            self.lap_eigvec = torch.tensor(eig_vec).float()
        else:
            self.register_buffer('lap_eigvec', torch.tensor(eig_vec).float())


class Node2Token(nn.Module):
    def __init__(self, sample_len, features, node_emb_dim, emb_dim, tim_dim, dropout, use_node_embedding):
        super().__init__()

        in_features = sample_len * features

        self.use_node_embedding = use_node_embedding

        state_features = tim_dim
        if use_node_embedding:
            state_features += node_emb_dim

        # Node feature embedding
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, emb_dim),
        )

        # State embedding (time + node_emb)
        hidden_size = node_emb_dim
        self.state_fc = nn.Sequential(
            nn.Linear(state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x, te, ne):
        # x: (B, N, T*F)   te: (B, T, tim_dim)   ne: (N, node_emb_dim)
        B, N, TF = x.shape

        x = self.fc1(x) 

        state = te[:, -1:, :].repeat(1, N, 1)

        if self.use_node_embedding:
            ne = ne.unsqueeze(0).repeat(B, 1, 1)
            state = torch.concat((state, ne), dim=-1)

        state = self.state_fc(state)

        # Combine
        out = x + state
        out = self.ln(out)

        return out