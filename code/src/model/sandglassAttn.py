import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.position import PositionalEncoding

class SpatialEncoder(nn.Module):
    def __init__(self, sag_dim, sag_tokens, emb_dim, dropout):
        super().__init__()
        self.sag_tokens = sag_tokens
        self.sag_dim = sag_dim
        
        # Learnable latent queries (replacing shared hyper_nodes)
        self.latent_queries = nn.Parameter(torch.randn(1, sag_tokens, sag_dim))
        
        self.pe = PositionalEncoding(num_hiddens=sag_dim, dropout=dropout, max_len=1024)
        
        self.x_fc = nn.Linear(in_features=emb_dim, out_features=sag_dim)
        
        # Attention: N -> K
        self.mha = nn.MultiheadAttention(embed_dim=sag_dim, num_heads=4, batch_first=True, dropout=dropout)
        
        self.out_fc = nn.Linear(in_features=sag_dim, out_features=emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: (B, N, emb_dim)
        B, N, _ = x.shape
        
        kv = self.x_fc(x) # (B, N, sag_dim)
        
        q = self.pe(self.latent_queries).repeat(B, 1, 1) # (B, K, sag_dim)

        out, attn_weights = self.mha(query=q, key=self.pe(kv), value=kv) # (B, K, sag_dim)
        
        out = self.out_fc(out)
        out = self.norm(out)
        
        return out, attn_weights

class SpatialDecoder(nn.Module):
    def __init__(self, sag_dim, emb_dim, dropout):
        super().__init__()
        self.sag_dim = sag_dim
        
        self.pe = PositionalEncoding(num_hiddens=sag_dim, dropout=dropout, max_len=1024)
        
        self.x_fc = nn.Linear(in_features=emb_dim, out_features=sag_dim)

        self.mha = nn.MultiheadAttention(embed_dim=sag_dim, num_heads=4, batch_first=True, dropout=dropout, vdim=emb_dim)
        
        self.out_fc = nn.Linear(in_features=sag_dim, out_features=emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, hidden_state, x_original):
        B = hidden_state.shape[0]
        
        q = self.pe(self.x_fc(x_original)) # (B, N, sag_dim)
        
        k = self.pe(self.x_fc(hidden_state)) # (B, K, sag_dim)
        
        v = hidden_state # (B, K, emb_dim)
        
        out, _ = self.mha(query=q, key=k, value=v) # (B, N, sag_dim)
        
        out = self.out_fc(out) # sag_dim -> emb_dim
        out = self.norm(out)
        
        return out

class LinearEncoder(nn.Module):
    def __init__(self, num_nodes, sag_tokens):
        super().__init__()
        self.encoder = nn.Linear(num_nodes, sag_tokens)
    
    def forward(self, x):
        # x: (B, N, D)
        x = x.transpose(1, 2) # (B, D, N)
        x = self.encoder(x)   # (B, D, K)
        x = x.transpose(1, 2) # (B, K, D)
        return x, None

class LinearDecoder(nn.Module):
    def __init__(self, num_nodes, sag_tokens):
        super().__init__()
        self.decoder = nn.Linear(sag_tokens, num_nodes)
    
    def forward(self, hidden_state, x_original=None):
        hidden_state = hidden_state.transpose(1, 2) # (B, D, K)
        out = self.decoder(hidden_state) # (B, D, N)
        out = out.transpose(1, 2) # (B, N, D)
        return out
