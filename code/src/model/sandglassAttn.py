import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.position import PositionalEncoding

class SAG(nn.Module):
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout):
        super().__init__()

        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim

        self.hyper_nodes = nn.Parameter(torch.randn(1,sag_tokens,sag_dim))
        #self.pe = nn.Identity()
        self.pe = PositionalEncoding(num_hiddens=sag_dim,dropout=dropout,max_len=1024)

        self.emc_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout)
        self.dec_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout,vdim=emb_dim)

        self.enc_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)
        self.dec_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)

        self.x_fc = nn.Linear(in_features=emb_dim,out_features=sag_dim)


        self.en_ln = nn.LayerNorm(emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)
    def encode(self,x):
        #x(B,N,D)
        B,N,H = x.shape
        # print(x.shape)

        kv = self.x_fc(x)

        q = self.pe(self.hyper_nodes)

        out,attn_weights = self.emc_mha(query=q.repeat(B,1,1),key=self.pe(kv),value=kv) #B,N',D

        out = self.enc_fc(out)

        out = self.en_ln(out)

        return out,attn_weights

    def decode(self,hidden_state,x):
        #hidden_state(B,N',D)
        B,_,_ = hidden_state.shape

        q = self.pe(self.x_fc(x))
        k = self.pe(self.hyper_nodes)
        v = hidden_state

        out,_ = self.dec_mha(query=q,key=k.repeat(B,1,1),value=v) #B,N,H

        out = self.dec_fc(out)

        out = self.de_ln(out)

        return out

class GSAG(nn.Module):
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout):
        super().__init__()

        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim

        self.hyper_nodes = nn.Parameter(torch.randn(1,sag_tokens,sag_dim))
        self.pe = PositionalEncoding(num_hiddens=sag_dim,dropout=dropout,max_len=1024)

        self.emc_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout)
        self.dec_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout,vdim=emb_dim)

        self.enc_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)
        self.dec_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)

        self.x_fc = nn.Linear(in_features=emb_dim,out_features=sag_dim)

        self.en_ln = nn.LayerNorm(emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)
        
        # Graph Convolutions
        self.gcn_enc = nn.Linear(emb_dim, emb_dim)
        self.gcn_dec = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def gcn_forward(self, x, adj, linear_layer):
        # x: (B, N, D)
        # adj: (N, N)
        if adj is not None:
            # AxW
            x_g = torch.einsum('nm,bmd->bnd', adj, x)
            x_g = linear_layer(x_g)
            x = x + self.dropout(F.relu(x_g)) # Residual connection
        return x

    def encode(self, x, adj=None):
        # x(B,N,D)
        
        # Apply GCN before encoding
        x = self.gcn_forward(x, adj, self.gcn_enc)

        B,N,H = x.shape
        kv = self.x_fc(x)

        q = self.pe(self.hyper_nodes)

        out, attn_weights = self.emc_mha(query=q.repeat(B,1,1),key=self.pe(kv),value=kv) #B,N',D

        out = self.enc_fc(out)
        out = self.en_ln(out)

        return out, attn_weights

    def decode(self, hidden_state, x, adj=None):
        # hidden_state(B,N',D)
        B,_,_ = hidden_state.shape

        q = self.pe(self.x_fc(x))
        k = self.pe(self.hyper_nodes)
        v = hidden_state

        out,_ = self.dec_mha(query=q,key=k.repeat(B,1,1),value=v) #B,N,H

        out = self.dec_fc(out)
        out = self.de_ln(out)
        
        # Apply GCN after decoding
        out = self.gcn_forward(out, adj, self.gcn_dec)

        return out


class PerceiverSAG(nn.Module):
    """
    Perceiver-style encoder/decoder with iterative cross-attention.
    Latent tokens are refined through multiple layers for better representation.
    """
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout, num_layers=2):
        super().__init__()
        
        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim
        self.num_layers = num_layers
        
        # Learnable latent array
        self.latent = nn.Parameter(torch.randn(1, sag_tokens, sag_dim))
        self.pe = PositionalEncoding(num_hiddens=sag_dim, dropout=dropout, max_len=1024)
        
        # Input projection
        self.x_fc = nn.Linear(emb_dim, sag_dim)
        
        # Encoder: multiple cross-attention + self-attention layers  
        self.enc_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.enc_self_attn = nn.ModuleList([
            nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.enc_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sag_dim, sag_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(sag_dim * 4, sag_dim)
            ) for _ in range(num_layers)
        ])
        self.enc_norms = nn.ModuleList([nn.LayerNorm(sag_dim) for _ in range(num_layers * 3)])
        
        # Output projection
        self.enc_fc = nn.Linear(sag_dim, emb_dim)
        self.en_ln = nn.LayerNorm(emb_dim)
        
        # Decoder
        self.dec_mha = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout, vdim=emb_dim)
        self.dec_fc = nn.Linear(sag_dim, emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)
    
    def encode(self, x):
        B, N, _ = x.shape
        
        kv = self.pe(self.x_fc(x))  # (B, N, sag_dim)
        latent = self.latent.repeat(B, 1, 1)  # (B, K, sag_dim)
        
        attn_weights = None
        for i in range(self.num_layers):
            # Cross-attention: latent attends to input
            residual = latent
            latent_out, attn_weights = self.enc_cross_attn[i](query=latent, key=kv, value=kv)
            latent = self.enc_norms[i*3](residual + latent_out)
            
            # Self-attention among latent tokens
            residual = latent
            latent_out, _ = self.enc_self_attn[i](query=latent, key=latent, value=latent)
            latent = self.enc_norms[i*3+1](residual + latent_out)
            
            # FFN
            residual = latent
            latent = self.enc_norms[i*3+2](residual + self.enc_ffn[i](latent))
        
        out = self.enc_fc(latent)
        out = self.en_ln(out)
        
        return out, attn_weights
    
    def decode(self, hidden_state, x):
        B = hidden_state.shape[0]
        
        q = self.pe(self.x_fc(x))
        k = self.pe(self.latent)
        v = hidden_state
        
        out, _ = self.dec_mha(query=q, key=k.repeat(B, 1, 1), value=v)
        out = self.dec_fc(out)
        out = self.de_ln(out)
        
        return out


class SetTransformerSAG(nn.Module):
    """
    Set Transformer style with Induced Set Attention Block (ISAB).
    Adds self-attention among induced points for better global reasoning.
    """
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout):
        super().__init__()
        
        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim
        
        # Inducing points
        self.inducing_points = nn.Parameter(torch.randn(1, sag_tokens, sag_dim))
        self.pe = PositionalEncoding(num_hiddens=sag_dim, dropout=dropout, max_len=1024)
        
        self.x_fc = nn.Linear(emb_dim, sag_dim)
        
        # ISAB encoder: cross-attn -> self-attn -> cross-attn back
        self.enc_mha1 = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout)
        self.enc_self_attn = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout)
        self.enc_mha2 = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout)
        
        self.enc_norm1 = nn.LayerNorm(sag_dim)
        self.enc_norm2 = nn.LayerNorm(sag_dim)
        self.enc_norm3 = nn.LayerNorm(sag_dim)
        
        self.enc_fc = nn.Linear(sag_dim, emb_dim)
        self.en_ln = nn.LayerNorm(emb_dim)
        
        # Decoder
        self.dec_mha = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout, vdim=emb_dim)
        self.dec_fc = nn.Linear(sag_dim, emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)
    
    def encode(self, x):
        B, N, _ = x.shape
        
        h = self.pe(self.x_fc(x))  # (B, N, sag_dim)
        inducing = self.inducing_points.repeat(B, 1, 1)  # (B, K, sag_dim)
        
        h_ind, attn_weights = self.enc_mha1(query=self.pe(inducing), key=h, value=h)
        h_ind = self.enc_norm1(inducing + h_ind)
        
        h_ind_self, _ = self.enc_self_attn(query=h_ind, key=h_ind, value=h_ind)
        h_ind = self.enc_norm2(h_ind + h_ind_self)
        
        out = self.enc_fc(h_ind)
        out = self.en_ln(out)
        
        return out, attn_weights
    
    def decode(self, hidden_state, x):
        B = hidden_state.shape[0]
        
        q = self.pe(self.x_fc(x))
        k = self.pe(self.inducing_points)
        v = hidden_state
        
        out, _ = self.dec_mha(query=q, key=k.repeat(B, 1, 1), value=v)
        out = self.dec_fc(out)
        out = self.de_ln(out)
        
        return out


class PoolingSAG(nn.Module):
    """
    Soft clustering-based pooling encoder/decoder.
    Uses learnable soft assignment matrix for node-to-cluster mapping.
    Similar to DiffPool but simplified.
    """
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout):
        super().__init__()
        
        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim
        
        self.x_fc = nn.Linear(emb_dim, sag_dim)
        
        # Soft assignment network: computes S (N x K) assignment matrix
        self.assign_net = nn.Sequential(
            nn.Linear(sag_dim, sag_dim),
            nn.ReLU(),
            nn.Linear(sag_dim, sag_tokens)
        )
        
        # Cluster embedding refinement
        self.cluster_refine = nn.Sequential(
            nn.Linear(sag_dim, sag_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sag_dim * 2, sag_dim)
        )
        
        self.enc_fc = nn.Linear(sag_dim, emb_dim)
        self.en_ln = nn.LayerNorm(emb_dim)
        
        # Decoder: uses transpose of assignment for unpooling
        self.dec_fc = nn.Linear(sag_dim, emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)
        
        self.pe = PositionalEncoding(num_hiddens=sag_dim, dropout=dropout, max_len=1024)
    
    def encode(self, x):
        B, N, _ = x.shape
        
        h = self.x_fc(x)  # (B, N, sag_dim)
        
        # Compute soft assignment matrix S: (B, N, K)
        S = F.softmax(self.assign_net(h), dim=-1)
        
        # Pool: X_cluster = S^T @ X -> (B, K, sag_dim)
        h_cluster = torch.einsum('bnk,bnd->bkd', S, h)
        
        # Refine cluster embeddings
        h_cluster = h_cluster + self.cluster_refine(h_cluster)
        
        # Store assignment for decoder
        self.last_S = S
        
        out = self.enc_fc(h_cluster)
        out = self.en_ln(out)
        
        # Return S as attention weights for compatibility
        attn_weights = S.transpose(1, 2)  # (B, K, N)
        
        return out, attn_weights
    
    def decode(self, hidden_state, x):
        B = hidden_state.shape[0]
        
        h = self.x_fc(x)
        
        # Recompute assignment or use stored
        if hasattr(self, 'last_S') and self.last_S.shape[0] == B:
            S = self.last_S
        else:
            S = F.softmax(self.assign_net(h), dim=-1)
        
        # Project hidden state to sag_dim
        h_hidden = self.x_fc(hidden_state)  # (B, K, sag_dim)
        
        # Unpool: X_out = S @ X_cluster -> (B, N, sag_dim)
        out = torch.einsum('bnk,bkd->bnd', S, h_hidden)
        
        out = self.dec_fc(out)
        out = self.de_ln(out)
        
        return out