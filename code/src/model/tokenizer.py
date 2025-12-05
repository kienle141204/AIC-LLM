import torch 
from torch import nn
import numpy as np

class Time2Token(nn.Module):
    def __init__(self, sample_len, features, emb_dim, tim_dim, num_tokens=3, drop_out=0.1):
        super(Time2Token, self).__init__()
        self.sample_len = sample_len
        self.features = features
        
        # Cfeatures: trend, seasonality, residual
        self.trend_conv = nn.Conv1d(features, emb_dim, kernel_size=5, padding=2)

        self.season_conv = nn.Conv1d(features, emb_dim, kernel_size=2, padding=1)

        self.residual_fc = nn.Linear(sample_len * features + tim_dim, emb_dim)
        
        self.ln = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(drop_out)
        
    def forward(self, x, te, mask=None):
        B, N, TF = x.shape
        x = x.view(B, N, self.sample_len, self.features)
        
        x = x.mean(dim=1)  # (B,T,F)
        
        # Trend token
        x_t = x.transpose(1, 2)  # (B,F,T)
        trend = self.trend_conv(x_t)  # (B,emb_dim,T)
        trend_token = trend.mean(dim=-1)  # (B,emb_dim)
        
        # Seasonality token
        season = self.season_conv(x_t)[:, :, :self.sample_len]  # (B,emb_dim,T)
        season_token = season.max(dim=-1)[0]  # (B,emb_dim)
        
        # Residual token
        x_flat = x.reshape(B, -1)  # (B,T*F)
        residual_input = torch.cat([x_flat, te[:,-1,:]], dim=1)
        residual_token = self.residual_fc(residual_input)  # (B,emb_dim)
        
        tokens = torch.stack([trend_token, season_token, residual_token], dim=1)
        tokens = self.dropout(tokens)
        tokens = self.ln(tokens)
        
        return tokens

class AnchorNet(nn.Module):
    def __init__(self, sample_len, features, emb_dim, tim_dim, drop_out=0.1):
        super(AnchorNet, self).__init__()
        self.sample_len = sample_len
        self.features = features
        
        # Input: Flattened anchor (T*F) + Time Embedding (tim_dim)
        input_dim = sample_len * features + tim_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        
        self.ln = nn.LayerNorm(emb_dim)
        
    def forward(self, x_anchor, te):
        # x_anchor: (B, N, T*F)
        # te: (B, T, tim_dim)
        
        B, N, TF = x_anchor.shape
        
        # Mean over nodes to get global anchor context
        x_anchor_mean = x_anchor.mean(dim=1) # (B, TF)
        
        # Take last time step embedding
        t_emb = te[:, -1, :] # (B, tim_dim)
        
        # Combine
        inp = torch.cat([x_anchor_mean, t_emb], dim=1) # (B, TF + tim_dim)
        
        out = self.mlp(inp) # (B, emb_dim)
        
        # Add dimension to match token shape (B, 1, emb_dim)
        out = out.unsqueeze(1)
        out = self.ln(out)
        
        return out

class AnchorDiffTokenizer(nn.Module):
    def __init__(self, sample_len, features, emb_dim, tim_dim, drop_out=0.1):
        super(AnchorDiffTokenizer, self).__init__()
        
        # Network for Diff (using original Time2Token logic)
        self.diff_net = Time2Token(sample_len, features, emb_dim, tim_dim, drop_out=drop_out)
        
        # Network for Anchor (using new MLP logic)
        self.anchor_net = AnchorNet(sample_len, features, emb_dim, tim_dim, drop_out=drop_out)
        
    def forward(self, x, x_anchor, te):
        # x: Current input (B, N, TF)
        # x_anchor: Anchor input (B, N, TF)
        
        # 1. Calculate Diff
        x_diff = x - x_anchor
        
        # 2. Get Diff Tokens (Trend, Season, Residual from Diff)
        diff_tokens = self.diff_net(x_diff, te) # (B, 3, D)
        
        # 3. Get Anchor Token (Global context from Anchor)
        anchor_token = self.anchor_net(x_anchor, te) # (B, 1, D)
        
        # 4. Combine
        return torch.cat([anchor_token, diff_tokens], dim=1)

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