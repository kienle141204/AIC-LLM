import torch 
from torch import nn

class TimeEmbedding(nn.Module):
    def __init__(self, t_dim):
        super(TimeEmbedding, self).__init__()
        self.day_embedding = nn.Embedding(num_embeddings=288, embedding_dim=t_dim)
        self.week_embedding = nn.Embedding(num_embeddings=7, embedding_dim=t_dim)
    
    def forward(self, TE):
        B, T, _ = TE.shape

        week = (TE[..., 2].to(torch.long) % 7).view(B * T, -1)
        hour = (TE[..., 3].to(torch.long) % 24).view(B * T, -1)
        minute = (TE[..., 4].to(torch.long) % 60).view(B * T, -1)

        WE = self.week_embedding(week).view(B, T, -1)
        DE = self.day_embedding((hour*60 + minute)//5).view(B, T, -1)

        TE = torch.cat([WE, DE], dim=-1).view(B, T, -1)
        return TE

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
    
class AT2Token(nn.Module):
    def __init__(self, sample_len, features, emb_dim, tim_dim, drop_out):
        super(AT2Token, self).__init__()
        self.sample_len = sample_len
        self.features = features
        self.emb_dim = emb_dim
        self.tim_dim = tim_dim
        self.drop_out = drop_out

        in_features = sample_len * features + tim_dim
        hidden_size = (in_features + emb_dim)*2//3
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        self.ln = nn.LayerNorm(emb_dim)    
    def forward(self, x1, x2, te, mask=None):
        B, N, TF = x1.shape

        x1 = x1.view(B, N, self.sample_len, -1)   #(B,N,T,F)
        x1 = x1.mean(dim=1)   #(B,T,F)

        x2 = x2.view(B, N, self.sample_len, -1)   #(B,N,T,F)
        x2 = x2.mean(dim=1)   #(B,T,F)

        x1 = x1.view(B,1,-1)
        x2 = x2.view(B,1,-1)

        x1 = torch.concat((x1,te[:,-1:,:]),dim=-1)    #(B,1,TF+tim_dim)
        x1 = self.fc1(x1)

        x2 = torch.concat((x2,te[:,-1:,:]),dim=-1)    #(B,1,TF+tim_dim)
        x2 = self.fc2(x2)

        out = torch.concat((x1,x2),dim=1)

        out = self.ln(out)

        return out

