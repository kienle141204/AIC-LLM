import torch 
from torch import nn
import numpy as np
from utils.utils import lap_eig, topological_sort
from typing import Optional
from model.sandglassAttn import SAG

class DecodingLayer(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim):
        super(DecodingLayer, self).__init__()
        hidden_size = (emb_dim + output_dim) * 2 // 3

        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
    
    def forward(self, x):
        out = self.fc(x)
        return out
    
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

class Time2Token(nn.Module):
    def __init__(self, sample_len, features, emb_dim, tim_dim, drop_out):
        super(Time2Token, self).__init__()
        self.sample_len = sample_len
        self.features = features
        self.emb_dim = emb_dim

        in_features = sample_len * features + tim_dim
        hidden_size = (in_features + emb_dim)*2//3

        self.fc_state = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        input_dim = tim_dim + features * (sample_len-1)
        hidden_size = (input_dim + emb_dim)*2//3
        self.fc_grad = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x, te, mask=None):
        B, N, TF = x.shape

        x = x.view(B, N, self.sample_len, -1)   #(B,N,T,F)
        x = x.mean(dim=1)   #(B,T,F)

        state = x.view(B,1,-1)
        state = torch.concat((state,te[:,-1:,:]),dim=-1)    #(B,1,TF+tim_dim)
        state = self.fc_state(state)

        grad = (x[:,1:,:] - x[:,:-1,:]).view(B,1,-1)    #(B,1,(T-1)F)
        grad = torch.concat((grad,te[:,-1:,:]),dim=-1)    #(B,1,(T-1)F+tim_dim)
        grad = self.fc_grad(grad)

        out = torch.concat((state,grad),dim=1)

        out = self.ln(out)

        return out
    

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
    

class AICLLM(nn.Module):
    def __init__(self, basemodel: nn.Module, sample_len: int, output_len: int,
                 input_dim: int, output_dim: int,
                 node_emb_dim: int,
                 sag_dim: int, sag_tokens: int,
                 dropout: float, adj_mx = None, dis_mx = None,
                 use_node_embedding: bool = True,
                 use_time_token: bool = True,
                 use_sandglassAttn: bool = True,
                 t_dim: int = 64, trunc_k=16, wo_conloss=False) :
        super(AICLLM, self).__init__()

        self.basemodel = basemodel
        self.sample_len = sample_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_node_embedding = use_node_embedding
        self.use_time_token = use_time_token
        self.sag_tokens = sag_tokens

        self.topological_sort_node = True

        self.emb_dim = basemodel.dim
        tim_dim = t_dim*2     #day, week
        self.setadj(adj_mx,dis_mx)
        
        self.time_tokenizer = Time2Token(
            sample_len=sample_len, 
            features=input_dim, 
            emb_dim=self.emb_dim, 
            tim_dim=tim_dim, 
            drop_out=dropout
        )
        self.node_tokenizer = Node2Token(
            sample_len=sample_len, 
            features=input_dim, 
            node_emb_dim=node_emb_dim, 
            emb_dim=self.emb_dim, 
            tim_dim=tim_dim, 
            dropout=dropout,
            use_node_embedding=use_node_embedding
        )

        self.node_embedding = NodeEmbedding(adj_mx=adj_mx, node_emb_dim=node_emb_dim, k=trunc_k, dropout=dropout)
        self.time_embedding = TimeEmbedding(t_dim=t_dim)

        # if use_sandglassAttn:
        self.sag = SAG(
            sag_dim=sag_dim,
            sag_tokens=sag_tokens,
            emb_dim=self.emb_dim,
            sample_len=sample_len,
            features=input_dim,
            dropout=dropout
        )
        self.wo_conloss = wo_conloss
        
        self.out_mlp = DecodingLayer(
            input_dim=output_dim*sample_len,
            emb_dim=self.emb_dim,
            output_dim=output_dim * output_len
        )

        self.layer_norm = nn.LayerNorm(self.emb_dim)
    
    def forward(self, x: torch.FloatTensor, timestamp: torch.Tensor, prompt_prefix: Optional[torch.Tensor]):
        B, N, TF = x.shape
        other_loss = []

        timestamp = timestamp[:, :self.sample_len, :]
        te = self.time_embedding(timestamp) 
        ne = self.node_embedding()

        # spatial tokenizer 
        spatial_tokens = self.node_tokenizer(x, te, ne)  # (B, N, emb_dim)
        if self.topological_sort_node:
            spatial_tokens = spatial_tokens[:, self.node_order, :]
        
        st_embedding = spatial_tokens
        s_num = N
        s_num = self.sag_tokens
        # if self.use_sandglassAttn:
        st_embedding, attn_weights = self.sag.encode(st_embedding)
        if not self.wo_conloss:
            scale = attn_weights.sum(dim=1)#(B,N)

            sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],attn_weights)
            other_loss.append(-((sag_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10)

            Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
            other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())
        
        # time tokenizer
        time_tokens = self.time_tokenizer(x, te)
        time_tokens_idx = st_embedding.shape[1]
        st_embedding = torch.concat((time_tokens, st_embedding), dim=1)  

        if prompt_prefix is not None:
            prompt_len,_ = prompt_prefix.shape
            prompt_embedding = self.basemodel.getembedding(prompt_prefix).view(1,prompt_len,-1)
            prompt_embedding = prompt_embedding.repeat(B,1,1)
            st_embedding = torch.concat([prompt_embedding,st_embedding],dim=1)
        
        hidden_state = st_embedding

        hidden_state = self.basemodel(hidden_state)
        s_state = hidden_state[:, -s_num:, :]  

        s_state = self.sag.decode(s_state, spatial_tokens)  
        s_state += spatial_tokens

        if self.topological_sort_node:
            s_state = s_state[:,self.node_order_rev,:]

        if self.use_time_token:
            t_state = hidden_state[:,-time_tokens_idx-1:-time_tokens_idx,:]
            t_state += time_tokens[:,-1:,:]

            s_state += t_state

        s_state = self.layer_norm(s_state)

        out = self.out_mlp(s_state)

        return out, other_loss
            
    
    def grad_state_dict(self):
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return  {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    
    def save(self, path:str):
        
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):

        loaded_params = torch.load(path)
        self.load_state_dict(loaded_params,strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params

    def setadj(self,adj_mx,dis_mx):

        self.adj_mx = torch.tensor(adj_mx).cuda()
        self.dis_mx = torch.tensor(dis_mx).cuda()
        self.d_mx = self.adj_mx.sum(dim=1)
        N = self.adj_mx.shape[0]
        self.alpha = torch.tensor([1.05] * N).cuda() + torch.softmax(self.d_mx,dim=0)*5 
        self.node_order,self.node_order_rev = topological_sort(adj_mx)

        