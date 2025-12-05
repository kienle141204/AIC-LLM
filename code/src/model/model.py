import torch 
from torch import nn
import numpy as np
from utils.utils import lap_eig, topological_sort
from typing import Optional
from model.sandglassAttn import SpatialEncoder, SpatialDecoder, LinearEncoder, LinearDecoder, SAG
from model.embedding import TimeEmbedding, NodeEmbedding
from model.tokenizer import AnchorDiffTokenizer, Time2Token, Node2Token

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

class AICLLM(nn.Module):
    def __init__(self, basemodel: nn.Module, sample_len: int, output_len: int,
                 input_dim: int, output_dim: int,
                 node_emb_dim: int,
                 sag_dim: int, sag_tokens: int,
                 dropout: float, adj_mx = None, dis_mx = None,
                 use_node_embedding: bool = True,
                 use_time_token: bool = True,
                 use_sandglassAttn: bool = True,
                 use_anchor_diff_token: bool = False,
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
        self.use_sandglassAttn = use_sandglassAttn
        self.use_anchor_diff_token = use_anchor_diff_token

        self.topological_sort_node = True

        self.emb_dim = basemodel.dim
        tim_dim = t_dim*2     #day, week
        self.setadj(adj_mx,dis_mx)
        
        # Original Time Tokenizer (always used for x)
        self.time_tokenizer = Time2Token(
            sample_len=sample_len, 
            features=input_dim, 
            emb_dim=self.emb_dim, 
            tim_dim=tim_dim, 
            drop_out=dropout
        )

        # Optional Anchor-Diff Tokenizer
        if self.use_anchor_diff_token == 1:
            self.anchor_diff_tokenizer = AnchorDiffTokenizer(
                sample_len=sample_len, 
                features=input_dim, 
                emb_dim=self.emb_dim, 
                tim_dim=tim_dim, 
                drop_out=dropout
            )
        elif self.use_anchor_diff_token == 2:
            self.anchor_tokenizer = Time2Token(
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

        if self.use_sandglassAttn:
            # self.precoder = SpatialEncoder(
            #     sag_dim=sag_dim,
            #     sag_tokens=sag_tokens,
            #     emb_dim=self.emb_dim,
            #     dropout=dropout
            # )
            # self.decoder = SpatialDecoder(
            #     sag_dim=sag_dim,
            #     emb_dim=self.emb_dim,
            #     dropout=dropout
            # )
            self.sag = SAG(
                sag_dim=sag_dim,
                sag_tokens=sag_tokens,
                emb_dim=self.emb_dim,
                dropout=dropout,
                use_node_embedding=use_node_embedding
            )
        else:
            N = self.adj_mx.shape[0]
            self.precoder = LinearEncoder(num_nodes=N, sag_tokens=sag_tokens)
            self.decoder = LinearDecoder(num_nodes=N, sag_tokens=sag_tokens)

        self.wo_conloss = wo_conloss
        
        self.out_mlp = DecodingLayer(
            input_dim=output_dim*sample_len,
            emb_dim=self.emb_dim,
            output_dim=output_dim * output_len
        )

        self.layer_norm = nn.LayerNorm(self.emb_dim)
    
    def forward(self, x: torch.FloatTensor, xa: torch.FloatTensor, timestamp: torch.Tensor, prompt_prefix: Optional[torch.Tensor]):
        B, N, TF = x.shape
        other_loss = []
        
        x_spatial = x
        # if self.use_diff:
        x_diff = x - xa

        timestamp = timestamp[:, :self.sample_len, :]
        te = self.time_embedding(timestamp) 
        ne = self.node_embedding()

        # spatial tokenizer 
        spatial_tokens = self.node_tokenizer(x_spatial, te, ne)  # (B, N, emb_dim)
        if self.topological_sort_node:
            spatial_tokens = spatial_tokens[:, self.node_order, :]
        
        st_embedding = spatial_tokens
        s_num = N
        s_num = self.sag_tokens
        
        # Precoder
        if self.use_sandglassAttn:
            st_embedding, attn_weights = self.sag.encoder(st_embedding)
        else:
            st_embedding = self.precoder(st_embedding)  
        
        if self.use_sandglassAttn and not self.wo_conloss:
            # Only calculate consistency loss if using Attention
            if attn_weights is not None:
                scale = attn_weights.sum(dim=1)#(B,N)

                sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],attn_weights)
                other_loss.append(-((sag_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10)

                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())
        
        # Time Tokenizer
        time_tokens = self.time_tokenizer(x, te)
        time_tokens_idx = st_embedding.shape[1]
        st_embedding = torch.concat((time_tokens, st_embedding), dim=1)  

        if self.use_anchor_diff_token == 1:
            ad_tokens = self.anchor_diff_tokenizer(x, xa, te)
            st_embedding = torch.concat((ad_tokens, st_embedding), dim=1)
        elif self.use_anchor_diff_token == 2:
            anchor_tokens = self.anchor_tokenizer(xa, te)
            st_embedding = torch.concat((anchor_tokens, st_embedding), dim=1)
        
        if prompt_prefix is not None:
            prompt_len,_ = prompt_prefix.shape
            prompt_embedding = self.basemodel.getembedding(prompt_prefix).view(1,prompt_len,-1)
            prompt_embedding = prompt_embedding.repeat(B,1,1)
            st_embedding = torch.concat([prompt_embedding,st_embedding],dim=1)
        
        hidden_state = st_embedding

        hidden_state = self.basemodel(hidden_state)
        s_state = hidden_state[:, -s_num:, :]  

        # Decoder
        if self.use_sandglassAttn:
            s_state = self.sag.decoder(s_state, spatial_tokens)
        else:
            s_state = self.decoder(s_state, spatial_tokens)  
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