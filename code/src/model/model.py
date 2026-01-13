import torch 
from torch import nn
import numpy as np
from utils.utils import lap_eig, topological_sort
from typing import Optional
from model.sandglassAttn import SAG, PerceiverSAG, SetTransformerSAG, PoolingSAG
from model.embedding import TimeEmbedding, NodeEmbedding
from model.tokenizer import Time2Token, Node2Token
from prompts import  ANCHOR_DATA_INSTRUCTION, CURRENT_DATA_INSTRUCTION, get_statistics

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
                 use_sandglassAttn: int = 0,

                 task_type: str = 'prediction',
                 user_instruction: bool = True,
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
        self.adj_mx = adj_mx
        self.dis_mx = dis_mx
        self.task_type = task_type
        self.user_instruction = user_instruction

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
        
        # Sandglass Attn
        if self.use_sandglassAttn == 1:
            self.sag = SAG(sag_dim=sag_dim, 
                           sag_tokens=sag_tokens, 
                           emb_dim=self.emb_dim, 
                           sample_len=sample_len, 
                           features=input_dim ,
                           dropout=dropout
                           )
            self.sag_anchor = SAG(sag_dim=sag_dim, 
                                  sag_tokens=sag_tokens, 
                                  emb_dim=self.emb_dim, 
                                  sample_len=sample_len, 
                                  features=input_dim ,
                                  dropout=dropout
                                  )
        # elif self.use_sandglassAttn == 1:
        #     self.sag = PerceiverSAG(sag_dim=sag_dim, 
        #                             sag_tokens=sag_tokens, 
        #                             emb_dim=self.emb_dim, 
        #                             sample_len=sample_len, 
        #                             features=input_dim ,
        #                             dropout=dropout
        #                             )
        elif self.use_sandglassAttn == 2:
            self.sag = SetTransformerSAG(sag_dim=sag_dim, 
                                        sag_tokens=sag_tokens, 
                                        emb_dim=self.emb_dim, 
                                        sample_len=sample_len, 
                                        features=input_dim ,
                                        dropout=dropout
                                        )
            self.sag_anchor = SetTransformerSAG(sag_dim=sag_dim, 
                                  sag_tokens=sag_tokens, 
                                  emb_dim=self.emb_dim, 
                                  sample_len=sample_len, 
                                  features=input_dim ,
                                  dropout=dropout
                                  )
        # elif self.use_sandglassAttn == 3:
        #     self.sag = PoolingSAG(sag_dim=sag_dim, 
        #                           sag_tokens=sag_tokens, 
        #                           emb_dim=self.emb_dim, 
        #                           sample_len=sample_len,    
        #                           features=input_dim ,
        #                           dropout=dropout
        #                           )
            

        self.wo_conloss = wo_conloss
        
        self.out_mlp = DecodingLayer(
            input_dim=output_dim*sample_len,
            emb_dim=self.emb_dim,
            output_dim=output_dim * output_len
        )

        self.layer_norm = nn.LayerNorm(self.emb_dim)
        
        if self.user_instruction:
            # Pre-tokenize instructions once to avoid repeated tokenization during training
            self.register_buffer('anchor_instruction_ids', None)
            self.register_buffer('current_instruction_ids', None)
    
    def forward(self, x: torch.FloatTensor, xa: torch.FloatTensor, ya: torch.FloatTensor, timestamp: torch.Tensor, prompt_prefix: Optional[torch.Tensor]):
        B, N, TF = x.shape
        other_loss = []

        statistics = get_statistics(x)
        prompt_prefix = prompt_prefix.format(statistics=statistics)
        
        timestamp = timestamp[:, :self.sample_len, :]
        te = self.time_embedding(timestamp) 
        ne = self.node_embedding()

        # ========== CURRENT DATA PROCESSING ==========
        # spatial tokenizer 
        spatial_tokens = self.node_tokenizer(x, te, ne)  # (B, N, emb_dim)
        if self.topological_sort_node:
            spatial_tokens = spatial_tokens[:, self.node_order, :]
        
        st_embedding_current = spatial_tokens
        s_num = self.sag_tokens
        
        # Precoder (SAG encode)
        if self.use_sandglassAttn:
            st_embedding_current, attn_weights = self.sag.encode(st_embedding_current)
        else:
            st_embedding_current, attn_weights = self.precoder(st_embedding_current)

        if self.use_sandglassAttn and not self.wo_conloss:
            if attn_weights is not None:
                scale = attn_weights.sum(dim=1)    #(B,N)
                sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],attn_weights)
                other_loss.append(-((sag_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10)
                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())
        
        # Time Tokenizer for current data
        time_tokens = self.time_tokenizer(x, te)
        current_data = torch.concat((time_tokens, st_embedding_current), dim=1)

        # Current instruction
        if self.user_instruction:
            if self.current_instruction_ids is None:
                tokenizer = self.basemodel.gettokenizer()
                current_instruction_tokens = tokenizer(CURRENT_DATA_INSTRUCTION, 
                                return_tensors="pt", return_attention_mask=False)
                self.current_instruction_ids = current_instruction_tokens['input_ids'].cuda()
            
            current_instruction_emb = self.basemodel.getembedding(self.current_instruction_ids).squeeze(0)
            current_instruction_emb = current_instruction_emb.unsqueeze(0).expand(B, -1, -1)
            current_data = torch.concat((current_instruction_emb, current_data), dim=1)

        # ========== ANCHOR DATA PROCESSING ==========
        spatial_anchor = self.node_tokenizer(xa, te, ne)
        if self.topological_sort_node:
            spatial_anchor = spatial_anchor[:, self.node_order, :]
        
        if self.use_sandglassAttn:
            spatial_anchor, antt_weights_1 = self.sag.encode(spatial_anchor)
        else:
            spatial_anchor, antt_weights_1 = self.precoder(spatial_anchor)
        
        if self.use_sandglassAttn and not self.wo_conloss:
            if antt_weights_1 is not None:
                scale = antt_weights_1.sum(dim=1)
                sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],antt_weights_1)
                other_loss.append(-((sag_score*antt_weights_1-antt_weights_1*antt_weights_1)).sum(dim=2).mean()*10)
                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())
        
        time_anchor = self.time_tokenizer(xa, te)
        anchor_data = torch.concat((time_anchor, spatial_anchor), dim=1)

        if self.user_instruction:
            if self.anchor_instruction_ids is None:
                tokenizer = self.basemodel.gettokenizer()
                anchor_instruction_tokens = tokenizer(ANCHOR_DATA_INSTRUCTION, 
                                return_tensors="pt", return_attention_mask=False)
                self.anchor_instruction_ids = anchor_instruction_tokens['input_ids'].cuda()
            
            anchor_instruction_emb = self.basemodel.getembedding(self.anchor_instruction_ids).squeeze(0)
            anchor_instruction_emb = anchor_instruction_emb.unsqueeze(0).expand(B, -1, -1)
            anchor_data = torch.concat((anchor_instruction_emb, anchor_data), dim=1)

        # ========== COMBINE ALL ==========
        st_embedding = torch.concat((anchor_data, current_data), dim=1)
        
        if prompt_prefix is not None and self.user_instruction:
            prompt_len,_ = prompt_prefix.shape
            prompt_embedding = self.basemodel.getembedding(prompt_prefix).view(1,prompt_len,-1)
            prompt_embedding = prompt_embedding.repeat(B,1,1)
            st_embedding = torch.concat([prompt_embedding, st_embedding],dim=1)
            prompt_offset = prompt_len
        else:
            prompt_offset = 0
        
        # ========== LLM PROCESSING ==========
        hidden_state = self.basemodel(st_embedding)
        
        anchor_len = anchor_data.shape[1]
        
        current_data_start_idx = prompt_offset + anchor_len
        
        current_instruction_len = current_instruction_emb.shape[1] if self.user_instruction else 0
        
        time_tokens_len = time_tokens.shape[1] # This is typically 1
        
        current_spatial_sag_start = current_data_start_idx + current_instruction_len + time_tokens_len
        s_state = hidden_state[:, current_spatial_sag_start : current_spatial_sag_start + s_num, :]

        # Decoder
        if self.use_sandglassAttn:
            s_state = self.sag.decode(s_state, spatial_tokens)
        else:
            s_state = self.decoder(s_state, spatial_tokens)  
        s_state += spatial_tokens

        if self.topological_sort_node:
            s_state = s_state[:,self.node_order_rev,:]

        if self.use_time_token:
            time_token_pos = current_data_start_idx + current_instruction_len
            t_state = hidden_state[:, time_token_pos : time_token_pos + time_tokens_len, :]
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