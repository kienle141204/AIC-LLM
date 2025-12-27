import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from model.graph_tokenizer import ADCRNN_Encoder, ADCRNN_Decoder

class AICLLM(nn.Module):
    def __init__(self, basemodel: nn.Module, sample_len: int, output_len: int,
                 input_dim: int, output_dim: int,
                 node_emb_dim: int,
                 sag_dim: int, sag_tokens: int, 
                 dropout: float, adj_mx, dis_mx = None,
                 use_node_embedding: bool = True,
                 use_time_token: bool = True, 
                 rnn_units=128, rnn_layers=1, cheb_k=3,
                 prototype_num=20, prototype_dim=64, tod_embed_dim=10, 
                 use_curriculum_learning=True, use_STE=True,
                 adaptive_embedding_dim=48, input_embedding_dim=128,
                 cl_decay_steps=2000, TDAY=288,
                 **kwargs) :
        super(AICLLM, self).__init__()

        self.basemodel = basemodel
        self.num_nodes = adj_mx.shape[0] if adj_mx is not None else 0
        
        self.sample_len = sample_len
        self.horizon = output_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # STSSDL Parameters
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.cheb_k = cheb_k
        self.prototype_num = prototype_num
        self.prototype_dim = prototype_dim
        self.tod_embed_dim = tod_embed_dim
        self.use_curriculum_learning = use_curriculum_learning
        self.use_STE = use_STE
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.input_embedding_dim = input_embedding_dim
        self.node_embedding_dim = node_emb_dim
        self.cl_decay_steps = cl_decay_steps
        self.TDAY = TDAY
        
        # Adjacency
        if isinstance(adj_mx, list):
             self.adj_mx_list = [ torch.tensor(a).cuda() if not isinstance(a, torch.Tensor) else a.cuda() for a in adj_mx]
        elif isinstance(adj_mx, torch.Tensor):
             self.adj_mx_list = [adj_mx.cuda()]
        elif isinstance(adj_mx, np.ndarray):
             self.adj_mx_list = [torch.tensor(adj_mx).cuda()]
        else:
             self.adj_mx_list = [] # Should not happen ideally
             
        self.d_mx = self.adj_mx_list[0].sum(dim=1) if len(self.adj_mx_list) > 0 else None
        
        self.total_embedding_dim = self.tod_embed_dim + self.adaptive_embedding_dim + self.node_embedding_dim
        
        # Prototypes
        self.prototypes = self.construct_prototypes()
        
        # STE & Projection
        if self.use_STE:
            if self.adaptive_embedding_dim > 0:
                self.adaptive_embedding = nn.init.xavier_uniform_(
                    nn.Parameter(torch.empty(12, self.num_nodes, self.adaptive_embedding_dim))
                )
            
            self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
            
            self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, self.node_embedding_dim))
            self.time_embedding = nn.Parameter(torch.empty(self.TDAY, self.tod_embed_dim))
            nn.init.xavier_uniform_(self.node_embedding)
            nn.init.xavier_uniform_(self.time_embedding)

        # LLM Integration
        self.feature_dim_per_step = self.input_embedding_dim + self.total_embedding_dim  # feature dim per timestep
        st_feature_dim = self.feature_dim_per_step * self.sample_len  # total flattened dim for LLM
        self.llm_in_proj = nn.Linear(st_feature_dim, basemodel.dim)
        self.llm_out_proj = nn.Linear(basemodel.dim, st_feature_dim)
        
        # Encoder: input is (B, T, N, feature_dim_per_step), not flattened
        self.encoder = ADCRNN_Encoder(
            self.num_nodes, 
            self.feature_dim_per_step,  # input dim per timestep, NOT st_feature_dim
            self.rnn_units, 
            self.cheb_k, 
            self.rnn_layers, 
            len(self.adj_mx_list)
        )
        
        # Decoder
        self.decoder_dim = self.rnn_units + self.prototype_dim
        if self.use_STE:
             decoder_input_dim = self.input_embedding_dim + self.total_embedding_dim - self.adaptive_embedding_dim
        else:
             decoder_input_dim = self.output_dim
             
        self.decoder = ADCRNN_Decoder(
            self.num_nodes, 
            decoder_input_dim, 
            self.decoder_dim, 
            self.cheb_k, 
            self.rnn_layers, 
            1 
        )
        
        # Output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        
        # Graph / Hypernet
        self.hypernet = nn.Sequential(nn.Linear(self.decoder_dim*2, self.tod_embed_dim, bias=True))

        self.act_fn = 'sigmoid'
        
        self.wo_conloss = False # default

    def construct_prototypes(self):
        prototypes_dict = nn.ParameterDict()
        prototype = torch.randn(self.prototype_num, self.prototype_dim)
        prototypes_dict['prototypes'] = nn.Parameter(prototype, requires_grad=True)     # (M, d)
        prototypes_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.prototype_dim), requires_grad=True)    # project to query
        for param in prototypes_dict.values():
            nn.init.xavier_normal_(param)
        return prototypes_dict
    
    def query_prototypes(self, h_t):
        query = torch.matmul(h_t, self.prototypes['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.prototypes['prototypes'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.prototypes['prototypes'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.prototypes['prototypes'][ind[:, :, 0]] # B, N, d
        neg = self.prototypes['prototypes'][ind[:, :, 1]] # B, N, d
        mask = torch.stack([ind[:, :, 0], ind[:, :, 1]], dim=-1) # B, N, 2
        return value, query, pos, neg, mask

    def calculate_distance(self, pos, pos_his, mask=None):
        score = torch.sum(torch.abs(pos - pos_his), dim=-1)
        return score, mask

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

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
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, total_trainable_params

    def forward(self, x, x_cov, timestamp, prompt_prefix=None, batches_seen=None):
        # x is (B, N, TF). Needs to be (B, T, N, F)
        
        B, N, TF = x.shape
        T = self.sample_len
        feat_dim = self.input_dim  # Renamed from F to avoid shadowing torch.nn.functional
        
        x = x.view(B, N, T, feat_dim).permute(0, 2, 1, 3)  # (B, T, N, feat_dim)
        
        # Feature Construction (STE)
        if self.use_STE:
            # x Projection
            if self.input_embedding_dim > 0:
                x_emb = self.input_proj(x)
            else:
                x_emb = x
            
            features = [x_emb]
            
            # Time Embedding
            
            if self.tod_embed_dim > 0:
                tod = timestamp[:, :self.sample_len, 0] # (B, T)
                tod = tod.unsqueeze(-1).expand(-1, -1, N) # (B, T, N)
                
                # Discretize
                tod_idx = (tod * self.TDAY).long()
                tod_idx = torch.clamp(tod_idx, 0, self.TDAY - 1)
                time_emb = self.time_embedding[tod_idx] # (B, T, N, d)
                features.append(time_emb)
            
            # Adaptive
            if self.adaptive_embedding_dim > 0:
                adp_emb = self.adaptive_embedding.expand(size=(B, *self.adaptive_embedding.shape)) # (B, 12, N, d) -> (B, T, N, d)
                if adp_emb.shape[1] != self.sample_len:
                     if adp_emb.shape[1] > self.sample_len:
                         adp_emb = adp_emb[:, :self.sample_len, :, :]
                     else:
                         adp_emb = F.pad(adp_emb, (0,0,0,0,0,0,0, self.sample_len - adp_emb.shape[1]))
                         
                features.append(adp_emb)
            
            # Node Embedding
            if self.node_embedding_dim > 0:
                node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(B, self.sample_len, -1, -1)
                features.append(node_emb)
                
            x_combined = torch.cat(features, dim=-1) # (B, T, N, D_total)
        else:
             x_combined = x

        # LLM Injection
        # (B, T, N, D) -> (B, N, T*D)
        
        B_T_N_D = x_combined.permute(0, 2, 1, 3).reshape(B, N, -1) # (B, N, T*D)
        
        # Project D -> LLM_Dim
        llm_in = self.llm_in_proj(B_T_N_D) # (B, N, LLM_Dim)
        
        # Run LLM
        llm_out = self.basemodel(llm_in) # (B, N, LLM_Dim)
        
        # Project back
        llm_out_proj = self.llm_out_proj(llm_out) # (B, N, T*D)
        
        # Reshape to (B, T, N, D)
        x_encoded_llm = llm_out_proj.view(B, N, T, -1).permute(0, 2, 1, 3) # (B, T, N, D)
        
        # 3. STSSDL Processing (Encoder -> Prototypes -> Decoder)
        supports_en = self.adj_mx_list
        init_state = self.encoder.init_hidden(B)
        
        # Encoder
        h_en, state_en = self.encoder(x_encoded_llm, init_state, supports_en) # (B, T, N, hidden)
        
        # Last State
        h_t = h_en[:, -1, :, :] # (B, N, hidden)
        
        # Prototypes
        v_t, q_t, p_t, n_t, mask = self.query_prototypes(h_t)
        
        # Historical / Anchor logic
        x_his = x_cov 
        
        other_loss = []
        
        h_a = None
        if x_his is not None:
             # Reprocess x_his like x
             B, N, TF_his = x_his.shape
             x_his_view = x_his.view(B, N, T, feat_dim).permute(0, 2, 1, 3)
             
             # STE for Hist
             if self.use_STE:
                if self.input_embedding_dim > 0:
                    x_his_emb = self.input_proj(x_his_view)
                else:
                    x_his_emb = x_his_view
                features_his = [x_his_emb]

                if self.tod_embed_dim > 0:
                     features_his.append(time_emb)
                if self.adaptive_embedding_dim > 0: features_his.append(adp_emb)
                if self.node_embedding_dim > 0: features_his.append(node_emb)
                
                x_his_combined = torch.cat(features_his, dim=-1)
             else:
                x_his_combined = x_his_view
             # LLM for history?
             # (B, N, T*D)
             B_T_N_D_his = x_his_combined.permute(0, 2, 1, 3).reshape(B, N, -1)
             llm_in_his = self.llm_in_proj(B_T_N_D_his)
             llm_out_his = self.basemodel(llm_in_his)
             llm_out_proj_his = self.llm_out_proj(llm_out_his)
             x_encoded_llm_his = llm_out_proj_his.view(B, N, T, -1).permute(0, 2, 1, 3)
             
             h_his_en, _ = self.encoder(x_encoded_llm_his, init_state, supports_en)
             h_a = h_his_en[:, -1, :, :]
             
             v_a, q_a, p_a, n_a, mask_his = self.query_prototypes(h_a)
             
             # Losses
             latent_dis, _ = self.calculate_distance(q_t, q_a)
             prototype_dis, mask_dis = self.calculate_distance(p_t, p_a)
             
             # deviation loss
             loss_d = F.l1_loss(latent_dis.detach(), prototype_dis)
             other_loss.append(loss_d * 1) # Weight
             
             # contrastive
             contrastive_loss_fn = nn.TripletMarginLoss(margin=0.5)
             loss_c = contrastive_loss_fn(q_t.detach(), p_t, n_t)
             other_loss.append(loss_c * 0.1)
             
             h_aug = torch.cat([h_t, v_t, h_a, v_a], dim=-1)
        else:
             h_aug = torch.cat([h_t, v_t, h_t, v_t], dim=-1) 
             
        # HyperNet -> Supports
        node_embeddings = self.hypernet(h_aug) 
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1) 
        supports_de = [support]
        
        # Decoder
        h_de = torch.cat([h_t, v_t], dim=-1)
        ht_list = [h_de] * self.rnn_layers
        
        # Go Token (Zeros)
        go = torch.zeros((B, N, self.output_dim), device=x.device)
        
        out_preds = []
        
        for t in range(self.horizon):
             if self.use_STE:
                 if self.input_embedding_dim > 0:
                     go_emb = self.input_proj(go.unsqueeze(1)).squeeze(1) # (B, N, D)
                 else:
                     go_emb = go
                 
                 features_de = [go_emb]

                 if timestamp.shape[1] >= self.sample_len + self.horizon:
                      curr_t_idx = self.sample_len + t
                      tod_de = timestamp[:, curr_t_idx, 0].unsqueeze(-1).expand(-1, N) # (B, N)
                      tod_idx_de = (tod_de * self.TDAY).long().clamp(0, self.TDAY-1)
                      time_emb_de = self.time_embedding[tod_idx_de] # (B, N, d)
                      features_de.append(time_emb_de)
                 elif self.tod_embed_dim > 0:
                      # Falback if no timestamp
                      features_de.append(torch.zeros(B, N, self.tod_embed_dim, device=x.device))

                 if self.node_embedding_dim > 0:
                      features_de.append(self.node_embedding.unsqueeze(0).expand(B, -1, -1))
                      
                 go_combined = torch.cat(features_de, dim=-1)
                 
                 h_de, ht_list = self.decoder(go_combined, ht_list, supports_de)
             else:
                 h_de, ht_list = self.decoder(go, ht_list, supports_de)
                 
             pred = self.proj(h_de) # (B, N, out)
             out_preds.append(pred)
             go = pred
             
        output = torch.stack(out_preds, dim=1) # (B, Horizon, N, Out)
        
        return output, other_loss
