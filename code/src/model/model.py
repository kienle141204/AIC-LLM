import torch 
from torch import nn
import numpy as np
from utils.utils import lap_eig, topological_sort
from typing import Optional, Tuple
from model.sandglassAttn import SAG, PerceiverSAG, SetTransformerSAG, PoolingSAG
from model.embedding import TimeEmbedding, NodeEmbedding
from model.tokenizer import AnchorDiffTokenizer, Time2Token, Node2Token
from model.adaptive_rl import AdaptiveSAGWrapper, create_adaptive_sag, AdaptiveTokenPolicy, RLTrainer

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
                 use_anchor_diff_token: int = 0,
                 use_diff: int = 0,
                 use_sep_token: bool = True,
                 use_sep2_token: bool = True,
                 use_task_token: bool = True,
                 use_context_token: bool = True,
                 use_quality_token: bool = True,
                 task_type: str = 'prediction',
                 t_dim: int = 64, trunc_k=16, wo_conloss=False,
                 use_adaptive_rl: bool = False, min_sag_tokens: int = 4) :
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
        self.adj_mx = adj_mx
        self.dis_mx = dis_mx
        self.use_diff = use_diff 
        self.use_sep_token = use_sep_token
        self.use_sep2_token = use_sep2_token
        self.use_task_token = use_task_token
        self.use_context_token = use_context_token
        self.use_quality_token = use_quality_token
        self.task_type = task_type
        self.use_adaptive_rl = use_adaptive_rl
        self.min_sag_tokens = min_sag_tokens

        self.topological_sort_node = True

        self.emb_dim = basemodel.dim
        tim_dim = t_dim*2     #day, week
        self.setadj(adj_mx,dis_mx)
        
        # separator token
        if self.use_sep_token:
            self.sep_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            nn.init.normal_(self.sep_token, std=0.02)
        
        # separator2 token
        if self.use_sep2_token:
            self.sep2_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            nn.init.normal_(self.sep2_token, std=0.02)
        
        # task token
        if self.use_task_token:
            self.task_token_forecast = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            nn.init.normal_(self.task_token_forecast, std=0.02)
        
        # context token
        if self.use_context_token:
            self.context_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            nn.init.normal_(self.context_token, std=0.02)
            # Context aggregation network
            self.context_aggregator = nn.Sequential(
                nn.Linear(sample_len * input_dim, self.emb_dim),
                nn.GELU(),
                nn.Linear(self.emb_dim, self.emb_dim),
                nn.LayerNorm(self.emb_dim)
            )
        
        # quality token (based on diff with anchor)
        if self.use_quality_token:
            self.quality_token_high = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            nn.init.normal_(self.quality_token_high, std=0.02)
            self.quality_token_low = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            nn.init.normal_(self.quality_token_low, std=0.02)
            # Input: diff features (mean_diff per sample)
            self.quality_scorer = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        self.time_tokenizer = Time2Token(
            sample_len=sample_len, 
            features=input_dim, 
            emb_dim=self.emb_dim, 
            tim_dim=tim_dim, 
            drop_out=dropout
        )

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
        
        # Sandglass Attn - Create base SAG first
        if self.use_sandglassAttn == 0:
            base_sag = SAG(sag_dim=sag_dim, 
                           sag_tokens=sag_tokens, 
                           emb_dim=self.emb_dim, 
                           sample_len=sample_len, 
                           features=input_dim,
                           dropout=dropout
                           )
        elif self.use_sandglassAttn == 1:
            base_sag = PerceiverSAG(sag_dim=sag_dim, 
                                    sag_tokens=sag_tokens, 
                                    emb_dim=self.emb_dim, 
                                    sample_len=sample_len, 
                                    features=input_dim,
                                    dropout=dropout
                                    )
        elif self.use_sandglassAttn == 2:
            base_sag = SetTransformerSAG(sag_dim=sag_dim, 
                                        sag_tokens=sag_tokens, 
                                        emb_dim=self.emb_dim, 
                                        sample_len=sample_len, 
                                        features=input_dim,
                                        dropout=dropout
                                        )
        elif self.use_sandglassAttn == 3:
            base_sag = PoolingSAG(sag_dim=sag_dim, 
                                  sag_tokens=sag_tokens, 
                                  emb_dim=self.emb_dim, 
                                  sample_len=sample_len,    
                                  features=input_dim,
                                  dropout=dropout
                                  )
        else:
            base_sag = SAG(sag_dim=sag_dim, 
                           sag_tokens=sag_tokens, 
                           emb_dim=self.emb_dim, 
                           sample_len=sample_len, 
                           features=input_dim,
                           dropout=dropout
                           )
        
        # Wrap with AdaptiveSAGWrapper if using RL
        if self.use_adaptive_rl:
            self.sag = AdaptiveSAGWrapper(
                sag_module=base_sag,
                min_tokens=min_sag_tokens,
                max_tokens=sag_tokens,
                emb_dim=self.emb_dim,
                use_rl=True
            )
            # Initialize RL trainer (will be set from outside during training)
            self.rl_trainer = None
            self.rl_info = {}
        else:
            self.sag = base_sag
            self.rl_trainer = None
            self.rl_info = {}

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
        if self.use_diff == 1:
            x_diff = x - xa
        else:
            x_diff = xa

        timestamp = timestamp[:, :self.sample_len, :]
        te = self.time_embedding(timestamp) 
        ne = self.node_embedding()

        special_tokens_list = []
        
        # task token
        if self.use_task_token:
            task_token = self.task_token_forecast.repeat(B, 1, 1)
            special_tokens_list.append(task_token)
        
        # quality token (based on diff with anchor)
        if self.use_quality_token:
            diff = (x - xa).abs().mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
            diff = diff.squeeze(-1)  # (B, 1)
            
            quality_score = self.quality_scorer(diff)  # (B, 1)
            
            quality_token = (quality_score.unsqueeze(-1) * self.quality_token_high + 
                           (1 - quality_score.unsqueeze(-1)) * self.quality_token_low)
            quality_token = quality_token.repeat(1, 1, 1)  # (B, 1, emb_dim)
            special_tokens_list.append(quality_token)
        
        # context token
        if self.use_context_token:
            x_context = x.mean(dim=1)  # (B, TF)
            context_embedding = self.context_aggregator(x_context)  # (B, emb_dim)
            context_token = self.context_token.repeat(B, 1, 1) + context_embedding.unsqueeze(1)
            special_tokens_list.append(context_token)

        # spatial tokenizer 
        spatial_tokens = self.node_tokenizer(x_spatial, te, ne)  # (B, N, emb_dim)
        if self.topological_sort_node:
            spatial_tokens = spatial_tokens[:, self.node_order, :]
        
        st_embedding = spatial_tokens
        s_num = N
        s_num = self.sag_tokens
        
        # Precoder
        if self.use_sandglassAttn:
            if self.use_adaptive_rl:
                # AdaptiveSAGWrapper returns (out, attn_weights, rl_info)
                st_embedding, attn_weights, self.rl_info = self.sag.encode(
                    st_embedding, 
                    deterministic=not self.training
                )
                # Update s_num based on actual tokens used
                s_num = self.rl_info.get('effective_tokens', self.sag_tokens)
            else:
                st_embedding, attn_weights = self.sag.encode(st_embedding)
        else:
            st_embedding, attn_weights = self.precoder(st_embedding)

        
        if self.use_sandglassAttn and not self.wo_conloss:
            if attn_weights is not None:
                scale = attn_weights.sum(dim=1)    #(B,N)

                sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],attn_weights)
                other_loss.append(-((sag_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10)

                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())
        
        # Time Tokenizer
        time_tokens = self.time_tokenizer(x, te)
        time_tokens_idx = st_embedding.shape[1]
        st_embedding = torch.concat((time_tokens, st_embedding), dim=1)

        if self.use_sep2_token:
            sep2_token = self.sep2_token.repeat(B, 1, 1)
            st_embedding = torch.concat((sep2_token, st_embedding), dim=1)  

        if self.use_anchor_diff_token == 1:
            ad_tokens = self.anchor_diff_tokenizer(x, xa, te)
            st_embedding = torch.concat((ad_tokens, st_embedding), dim=1)
        elif self.use_anchor_diff_token == 2:
            anchor_tokens = self.anchor_tokenizer(x_diff, te)
            st_embedding = torch.concat((anchor_tokens, st_embedding), dim=1)
        
        # Final sequence: [TASK] | [QUALITY] | [CONTEXT] | [SEP] | [ANCHOR] | [TIME] | [SPATIAL]
        sep = None
        if self.use_sep_token:
            sep = self.sep_token.repeat(B, 1, 1)
            st_embedding = torch.concat((sep, st_embedding), dim=1)
        
        if len(special_tokens_list) > 0:
            special_tokens = torch.concat(special_tokens_list, dim=1)  # (B, num_special, emb_dim)
            st_embedding = torch.concat([special_tokens, st_embedding], dim=1)
        
        if prompt_prefix is not None:
            prompt_len,_ = prompt_prefix.shape
            prompt_embedding = self.basemodel.getembedding(prompt_prefix).view(1,prompt_len,-1)
            prompt_embedding = prompt_embedding.repeat(B,1,1)
            st_embedding = torch.concat([prompt_embedding, st_embedding],dim=1)
        
        hidden_state = st_embedding

        hidden_state = self.basemodel(hidden_state)
        s_state = hidden_state[:, -s_num:, :]  

        # Decoder
        if self.use_sandglassAttn:
            s_state = self.sag.decode(s_state, spatial_tokens)
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
    
    # ============ RL-related methods ============
    
    def init_rl_trainer(self, lr: float = 1e-4, efficiency_coef: float = 0.1):
        """
        Initialize the RL trainer for adaptive token selection.
        
        Args:
            lr: Learning rate for RL policy
            efficiency_coef: Coefficient for efficiency bonus in reward
        """
        if self.use_adaptive_rl and hasattr(self.sag, 'token_policy'):
            self.rl_trainer = RLTrainer(
                policy=self.sag.token_policy,
                lr=lr,
                efficiency_coef=efficiency_coef
            )
    
    def get_rl_info(self) -> dict:
        """
        Get RL information from the last forward pass.
        
        Returns:
            Dictionary containing RL-related info (num_tokens, log_prob, value, entropy)
        """
        return self.rl_info
    
    def store_rl_transition(self, mae_loss: torch.Tensor, done: bool = False):
        """
        Store RL transition and compute reward.
        
        Args:
            mae_loss: MAE loss from prediction
            done: Whether this is the end of an episode
        """
        if self.rl_trainer is None or not self.use_adaptive_rl:
            return
            
        rl_info = self.get_rl_info()
        if rl_info.get('log_prob') is not None and rl_info.get('state') is not None:
            # Compute reward
            reward = self.rl_trainer.compute_reward(
                mae_loss=mae_loss,
                selected_tokens=rl_info['num_tokens'],
                max_tokens=self.sag_tokens
            )
            
            # Use state from rl_info (already computed during encode)
            state = rl_info['state']
            
            # Handle batch dimension - take mean over batch for single state
            if isinstance(state, torch.Tensor) and state.dim() > 1 and state.shape[0] > 1:
                state = state.mean(dim=0, keepdim=True)
            
            # Safely squeeze tensors - handle both float and integer types
            def safe_squeeze_float(t):
                """For float tensors like log_prob and value"""
                if isinstance(t, torch.Tensor):
                    if t.dim() > 0:
                        return t.float().mean()
                    return t
                return torch.tensor(float(t), device=state.device if isinstance(state, torch.Tensor) else 'cuda')
            
            def safe_squeeze_int(t):
                """For integer tensors like num_tokens (action)"""
                if isinstance(t, torch.Tensor):
                    if t.dim() > 0:
                        # Take first element or most common value for action
                        return t[0].float()  # Convert to float for storage
                    return t.float()
                return torch.tensor(float(t), device=state.device if isinstance(state, torch.Tensor) else 'cuda')
            
            # Store transition
            self.rl_trainer.store_transition(
                state=state.squeeze(0) if isinstance(state, torch.Tensor) and state.dim() > 1 else state,
                action=safe_squeeze_int(rl_info['num_tokens']),
                log_prob=safe_squeeze_float(rl_info['log_prob']),
                value=safe_squeeze_float(rl_info['value']),
                reward=reward,
                done=done
            )
    
    def update_rl_policy(self, epochs: int = 4) -> dict:
        """
        Update the RL policy using stored transitions.
        
        Args:
            epochs: Number of PPO epochs
            
        Returns:
            Dictionary of loss values from update
        """
        if self.rl_trainer is None:
            return {}
        return self.rl_trainer.update(epochs=epochs)
    
    def get_avg_tokens_used(self) -> float:
        """
        Get the average number of tokens used in adaptive mode.
        
        Returns:
            Average token count
        """
        if hasattr(self, 'rl_info') and 'num_tokens' in self.rl_info:
            tokens = self.rl_info['num_tokens']
            if isinstance(tokens, torch.Tensor):
                return tokens.float().mean().item()
            return float(tokens)
        return float(self.sag_tokens)