"""
Adaptive Token Selection using Reinforcement Learning

This module implements an RL-based policy for dynamically selecting
the number of tokens in the Sandglass Attention mechanism.

Key concepts:
- Actor: Policy network that outputs probability distribution over token counts
- Critic: Value network that estimates expected reward
- Reward: -MAE + efficiency_bonus * (max_tokens - selected_tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AdaptiveTokenPolicy(nn.Module):
    """
    RL Policy for selecting the optimal number of tokens.
    
    Uses Actor-Critic architecture:
    - Actor: Outputs probability over possible token counts
    - Critic: Estimates value of current state
    
    Args:
        min_tokens: Minimum number of tokens to use
        max_tokens: Maximum number of tokens to use
        state_dim: Dimension of the state representation
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(self, min_tokens: int, max_tokens: int, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.num_actions = max_tokens - min_tokens + 1
        
        # Actor network - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_actions)
        )
        
        # Critic network - estimates state value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # State encoder - compresses input into state representation
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract state representation from input.
        
        Args:
            x: Input tensor of shape (B, N, D) or (B, D)
            
        Returns:
            State tensor of shape (B, state_dim)
        """
        if x.dim() == 3:
            # Pool over nodes: mean + max + std
            x_mean = x.mean(dim=1)
            x_max = x.max(dim=1)[0]
            x_std = x.std(dim=1)
            state = x_mean + x_max + x_std
        else:
            state = x
            
        return self.state_encoder(state)
    
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Args:
            state: State tensor of shape (B, state_dim)
            deterministic: If True, select action greedily; else sample
            
        Returns:
            action: Selected token count (B,)
            log_prob: Log probability of selected action (B,)
            value: Estimated value of state (B,)
            entropy: Entropy of action distribution (B,)
        """
        # Get action logits and value
        action_logits = self.actor(state)
        value = self.critic(state).squeeze(-1)
        
        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        # Entropy for exploration bonus
        entropy = dist.entropy()
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        
        # Convert action index to actual token count
        actual_tokens = action + self.min_tokens
        
        return actual_tokens, log_prob, value, entropy
    
    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions (used during PPO update).
        
        Args:
            state: State tensor (B, state_dim)
            actions: Actions to evaluate (B,) - token counts
            
        Returns:
            log_prob: Log probability of actions
            value: State values
            entropy: Distribution entropy
        """
        action_logits = self.actor(state)
        value = self.critic(state).squeeze(-1)
        
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        # Convert token counts back to action indices
        action_indices = actions - self.min_tokens
        
        log_prob = dist.log_prob(action_indices)
        entropy = dist.entropy()
        
        return log_prob, value, entropy


class RLTrainer:
    """
    Trainer for the adaptive token selection policy.
    Uses PPO (Proximal Policy Optimization) algorithm.
    
    Args:
        policy: AdaptiveTokenPolicy instance
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        efficiency_coef: Efficiency bonus coefficient
    """
    
    def __init__(
        self,
        policy: AdaptiveTokenPolicy,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        efficiency_coef: float = 0.1
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.efficiency_coef = efficiency_coef
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool = False
    ):
        """Store a transition in the experience buffer."""
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        
    def compute_reward(
        self,
        mae_loss: torch.Tensor,
        selected_tokens: torch.Tensor,
        max_tokens: int
    ) -> torch.Tensor:
        """
        Compute reward for the RL agent.
        
        Reward = -MAE + efficiency_bonus * normalized_savings
        
        Args:
            mae_loss: MAE prediction loss
            selected_tokens: Number of tokens used
            max_tokens: Maximum possible tokens
            
        Returns:
            Reward value
        """
        # Negative MAE (lower is better, so reward is higher for lower MAE)
        prediction_reward = -mae_loss.mean().item()
        
        # Efficiency bonus: reward for using fewer tokens
        tokens_saved = (max_tokens - selected_tokens.float().mean()) / max_tokens
        efficiency_reward = self.efficiency_coef * tokens_saved.item()
        
        total_reward = prediction_reward + efficiency_reward
        
        return total_reward
    
    def compute_gae(self, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        values = torch.stack(self.values)
        rewards = torch.tensor(self.rewards, device=values.device)
        dones = torch.tensor(self.dones, dtype=torch.float, device=values.device)
        
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.stack(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, epochs: int = 4) -> dict:
        """
        Perform PPO update.
        
        Args:
            epochs: Number of PPO epochs
            
        Returns:
            Dictionary of loss values
        """
        if len(self.states) == 0:
            return {}
        
        try:
            # Stack experiences - ensure consistent dimensions
            states = torch.stack(self.states)  # (T, state_dim)
            actions = torch.stack([a if a.dim() == 0 else a.squeeze() for a in self.actions])  # (T,)
            old_log_probs = torch.stack([lp if lp.dim() == 0 else lp.squeeze() for lp in self.log_probs])  # (T,)
            values_stacked = torch.stack([v if v.dim() == 0 else v.squeeze() for v in self.values])  # (T,)
            
            # Ensure states have correct dimensions for policy
            if states.dim() == 1:
                states = states.unsqueeze(0)
            
            # Compute GAE
            with torch.no_grad():
                # Get next value for GAE computation
                last_state = states[-1:] if states.dim() == 2 else states[-1:].unsqueeze(0)
                _, _, next_value, _ = self.policy(last_state)
                next_value = next_value.squeeze()
            
            # Compute advantages using stored values
            rewards = torch.tensor(self.rewards, device=states.device, dtype=torch.float32)
            dones = torch.tensor(self.dones, dtype=torch.float32, device=states.device)
            
            advantages = []
            gae = torch.tensor(0.0, device=states.device)
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_val = next_value
                else:
                    next_val = values_stacked[t + 1]
                    
                delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values_stacked[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages.insert(0, gae.clone())
                
            advantages = torch.stack(advantages)
            returns = advantages + values_stacked
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            
            for _ in range(epochs):
                # Evaluate current policy
                log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
            
        except Exception as e:
            # If there's an error, just clear buffer and return empty
            self.clear_buffer()
            return {'error': str(e)}
            
        # Clear buffer
        self.clear_buffer()
        
        return {
            'policy_loss': total_policy_loss / epochs,
            'value_loss': total_value_loss / epochs,
            'entropy': total_entropy / epochs
        }
    
    def clear_buffer(self):
        """Clear the experience buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []


class AdaptiveSAGWrapper(nn.Module):
    """
    Wrapper for SAG modules that adds adaptive token selection.
    
    This wrapper can be used with any SAG variant (SAG, PerceiverSAG, etc.)
    to make the number of tokens dynamic based on input complexity.
    
    Args:
        sag_module: The underlying SAG module
        min_tokens: Minimum number of tokens
        max_tokens: Maximum number of tokens (sag_module's sag_tokens)
        emb_dim: Embedding dimension
        use_rl: Whether to use RL for token selection
    """
    
    def __init__(
        self,
        sag_module: nn.Module,
        min_tokens: int,
        max_tokens: int,
        emb_dim: int,
        use_rl: bool = True
    ):
        super().__init__()
        
        self.sag = sag_module
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.use_rl = use_rl
        
        if use_rl:
            self.token_policy = AdaptiveTokenPolicy(
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                state_dim=emb_dim,
                hidden_dim=128
            )
        else:
            self.token_policy = None
            
        # Learnable token selector (fallback without RL)
        self.complexity_estimator = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    @property
    def sag_tokens(self):
        return self.max_tokens
        
    def estimate_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate input complexity to determine token count.
        
        Args:
            x: Input tensor (B, N, D)
            
        Returns:
            Complexity score (B,) in range [0, 1]
        """
        # Use variance and mean of input as complexity indicators
        x_pooled = x.mean(dim=1)  # (B, D)
        complexity = self.complexity_estimator(x_pooled).squeeze(-1)
        return complexity
    
    def select_tokens(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Select the number of tokens to use.
        
        Args:
            x: Input tensor (B, N, D)
            deterministic: Whether to use deterministic selection
            
        Returns:
            num_tokens: Selected token count (B,)
            log_prob: Log probability (if RL)
            value: Value estimate (if RL)
            entropy: Entropy (if RL)
        """
        if self.use_rl and self.token_policy is not None:
            state = self.token_policy.get_state(x)
            return self.token_policy(state, deterministic)
        else:
            # Fallback: use complexity estimator
            complexity = self.estimate_complexity(x)
            num_tokens = (
                self.min_tokens +
                (complexity * (self.max_tokens - self.min_tokens)).long()
            )
            return num_tokens, None, None, None
    
    def encode(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Encode with adaptive token selection.
        
        Args:
            x: Input tensor (B, N, D)
            deterministic: Whether to use deterministic token selection
            
        Returns:
            out: Encoded output (B, K, D) where K may vary
            attn_weights: Attention weights
            rl_info: Dictionary containing RL-related information
        """
        B = x.shape[0]
        
        # Get state for RL (before token selection)
        state = None
        if self.use_rl and self.token_policy is not None:
            state = self.token_policy.get_state(x)
        
        # Select number of tokens
        num_tokens, log_prob, value, entropy = self.select_tokens(x, deterministic)
        
        # For simplicity, use the same token count for all samples in batch
        # (Taking the max to ensure we can process all)
        if isinstance(num_tokens, torch.Tensor):
            effective_tokens = num_tokens.max().item()
        else:
            effective_tokens = num_tokens
            
        # Temporarily adjust sag_tokens in the underlying module
        original_tokens = self.sag.sag_tokens
        
        # Encode using original SAG
        out, attn_weights = self.sag.encode(x)
        
        # Truncate to effective tokens if needed
        if effective_tokens < self.max_tokens:
            out = out[:, :effective_tokens, :]
            if attn_weights is not None:
                attn_weights = attn_weights[:, :effective_tokens, :]
        
        # Collect RL info (including state for PPO update)
        rl_info = {
            'state': state,  # Add state for RL training
            'num_tokens': num_tokens,
            'log_prob': log_prob,
            'value': value,
            'entropy': entropy,
            'effective_tokens': effective_tokens
        }
        
        return out, attn_weights, rl_info
    
    def decode(self, hidden_state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Decode using the underlying SAG module.
        
        Args:
            hidden_state: Encoded hidden state (B, K, D)
            x: Original input for reconstruction (B, N, D)
            
        Returns:
            Decoded output (B, N, D)
        """
        # Pad hidden_state if needed (in case it was truncated)
        K = hidden_state.shape[1]
        if K < self.max_tokens:
            B, _, D = hidden_state.shape
            padding = torch.zeros(B, self.max_tokens - K, D, device=hidden_state.device)
            hidden_state = torch.cat([hidden_state, padding], dim=1)
            
        return self.sag.decode(hidden_state, x)


# Utility functions for integration

def create_adaptive_sag(
    sag_type: int,
    sag_dim: int,
    max_tokens: int,
    emb_dim: int,
    sample_len: int,
    features: int,
    dropout: float,
    min_tokens: int = 4,
    use_rl: bool = True
) -> AdaptiveSAGWrapper:
    """
    Factory function to create an AdaptiveSAGWrapper with the specified SAG type.
    
    Args:
        sag_type: 0=SAG, 1=PerceiverSAG, 2=SetTransformerSAG, 3=PoolingSAG
        sag_dim: SAG dimension
        max_tokens: Maximum number of tokens
        emb_dim: Embedding dimension
        sample_len: Sample length
        features: Number of features
        dropout: Dropout rate
        min_tokens: Minimum number of tokens
        use_rl: Whether to use RL-based selection
        
    Returns:
        AdaptiveSAGWrapper instance
    """
    from model.sandglassAttn import SAG, PerceiverSAG, SetTransformerSAG, PoolingSAG
    
    sag_classes = {
        0: SAG,
        1: PerceiverSAG,
        2: SetTransformerSAG,
        3: PoolingSAG
    }
    
    sag_class = sag_classes.get(sag_type, SAG)
    sag_module = sag_class(
        sag_dim=sag_dim,
        sag_tokens=max_tokens,
        emb_dim=emb_dim,
        sample_len=sample_len,
        features=features,
        dropout=dropout
    )
    
    return AdaptiveSAGWrapper(
        sag_module=sag_module,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        emb_dim=emb_dim,
        use_rl=use_rl
    )
