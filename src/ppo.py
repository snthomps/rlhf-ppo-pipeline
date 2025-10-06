"""PPO algorithm implementation with optimization levels."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    batch_size: int = 8
    seq_length: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    optimization_level: str = "baseline"  # baseline, basic, advanced, optimal


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    optimization: str = "baseline"
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: [batch_size, seq_length]
        values: [batch_size, seq_length]
        next_values: [batch_size, seq_length]
        dones: [batch_size, seq_length]
        gamma: discount factor
        gae_lambda: GAE lambda parameter
        optimization: optimization level
    
    Returns:
        advantages: [batch_size, seq_length]
    """
    if optimization == "baseline":
        # Baseline: Python loop (slow)
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(rewards.shape[1])):
            delta = rewards[:, t] + gamma * next_values[:, t] * (1 - dones[:, t]) - values[:, t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae
            
        return advantages
    
    else:
        # Optimized: Vectorized computation
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(rewards.shape[1])):
            last_gae = deltas[:, t] + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae
            
        return advantages


def ppo_loss(
    policy_logits: torch.Tensor,
    old_log_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PPO loss with clipping.
    
    Args:
        policy_logits: [batch_size, seq_length, vocab_size]
        old_log_probs: [batch_size, seq_length]
        actions: [batch_size, seq_length]
        advantages: [batch_size, seq_length]
        values: [batch_size, seq_length]
        returns: [batch_size, seq_length]
        clip_epsilon: clipping parameter
        value_coef: value loss coefficient
        entropy_coef: entropy bonus coefficient
    
    Returns:
        total_loss: scalar loss
        metrics: dictionary of metrics
    """
    # Policy loss with clipping
    log_probs = F.log_softmax(policy_logits, dim=-1)
    action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    ratio = torch.exp(action_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = F.mse_loss(values, returns)
    
    # Entropy bonus for exploration
    probs = F.softmax(policy_logits, dim=-1)
    entropy = -(probs * log_probs).sum(-1).mean()
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    # Metrics
    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "total_loss": total_loss.item(),
        "approx_kl": ((ratio - 1) - torch.log(ratio)).mean().item(),
        "clip_fraction": (torch.abs(ratio - 1) > clip_epsilon).float().mean().item()
    }
    
    return total_loss, metrics


class PPOTrainer:
    """PPO trainer with profiling."""
    
    def __init__(self, policy, config: PPOConfig, optimizer=None):
        self.policy = policy
        self.config = config
        self.optimizer = optimizer or torch.optim.Adam(policy.parameters(), lr=3e-4)
        
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> dict:
        """Perform single PPO update."""
        # Forward pass
        logits, values = self.policy(states)
        values = values.squeeze(-1)
        
        # Compute advantages
        with torch.no_grad():
            next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
            advantages = compute_gae(
                rewards, values, next_values, dones,
                self.config.gamma, self.config.gae_lambda,
                self.config.optimization_level
            )
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute loss
        total_loss, metrics = ppo_loss(
            logits, old_log_probs, actions, advantages, values, returns,
            self.config.clip_epsilon, self.config.value_coef, self.config.entropy_coef
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return metrics