"""Main training script with profiling."""

import torch
from model import PolicyNetwork
from ppo import PPOTrainer, PPOConfig
from profiler import Profiler
import argparse


def train(config: PPOConfig, num_steps: int = 100, device: str = "cuda"):
    """Train RLHF/PPO pipeline with profiling."""
    
    # Initialize
    policy = PolicyNetwork(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8
    ).to(device)
    
    trainer = PPOTrainer(policy, config)
    profiler = Profiler(enable_gpu=(device == "cuda"))
    
    print(f"Starting training with {config.optimization_level} optimization")
    print(f"Device: {device}")
    print("-" * 60)
    
    profiler.start()
    
    for step in range(num_steps):
        profiler.step_start()
        
        # Generate rollout data
        batch_size = config.batch_size
        seq_length = config.seq_length
        
        states = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        actions = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        
        with torch.no_grad():
            logits, _ = policy(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            old_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Simulate rewards
        rewards = torch.randn(batch_size, seq_length).to(device)
        dones = torch.zeros(batch_size, seq_length).to(device)
        
        # PPO update
        metrics = trainer.update(states, actions, old_log_probs, rewards, dones)
        
        # Record metrics
        step_metrics = profiler.step_end(
            avg_reward=rewards.mean().item(),
            **metrics
        )
        
        if step % 10 == 0:
            print(f"Step {step:3d} | "
                  f"Loss: {metrics['total_loss']:.4f} | "
                  f"Reward: {rewards.mean():.4f} | "
                  f"Time: {step_metrics.compute_time_ms:.0f}ms")
    
    # Print summary
    print("-" * 60)
    print("Training completed!")
    summary = profiler.summary()
    print(f"Average compute time: {summary['avg_compute_time_ms']:.0f}ms")
    print(f"Average CPU usage: {summary['avg_cpu_percent']:.1f}%")
    print(f"Throughput: {summary['steps_per_sec']:.2f} steps/sec")
    
    bottlenecks = profiler.detect_bottlenecks()
    if bottlenecks:
        print("\nDetected bottlenecks:")
        for b in bottlenecks:
            print(f"  ⚠️  {b}")
    
    return profiler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization", type=str, default="baseline",
                       choices=["baseline", "basic", "advanced", "optimal"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    config = PPOConfig(optimization_level=args.optimization)
    profiler = train(config, num_steps=args.steps, device=args.device)