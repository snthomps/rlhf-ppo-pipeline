# RLHF/PPO Training Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# RLHF/PPO Training Pipeline with Performance Profiling

A comprehensive demonstration of Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO), featuring advanced performance profiling and optimization techniques.

## 🎬 Demo

![RLHF Pipeline Demo](assets/rlhf-ppo-training-pipeline-web-demo.gif)
*Interactive training pipeline showing 4x speedup from optimization*

## 🎯 What Is This?

This is an **interactive training pipeline** that demonstrates how modern AI models like ChatGPT are fine-tuned using reinforcement learning. It showcases the complete RLHF workflow with real-time performance monitoring and optimization strategies.

### Key Components:
- **PPO Algorithm**: The core reinforcement learning algorithm used by OpenAI and Anthropic
- **Performance Profiler**: Detects bottlenecks in CPU, GPU, memory, and Python GIL contention
- **Optimization Showcase**: Demonstrates 4 levels of optimization with up to 4x speedup
- **Real-time Metrics**: Live visualization of training progress and resource utilization

## 🚀 What Does It Do?

### 1. **RLHF Training Simulation**
Simulates the process of training a language model to align with human preferences:
- Policy network generates text responses
- Reward model scores the quality of responses
- PPO optimizes the policy to maximize rewards
- Value function predicts expected future rewards

### 2. **Performance Profiling**
Tracks critical performance metrics during training:
- **GPU Utilization**: How efficiently the GPU is being used (target: >85%)
- **CPU Utilization**: CPU workload during training
- **Memory Usage**: RAM consumption in megabytes
- **Throughput**: Training samples processed per second
- **GIL Contention**: Python's Global Interpreter Lock bottleneck percentage
- **Compute Time**: Time per training step in milliseconds

### 3. **Bottleneck Detection**
Automatically identifies performance issues:
- **Low GPU Util**: Data loading or CPU preprocessing bottleneck
- **High GIL Contention**: Python threading issues limiting parallelism
- **Memory Leaks**: Growing memory usage indicating inefficient allocation
- **Poor Throughput**: Overall pipeline inefficiency

### 4. **Optimization Strategies**
Four progressive optimization levels with measurable speedups:

| Level | Speedup | Key Techniques |
|-------|---------|----------------|
| Baseline | 1.0x | No optimizations |
| Basic | 1.43x | Vectorization, memory pooling |
| Advanced | 2.5x | GPU acceleration, mixed precision |
| Optimal | 4.0x | Flash Attention, tensor parallelism |

## 🔧 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 RLHF Training Loop                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Policy Network (Transformer)                    │
│     ├─> Generate token probabilities                │
│     └─> Predict state values                        │
│                                                     │
│  2. Rollout Collection                              │
│     ├─> Sample actions from policy                  │
│     ├─> Compute rewards (from reward model)         │
│     └─> Store trajectories                          │
│                                                     │
│  3. Advantage Estimation (GAE)                      │
│     ├─> Calculate advantages                        │
│     └─> Normalize for stability                     │
│                                                     │
│  4. PPO Update                                      │
│     ├─> Clipped policy loss                         │
│     ├─> Value function loss                         │
│     ├─> Entropy bonus                               │
│     └─> KL divergence monitoring                    │
│                                                     │
│  5. Profiling & Metrics                             │
│     ├─> Track resource utilization                  │
│     ├─> Detect bottlenecks                          │
│     └─> Log performance metrics                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### PPO Algorithm Explained

**Proximal Policy Optimization** is the gold standard for RLHF because it's:
- **Stable**: Prevents catastrophic policy updates
- **Sample Efficient**: Reuses data multiple times
- **Simple**: Easy to implement and tune

#### Core PPO Components:

1. **Clipped Objective Function**
   ```python
   ratio = π_new(a|s) / π_old(a|s)
   L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
   ```
   - Prevents policy from changing too drastically
   - `ε` typically 0.2 (20% change limit)

2. **Generalized Advantage Estimation (GAE)**
   ```python
   A_t = Σ(γλ)^l * δ_{t+l}
   where δ_t = r_t + γV(s_{t+1}) - V(s_t)
   ```
   - Balances bias-variance tradeoff
   - `λ` controls how far we look ahead

3. **Value Function Loss**
   ```python
   L_value = (V_θ(s) - V_target)²
   ```
   - Helps predict expected future rewards
   - Reduces variance in advantage estimation

4. **Entropy Regularization**
   ```python
   H = -Σ π(a|s) log π(a|s)
   ```
   - Encourages exploration
   - Prevents premature convergence

### Optimization Techniques Explained

#### **Baseline (No Optimization)**
Standard Python implementation with sequential processing:
- Pure Python loops for advantage computation
- CPU-only tensor operations
- Synchronous data loading
- No memory optimizations
- **Result**: 45% GPU utilization, 25% GIL contention

#### **Basic Optimization (1.43x)**
```python
# Vectorized advantage computation
advantages = (rewards + gamma * next_values - values) + \
             gamma * lambda_ * (1 - dones) * last_gae
             
# In-place operations
tensor.add_(other_tensor)  # Instead of tensor = tensor + other
```
- NumPy/PyTorch vectorization
- Reduced Python overhead
- Memory pooling for tensors
- **Result**: 65% GPU utilization, 15% GIL contention

#### **Advanced Optimization (2.5x)**
```python
# Mixed precision training
with torch.cuda.amp.autocast():
    logits, values = policy(states)
    loss = compute_loss(...)
    
scaler.scale(loss).backward()
scaler.step(optimizer)

# Async data loading
dataloader = DataLoader(..., num_workers=4, pin_memory=True)
```
- GPU acceleration with CUDA
- FP16 mixed precision (2x memory savings)
- Asynchronous data loading (overlap I/O with compute)
- Gradient accumulation for larger effective batches
- **Result**: 85% GPU utilization, 8% GIL contention

#### **Optimal Optimization (4.0x)**
```python
# Flash Attention (memory-efficient attention)
from flash_attn import flash_attn_func
attn_output = flash_attn_func(q, k, v)

# Model parallelism
model = torch.nn.parallel.DistributedDataParallel(model)

# Custom fused kernels
@torch.jit.script
def fused_ppo_loss(ratio, advantage, clip_eps):
    return -torch.min(
        ratio * advantage,
        torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantage
    ).mean()
```
- Flash Attention (10x faster attention)
- Tensor/model parallelism across GPUs
- Compiled custom CUDA kernels
- Zero-copy memory transfers
- Fused operations (combine multiple ops into one kernel)
- **Result**: 92% GPU utilization, 3% GIL contention

## 📊 Performance Metrics Explained

### GPU Utilization
- **Target**: >85% for optimal training
- **Low (<60%)**: Indicates CPU bottleneck or data loading issues
- **High (>85%)**: GPU is fully utilized, compute-bound workload

### GIL Contention
Python's Global Interpreter Lock prevents true parallel execution:
- **High (>20%)**: Threading bottleneck, use multiprocessing instead
- **Low (<5%)**: Minimal Python overhead, well-optimized
- **Solutions**: Use compiled extensions, reduce Python loops, use C++/Rust bindings

### Memory Usage
- **Growing**: Potential memory leak, check tensor detachment
- **Stable**: Good memory management
- **Spiking**: Batch size too large or accumulating gradients

### Throughput
Samples processed per second:
- **Baseline**: ~8 samples/sec
- **Basic**: ~11 samples/sec (1.43x)
- **Advanced**: ~20 samples/sec (2.5x)
- **Optimal**: ~32 samples/sec (4.0x)

## 🎓 Key Concepts

### Why RLHF?
Traditional supervised learning trains on input-output pairs, but for tasks like generating helpful, harmless, and honest text, it's hard to define the "correct" output. RLHF instead:
1. Trains a **reward model** on human preferences
2. Uses RL (PPO) to optimize the policy to maximize reward
3. Results in models that align with human values

### Why PPO over other RL algorithms?
- **REINFORCE**: High variance, unstable
- **A3C/A2C**: Harder to tune, less sample efficient
- **TRPO**: Complex to implement, computationally expensive
- **PPO**: Simple, stable, sample efficient ✅

### Training Pipeline Flow
```
Input Text → Policy Network → Generated Response
                                      ↓
                              Reward Model Scores
                                      ↓
                            Compute Advantages (GAE)
                                      ↓
                              PPO Update Step
                                      ↓
                            Updated Policy Network
```

## 🔍 Common Bottlenecks & Solutions

| Bottleneck | Symptoms | Solution |
|------------|----------|----------|
| **Data Loading** | Low GPU util, high CPU | Async loading, more workers |
| **GIL Contention** | Multi-thread slowdown | Use multiprocessing or C extensions |
| **Memory Bandwidth** | Low GPU util with high memory | Reduce batch size, use mixed precision |
| **Small Batch Size** | Low throughput | Gradient accumulation |
| **Python Overhead** | High CPU, low GPU | Vectorize operations, JIT compile |
| **Attention Complexity** | O(n²) memory growth | Flash Attention, sparse attention |

## 📈 Expected Results

### Baseline Run
- GPU Utilization: ~45%
- Throughput: ~8 samples/sec
- GIL Contention: ~25%
- Training Time: 20 seconds (20 steps)

### Optimal Run
- GPU Utilization: ~92%
- Throughput: ~32 samples/sec
- GIL Contention: ~3%
- Training Time: 5 seconds (20 steps)
- **4x overall speedup!**

## 🛠️ Implementation Details

### Policy Network Architecture
```python
class PolicyNetwork(nn.Module):
    - Embedding layer (vocab_size → hidden_size)
    - Transformer encoder (4 layers, 8 heads)
    - Policy head (hidden_size → vocab_size)
    - Value head (hidden_size → 1)
```

### Hyperparameters
- **Batch Size**: 8 sequences
- **Sequence Length**: 128 tokens
- **Learning Rate**: 3e-4
- **Discount Factor (γ)**: 0.99
- **GAE Lambda (λ)**: 0.95
- **Clip Epsilon (ε)**: 0.2
- **Value Coefficient**: 0.5
- **Entropy Coefficient**: 0.01

## 🎯 Use Cases

1. **Learning RLHF**: Understand how ChatGPT-like models are trained
2. **Performance Optimization**: See real-world speedup from different techniques
3. **Bottleneck Analysis**: Learn to identify and fix training slowdowns
4. **Algorithm Comparison**: Compare different RL approaches (future work)

## 🚧 Limitations

This is a **demonstration** pipeline with simplifications:
- Simulated reward model (real RLHF uses trained reward networks)
- Small model size (real models have billions of parameters)
- Single GPU simulation (real training uses clusters)
- Synthetic data (real training uses human feedback datasets)

## 📚 Further Reading

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT paper)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO paper)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [High-level explanation of GAE](https://arxiv.org/abs/1506.02438)

## 🤝 Contributing

This is an educational demonstration. For production RLHF, consider:
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

## 📄 License

MIT License - Feel free to use for learning and education!

---

**Built to demonstrate the power of RLHF and the importance of performance optimization in modern AI training.**
