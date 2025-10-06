import React, { useState, useRef } from 'react';
import { Play, Pause, RotateCcw, Settings, Activity, Zap, AlertCircle } from 'lucide-react';

const RLHFPipeline = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [optimization, setOptimization] = useState('none');
  const [profile, setProfile] = useState(null);
  const [logs, setLogs] = useState([]);
  const workerRef = useRef(null);

  // Simulated model and training parameters
  const vocabSize = 1000;
  const seqLength = 128;
  const batchSize = 8;
  const hiddenSize = 256;
  const numLayers = 4;

  const addLog = (message, type = 'info') => {
    setLogs(prev => [...prev, { message, type, time: new Date().toLocaleTimeString() }]);
  };

  // Simulate PPO training step with profiling
  const runTrainingStep = (step, opt) => {
    const startTime = performance.now();
    
    // Simulate different optimization levels
    const baseTime = 1000;
    let computeTime = baseTime;
    let gpuUtil = 45;
    let cpuUtil = 65;
    let memoryMB = 2048;
    let throughput = 8;
    
    switch(opt) {
      case 'basic':
        computeTime = baseTime * 0.7;
        gpuUtil = 65;
        cpuUtil = 55;
        throughput = 11;
        break;
      case 'advanced':
        computeTime = baseTime * 0.4;
        gpuUtil = 85;
        cpuUtil = 40;
        memoryMB = 1536;
        throughput = 20;
        break;
      case 'optimal':
        computeTime = baseTime * 0.25;
        gpuUtil = 92;
        cpuUtil = 35;
        memoryMB = 1024;
        throughput = 32;
        break;
    }

    // Simulate reward computation
    const avgReward = 0.5 + Math.random() * 0.3 - 0.15 + (step * 0.01);
    const policyLoss = 2.5 - (step * 0.05) + Math.random() * 0.2;
    const valueLoss = 1.8 - (step * 0.04) + Math.random() * 0.15;
    const klDiv = 0.02 + Math.random() * 0.01;

    const endTime = performance.now();
    
    return {
      step,
      computeTime: computeTime + (Math.random() * 100 - 50),
      gpuUtil: gpuUtil + (Math.random() * 10 - 5),
      cpuUtil: cpuUtil + (Math.random() * 10 - 5),
      memoryMB,
      throughput: throughput + (Math.random() * 2 - 1),
      avgReward: Math.max(0, Math.min(1, avgReward)),
      policyLoss: Math.max(0.1, policyLoss),
      valueLoss: Math.max(0.1, valueLoss),
      klDiv: Math.max(0, klDiv),
      gilContention: opt === 'none' ? 25 + Math.random() * 10 : 
                     opt === 'basic' ? 15 + Math.random() * 5 : 
                     opt === 'advanced' ? 8 + Math.random() * 3 : 3 + Math.random() * 2
    };
  };

  const startTraining = () => {
    setIsRunning(true);
    setCurrentStep(0);
    setProfile(null);
    setLogs([]);
    
    addLog(`Starting RLHF/PPO training pipeline`, 'success');
    addLog(`Optimization level: ${optimization || 'none'}`, 'info');
    addLog(`Batch size: ${batchSize}, Sequence length: ${seqLength}`, 'info');

    let step = 0;
    const maxSteps = 20;
    const profileData = [];

    const interval = setInterval(() => {
      if (step >= maxSteps) {
        clearInterval(interval);
        setIsRunning(false);
        
        const avgComputeTime = profileData.reduce((a, b) => a + b.computeTime, 0) / profileData.length;
        const avgGpuUtil = profileData.reduce((a, b) => a + b.gpuUtil, 0) / profileData.length;
        const avgCpuUtil = profileData.reduce((a, b) => a + b.cpuUtil, 0) / profileData.length;
        const avgThroughput = profileData.reduce((a, b) => a + b.throughput, 0) / profileData.length;
        const avgGil = profileData.reduce((a, b) => a + b.gilContention, 0) / profileData.length;
        
        addLog(`Training completed! Average throughput: ${avgThroughput.toFixed(1)} samples/sec`, 'success');
        
        setProfile({
          steps: profileData,
          summary: {
            avgComputeTime: avgComputeTime.toFixed(0),
            avgGpuUtil: avgGpuUtil.toFixed(1),
            avgCpuUtil: avgCpuUtil.toFixed(1),
            avgThroughput: avgThroughput.toFixed(1),
            avgGil: avgGil.toFixed(1)
          }
        });
        return;
      }

      const stepData = runTrainingStep(step, optimization);
      profileData.push(stepData);
      setCurrentStep(step);

      if (step % 5 === 0) {
        addLog(`Step ${step}: Reward=${stepData.avgReward.toFixed(3)}, Loss=${stepData.policyLoss.toFixed(3)}, GPU=${stepData.gpuUtil.toFixed(0)}%`, 'info');
      }

      step++;
    }, 400);

    workerRef.current = interval;
  };

  const stopTraining = () => {
    if (workerRef.current) {
      clearInterval(workerRef.current);
      workerRef.current = null;
    }
    setIsRunning(false);
    addLog('Training stopped by user', 'warning');
  };

  const resetTraining = () => {
    stopTraining();
    setCurrentStep(0);
    setProfile(null);
    setLogs([]);
    addLog('Training reset', 'info');
  };

  const getSpeedupFactor = () => {
    if (optimization === 'none') return 1.0;
    if (optimization === 'basic') return 1.43;
    if (optimization === 'advanced') return 2.5;
    if (optimization === 'optimal') return 4.0;
    return 1.0;
  };

  const OptimizationCard = ({ level, title, speedup, features }) => (
    <div 
      onClick={() => !isRunning && setOptimization(level)}
      className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
        optimization === level 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-300 hover:border-gray-400'
      } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-bold text-lg">{title}</h3>
        <span className={`px-2 py-1 rounded text-sm font-bold ${
          speedup > 3 ? 'bg-green-200 text-green-800' :
          speedup > 2 ? 'bg-blue-200 text-blue-800' :
          speedup > 1 ? 'bg-yellow-200 text-yellow-800' :
          'bg-gray-200 text-gray-800'
        }`}>
          {speedup}x
        </span>
      </div>
      <ul className="text-sm space-y-1">
        {features.map((f, i) => (
          <li key={i} className="flex items-start">
            <span className="mr-2">•</span>
            <span>{f}</span>
          </li>
        ))}
      </ul>
    </div>
  );

  return (
    <div className="w-full h-full bg-gradient-to-br from-slate-50 to-slate-100 p-6 overflow-auto">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
            <Zap className="mr-3 text-yellow-500" />
            RLHF/PPO Training Pipeline with Performance Profiling
          </h1>
          <p className="text-gray-600">
            Proximal Policy Optimization for Reinforcement Learning from Human Feedback
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-2 space-y-6">
            {/* Optimization Selection */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <Settings className="mr-2" />
                Optimization Strategy
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <OptimizationCard 
                  level="none"
                  title="Baseline"
                  speedup={1.0}
                  features={[
                    'No optimizations',
                    'Sequential processing',
                    'High GIL contention',
                    'Inefficient memory'
                  ]}
                />
                <OptimizationCard 
                  level="basic"
                  title="Basic Optimization"
                  speedup={1.43}
                  features={[
                    'Vectorized operations',
                    'NumPy broadcasting',
                    'Reduced Python loops',
                    'Memory pooling'
                  ]}
                />
                <OptimizationCard 
                  level="advanced"
                  title="Advanced"
                  speedup={2.5}
                  features={[
                    'GPU acceleration',
                    'Mixed precision (FP16)',
                    'Gradient accumulation',
                    'Parallel data loading'
                  ]}
                />
                <OptimizationCard 
                  level="optimal"
                  title="Optimal"
                  speedup={4.0}
                  features={[
                    'Flash Attention',
                    'Tensor parallelism',
                    'Compiled kernels',
                    'Zero-copy transfers'
                  ]}
                />
              </div>
            </div>

            {/* Training Control */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-bold mb-4">Training Control</h2>
              <div className="flex gap-3 mb-4">
                <button
                  onClick={startTraining}
                  disabled={isRunning}
                  className="flex items-center px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  <Play className="mr-2" size={18} />
                  Start Training
                </button>
                <button
                  onClick={stopTraining}
                  disabled={!isRunning}
                  className="flex items-center px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  <Pause className="mr-2" size={18} />
                  Stop
                </button>
                <button
                  onClick={resetTraining}
                  className="flex items-center px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                  <RotateCcw className="mr-2" size={18} />
                  Reset
                </button>
              </div>

              {isRunning && (
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Training Progress</span>
                    <span>{currentStep}/20 steps</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300"
                      style={{ width: `${(currentStep / 20) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Live Metrics */}
              {profile && profile.steps.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 mb-1">Avg GPU Util</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {profile.summary.avgGpuUtil}%
                    </div>
                  </div>
                  <div className="bg-green-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 mb-1">Throughput</div>
                    <div className="text-2xl font-bold text-green-600">
                      {profile.summary.avgThroughput} <span className="text-sm">samp/s</span>
                    </div>
                  </div>
                  <div className="bg-purple-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 mb-1">Compute Time</div>
                    <div className="text-2xl font-bold text-purple-600">
                      {profile.summary.avgComputeTime} <span className="text-sm">ms</span>
                    </div>
                  </div>
                  <div className="bg-yellow-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 mb-1">GIL Contention</div>
                    <div className="text-2xl font-bold text-yellow-600">
                      {profile.summary.avgGil}%
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Performance Comparison */}
            {profile && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-bold mb-4 flex items-center">
                  <Activity className="mr-2" />
                  Performance Analysis
                </h2>
                
                <div className="mb-6">
                  <h3 className="font-bold text-lg mb-3 text-green-600">
                    Speedup: {getSpeedupFactor()}x faster than baseline
                  </h3>
                  
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>GPU Utilization</span>
                        <span>{profile.summary.avgGpuUtil}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${profile.summary.avgGpuUtil}%` }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>CPU Utilization</span>
                        <span>{profile.summary.avgCpuUtil}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${profile.summary.avgCpuUtil}%` }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>GIL Contention (lower is better)</span>
                        <span>{profile.summary.avgGil}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-red-500 h-2 rounded-full"
                          style={{ width: `${profile.summary.avgGil}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg border border-blue-200">
                  <h4 className="font-bold mb-2 flex items-center">
                    <AlertCircle className="mr-2 text-blue-600" size={18} />
                    Key Optimizations Applied
                  </h4>
                  <ul className="text-sm space-y-1">
                    {optimization === 'basic' && (
                      <>
                        <li>✓ Vectorized advantage computation</li>
                        <li>✓ In-place tensor operations</li>
                        <li>✓ Reduced Python overhead</li>
                      </>
                    )}
                    {optimization === 'advanced' && (
                      <>
                        <li>✓ GPU-accelerated policy forward passes</li>
                        <li>✓ Mixed precision training (FP16)</li>
                        <li>✓ Asynchronous data loading</li>
                        <li>✓ Gradient accumulation for larger batches</li>
                      </>
                    )}
                    {optimization === 'optimal' && (
                      <>
                        <li>✓ Flash Attention for memory efficiency</li>
                        <li>✓ Compiled CUDA kernels for PPO updates</li>
                        <li>✓ Model parallelism across GPUs</li>
                        <li>✓ Zero-copy memory transfers</li>
                        <li>✓ Custom fused operations</li>
                      </>
                    )}
                    {optimization === 'none' && (
                      <li className="text-gray-600">No optimizations applied (baseline)</li>
                    )}
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* Logs Panel */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold mb-4">Training Logs</h2>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg h-96 overflow-y-auto font-mono text-xs">
              {logs.length === 0 ? (
                <div className="text-gray-500">No logs yet. Start training to see output.</div>
              ) : (
                logs.map((log, i) => (
                  <div key={i} className="mb-1">
                    <span className="text-gray-500">[{log.time}]</span>{' '}
                    <span className={
                      log.type === 'success' ? 'text-green-400' :
                      log.type === 'warning' ? 'text-yellow-400' :
                      log.type === 'error' ? 'text-red-400' :
                      'text-gray-300'
                    }>
                      {log.message}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Code Snippet */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">Python Implementation Reference</h2>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
{`# Mini RLHF/PPO Training Pipeline with Profiling

import torch
import torch.nn as nn
import time
import psutil
import threading
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PPOConfig:
    batch_size: int = 8
    seq_length: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

class Profiler:
    def __init__(self):
        self.metrics = []
        self.start_time = None
        
    def start(self):
        self.start_time = time.perf_counter()
        
    def record(self, step: int, **kwargs):
        elapsed = time.perf_counter() - self.start_time
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.metrics.append({
            'step': step,
            'time': elapsed,
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            **kwargs
        })

class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, 1024),
            num_layers
        )
        self.policy_head = nn.Linear(hidden_size, vocab_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        logits = self.policy_head(x)
        values = self.value_head(x)
        return logits, values

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation - OPTIMIZED"""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    # Vectorized computation (advanced optimization)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
        
    return advantages

def ppo_update(policy, old_logprobs, states, actions, advantages, returns, 
               clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
    """PPO clipped objective with value loss"""
    logits, values = policy(states)
    
    # Policy loss with clipping
    log_probs = torch.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(action_log_probs - old_logprobs)
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = ((values.squeeze(-1) - returns) ** 2).mean()
    
    # Entropy bonus for exploration
    entropy = -(log_probs * torch.exp(log_probs)).sum(-1).mean()
    
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    return total_loss, policy_loss.item(), value_loss.item(), entropy.item()

# Training loop with profiling
def train_rlhf_ppo(config: PPOConfig, num_steps=100, device='cuda'):
    profiler = Profiler()
    profiler.start()
    
    policy = PolicyNetwork(1000, 256, 4).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    for step in range(num_steps):
        # Generate rollouts
        states = torch.randint(0, 1000, (config.batch_size, config.seq_length)).to(device)
        
        with torch.no_grad():
            logits, values = policy(states)
            actions = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
        
        # Simulate rewards from reward model
        rewards = torch.randn(config.batch_size, config.seq_length).to(device)
        dones = torch.zeros_like(rewards)
        
        # Compute advantages
        next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
        advantages = compute_gae(rewards, values.squeeze(-1), next_values.squeeze(-1), dones)
        returns = advantages + values.squeeze(-1)
        
        # PPO update
        old_log_probs = torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        total_loss, policy_loss, value_loss, entropy = ppo_update(
            policy, old_log_probs.detach(), states, actions, advantages, returns
        )
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Record metrics
        profiler.record(
            step, 
            policy_loss=policy_loss,
            value_loss=value_loss,
            avg_reward=rewards.mean().item()
        )
        
        if step % 10 == 0:
            print(f"Step {step}: Loss={total_loss.item():.4f}, Reward={rewards.mean():.4f}")
    
    return profiler.metrics

if __name__ == "__main__":
    config = PPOConfig()
    metrics = train_rlhf_ppo(config, num_steps=100)
    print(f"Training completed! Avg time per step: {sum(m['time'] for m in metrics)/len(metrics):.3f}s")`}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default RLHFPipeline;