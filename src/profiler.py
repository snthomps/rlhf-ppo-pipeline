"""Performance profiling utilities."""

import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class ProfileMetrics:
    """Performance metrics for a single step."""
    step: int
    timestamp: float
    compute_time_ms: float
    cpu_percent: float
    memory_mb: float
    gpu_util_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    throughput: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class Profiler:
    """Performance profiler for training pipelines."""
    
    def __init__(self, enable_gpu: bool = True):
        self.metrics: List[ProfileMetrics] = []
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self._start_time = None
        self._step_start = None
        self.current_step = 0
        
    def start(self):
        """Start profiling session."""
        self._start_time = time.perf_counter()
        self.metrics = []
        self.current_step = 0
        
    def step_start(self):
        """Mark start of training step."""
        self._step_start = time.perf_counter()
        
    def step_end(self, **custom_metrics):
        """Record metrics for completed step."""
        if self._step_start is None:
            raise RuntimeError("Must call step_start() before step_end()")
            
        compute_time = (time.perf_counter() - self._step_start) * 1000  # ms
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.01)
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # GPU metrics
        gpu_util = None
        gpu_memory = None
        if self.enable_gpu:
            try:
                gpu_util = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            except:
                pass
        
        metrics = ProfileMetrics(
            step=self.current_step,
            timestamp=time.perf_counter() - self._start_time,
            compute_time_ms=compute_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_util_percent=gpu_util,
            gpu_memory_mb=gpu_memory,
            custom_metrics=custom_metrics
        )
        
        self.metrics.append(metrics)
        self.current_step += 1
        self._step_start = None
        
        return metrics
    
    def summary(self) -> Dict[str, float]:
        """Compute summary statistics."""
        if not self.metrics:
            return {}
            
        return {
            "avg_compute_time_ms": sum(m.compute_time_ms for m in self.metrics) / len(self.metrics),
            "avg_cpu_percent": sum(m.cpu_percent for m in self.metrics) / len(self.metrics),
            "avg_memory_mb": sum(m.memory_mb for m in self.metrics) / len(self.metrics),
            "total_time_s": self.metrics[-1].timestamp,
            "steps_per_sec": len(self.metrics) / self.metrics[-1].timestamp if self.metrics[-1].timestamp > 0 else 0
        }
    
    def detect_bottlenecks(self) -> List[str]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        summary = self.summary()
        
        if self.enable_gpu and self.metrics[0].gpu_util_percent is not None:
            avg_gpu = sum(m.gpu_util_percent for m in self.metrics if m.gpu_util_percent) / len(self.metrics)
            if avg_gpu < 60:
                bottlenecks.append(f"Low GPU utilization ({avg_gpu:.1f}%) - possible CPU/data loading bottleneck")
        
        if summary.get("avg_cpu_percent", 0) > 90:
            bottlenecks.append(f"High CPU usage ({summary['avg_cpu_percent']:.1f}%) - possible GIL contention")
        
        # Check for memory growth
        if len(self.metrics) > 10:
            early_mem = sum(m.memory_mb for m in self.metrics[:5]) / 5
            late_mem = sum(m.memory_mb for m in self.metrics[-5:]) / 5
            if late_mem > early_mem * 1.2:
                bottlenecks.append(f"Memory growth detected ({early_mem:.0f}MB → {late_mem:.0f}MB) - possible memory leak")
        
        return bottlenecks