# Benchmarking Results: RTX 4090 Optimized vs. Baseline

**Hardware:** 1x NVIDIA RTX 4090 (24GB GDDR6X)
**Model:** Llama-3-8B-Instruct
**Baseline:** PyTorch Eager (BF16)
**Optimized:** TensorRT-LLM (FP8 + PagedAttention)

| Metric | Baseline | Optimized | Improvement |
| :--- | :--- | :--- | :--- |
| **TTFT (Time to First Token)** | 115ms | **42ms** | 2.7x |
| **ITL (Inter-Token Latency)** | 22ms | **9ms** | 2.4x |
| **Throughput (Tokens/sec)** | 45 | **112** | 2.5x |
| **Max Concurrent Users** | 3 | **14** | 4.6x |

### Observations
The **2.5x Latency reduction** is directly attributed to the **FP8 Transformer Engine** and **Kernel Fusion**, which reduced the time the GPU spent waiting for memory transfers (Memory Bandwidth Bound).