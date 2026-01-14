# Multi-Agent Code Assistant (RTX 4090 – Reference Architecture)

This repository presents a **reference architecture** for a **multi-agent LLM inference system** designed to run on a **single RTX 4090 (24GB, Ada Lovelace)** GPU.

The goal of this project is **not production deployment**, but to **demonstrate systems-level reasoning** around:
- LLM inference performance
- GPU memory constraints
- Agentic workflows (planner / retriever / verifier)
- Trade-offs between latency, throughput, and stability

The repository is structured to support **design discussion, benchmarking methodology, and architectural defense** in performance-focused AI infrastructure interviews.

---
<img width="8192" height="851" alt="multi-agent-system" src="https://github.com/user-attachments/assets/4b423226-4bbf-4a02-b6b9-1086334eb984" />



## Conceptual Architecture

Client / IDE  
→ LangGraph Orchestrator (Planner → Retriever → Verifier)  
→ Triton Inference Server (request routing, dynamic batching)  
→ TensorRT-LLM Engine (FP8, FlashAttention-2, paged KV cache)  
→ RTX 4090 GPU (sm_89, 24GB)

Responsibility boundaries:
- LangGraph: agent logic, control flow, and state transitions
- Triton: request scheduling, batching, and lifecycle management
- TensorRT-LLM: compiled execution graph and optimized kernels
- GPU: execution of the compiled inference plan

This separation mirrors how modern inference stacks are structured, even though this repository is intentionally simplified.

---

## Why RTX 4090 Is an Interesting Constraint

The RTX 4090 provides very high compute throughput but operates under:
- 24GB VRAM
- GDDR6X memory bandwidth (lower than HBM-based data center GPUs)

As a result:
- LLM decoding is memory-bandwidth bound
- KV-cache growth becomes the primary failure mode
- Agentic loops amplify memory pressure and fragmentation risk

This repository explores how these constraints influence system design choices.

---

## Core Design Concepts (High Level)

### TensorRT-LLM vs PyTorch Eager
- PyTorch eager executes operations one-by-one at runtime
- TensorRT-LLM compiles the full model graph ahead of time
- Compilation enables kernel fusion and memory planning
- This reduces kernel launch overhead and VRAM traffic

The project uses TensorRT-LLM to illustrate how compiled inference engines improve latency characteristics.

---

### FP8 Quantization (Ada Lovelace)
- Decoding repeatedly reads the KV cache
- FP8 reduces bytes moved per token relative to FP16/BF16
- Lower memory traffic improves effective GPU utilization
- FP8 is used here as a design choice, not a claim of universal applicability

---

### PagedAttention (Paged KV Cache)
- Agentic workflows create variable-length KV caches
- Naive contiguous allocation leads to fragmentation and OOM
- PagedAttention allocates KV memory in fixed-size blocks
- This enables reuse and prevents external fragmentation

In this reference architecture, paged KV caching is treated as mandatory for stable multi-agent execution on a 24GB GPU.

---

### Dynamic Batching (With Guardrails)
- Small preferred batch sizes: [4, 8, 16]
- Short queue delay to limit impact on interactivity
- Explicit VRAM headroom to avoid allocator pressure

The batching configuration is intentionally conservative to emphasize predictability over peak throughput.

---


## Benchmarking (Conceptual)

The included benchmark:
- Measures end-to-end request latency from the client perspective
- Computes P50 / P95 / P99 latency
- Reports single-stream requests per second (RPS)

The benchmark is intended for relative comparison:
- Baseline (e.g., PyTorch eager or unoptimized engine)
- Optimized configuration (TensorRT-LLM FP8)

It is not intended to represent:
- Multi-tenant production load
- Streaming TTFT / ITL analysis
- Token-normalized throughput measurements

Methodology and limitations are documented in TRADE-OFFS.md.

---

## Scope and Non-Goals

This repository is a representation of system design thinking.

It intentionally does not attempt to:
- Provide a production-ready deployment
- Cover multi-node or multi-GPU scaling
- Implement custom CUDA kernels
- Optimize for cost efficiency or autoscaling

The focus is on reasoning about inference behavior, not operational completeness.

---

## Intended Use

This repository is intended to support:
- Technical interviews
- Architecture discussions
- Performance trade-off explanations
- Reasoning about GPU-constrained inference systems
