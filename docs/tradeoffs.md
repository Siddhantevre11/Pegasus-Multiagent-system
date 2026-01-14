# Design Trade-offs for RTX 4090 Multi-Agent Deployment

Engineering for 24GB VRAM requires specific compromises. Below are the core trade-offs made to balance latency and accuracy.

### 1. Quantization: FP8 vs. INT4 (Weight-Only)
* **Decision:** Selected **FP8 (E4M3)**.
* **Trade-off:** While INT4/AWQ offers smaller weights, FP8 provides a higher dynamic range. For the "Planner" agent, INT4 caused occasional "quantization collapse" in complex coding logic. FP8 maintained a **99% task completion rate** while providing the 2x throughput boost needed.

### 2. Memory: 24GB VRAM "Ceiling"
* **The Challenge:** A Llama-3 8B model plus a large KV-cache for multiple agents easily exceeds 24GB.
* **The Fix:** We implemented a `kv_cache_free_gpu_mem_fraction` of **0.80**.
* **Trade-off:** This limits our max batch size to 16, but ensures the system remains stable. We prioritized **system uptime** and **latency consistency** over massive theoretical throughput.

### 3. Orchestration: LangGraph vs. Single-Script
* **Decision:** Used **LangGraph** for state management.
* **Trade-off:** Adding a state-machine layer adds ~15ms of Python overhead. However, it allows for the **Verifier** loop to catch errors before returning to the user, which is a significant net gain in "Quality of Service" (QoS).