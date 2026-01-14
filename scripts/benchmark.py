"""
Conceptual benchmarking script for measuring latency and throughput on a
Triton TensorRT‑LLM deployment running on a single RTX 4090 (Ada Lovelace).

The script leverages the `tritonclient` HTTP API to send synthetic requests to
a running Triton server that serves a TensorRT‑LLM model compiled for
`sm_89` with FP8 quantization.  It records the latency of each request and
computes percentile metrics (P50, P95, P99) as well as requests per second
(RPS).  To demonstrate the **2.5× latency reduction** achieved by moving
from an eager‑mode PyTorch implementation to a compiled TensorRT‑LLM engine
optimized for Ada Lovelace, run this benchmark twice: once against the
baseline configuration (e.g., PyTorch or TensorRT‑LLM without dynamic
batching/quantization) and once against the optimized configuration.  Compare
the P99 latencies and RPS to quantify improvement.

"""

import time
from typing import List

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


def send_request(client: InferenceServerClient, model_name: str, prompt: str) -> float:
    """Send a single inference request and return the latency in seconds.

    Parameters
    ----------
    client : InferenceServerClient
        Connected Triton client.
    model_name : str
        The name of the model configured in `config.pbtxt` (e.g., "llm_model").
    prompt : str
        The text prompt to send to the model.  In a real implementation this
        should be tokenized and placed into an input tensor appropriate for
        TensorRT‑LLM.

    Returns
    -------
    float
        The time in seconds between request submission and response reception.
    """
    # Example: create a dummy input tensor.  The actual input tensor depends on
    # your tokenizer and model.  Here we use a bytes tensor for illustration.
    # Suppose the model expects a tensor named "INPUT0" of shape [1] and type
    # BYTES.  Consult your model repository for actual input names and types.
    input_tensor = InferInput("INPUT0", [1], "BYTES")
    input_tensor.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

    # Specify which output to request.  We assume the model returns a tensor
    # named "OUTPUT0" containing the generated tokens or text.
    output0 = InferRequestedOutput("OUTPUT0")

    # Submit the inference request and measure latency.
    start = time.time()
    _ = client.infer(model_name=model_name, inputs=[input_tensor], outputs=[output0])
    latency = time.time() - start
    return latency


def benchmark(client: InferenceServerClient, model_name: str, prompts: List[str]) -> None:
    """Run a benchmark over a list of prompts and print latency statistics.

    The function records the latency of each request, computes the mean,
    median, and tail latencies (P95, P99) and calculates requests per second.
    """
    latencies: List[float] = []
    for prompt in prompts:
        latencies.append(send_request(client, model_name, prompt))

    latencies_np = np.array(latencies)
    mean_latency = latencies_np.mean()
    p50 = np.percentile(latencies_np, 50)
    p95 = np.percentile(latencies_np, 95)
    p99 = np.percentile(latencies_np, 99)
    total_time = latencies_np.sum()
    rps = len(latencies) / total_time if total_time > 0 else 0.0

    print("Requests:", len(prompts))
    print(f"Mean latency: {mean_latency*1000:.1f} ms")
    print(f"P50 latency: {p50*1000:.1f} ms")
    print(f"P95 latency: {p95*1000:.1f} ms")
    print(f"P99 latency: {p99*1000:.1f} ms")
    print(f"Throughput (RPS): {rps:.2f}")


if __name__ == "__main__":
    # Connect to a local Triton server. 
    triton_client = InferenceServerClient(url="localhost:8000")


    sample_prompts = [f"Explain error {i}" for i in range(100)]

    # Run the benchmark.  Measure latencies and throughput.
    benchmark(triton_client, model_name="llm_model", prompts=sample_prompts)