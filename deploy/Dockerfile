FROM nvcr.io/nvidia/tritonserver:24.01-py3

# This Dockerfile builds a lightweight LangGraph API service on top of
# NVIDIA’s Triton server base image.  The container is designed for a
# consumer desktop with a single Ada Lovelace GPU (e.g., RTX 4090) and
# relies on a separate Triton container for inference.  Python
# dependencies include LangGraph for orchestration, Redis for the vector
# database client and Triton client libraries for interacting with the
# inference server.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir langgraph langchain redis tritonclient[http] fastapi uvicorn

# Copy the application code and model repository into the image.  The model
# repository is not executed here but is available for introspection or
# offline testing.  In production, the Triton server mounts the model
# repository directly.
WORKDIR /app
COPY langgraph_impl.py /app/langgraph_impl.py
COPY triton_model_repository /models

# Environment variables specifying the model repository path and disabling
# output buffering.  These variables are optional for the API container
# because inference is delegated to Triton.
ENV MODEL_REPOSITORY_PATH=/models
ENV PYTHONUNBUFFERED=1

# Expose the FastAPI port.  The actual port mapping is defined in
# docker-compose.yml.
EXPOSE 8001

# Launch the FastAPI service via Uvicorn.  Note that the Triton server
# runs in its own container.
CMD ["uvicorn", "langgraph_impl:app", "--host", "0.0.0.0", "--port", "8001"]