"""Opt-in smoke test for loading a tiny model and serving one completion."""

import gc
import os
import socket

import pytest
import torch

from utils import helper, vllm_utils


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1",
    reason="set RUN_E2E=1 to run model/vLLM end-to-end tests",
)
def test_tiny_model_loads_and_vllm_serves_completion() -> None:
    """Exercise model loading, vLLM startup, HTTP generation, and shutdown."""
    if not torch.cuda.is_available():
        pytest.skip("vLLM end-to-end test requires a CUDA GPU")

    model_id = os.environ.get("E2E_MODEL_ID", "HuggingFaceTB/SmolLM2-135M")
    gpu = int(os.environ.get("E2E_GPU", "0"))
    if gpu < 0 or gpu >= torch.cuda.device_count():
        pytest.skip(f"E2E_GPU={gpu} is not an available CUDA device")

    # Load through the project's public helper on CPU first. Keeping this tiny
    # model off the vLLM GPU avoids competing allocations during this smoke test.
    model, tokenizer = helper.get_model_and_tokenizer(
        model_id_or_dir=model_id,
        device="cpu",
    )
    assert next(model.parameters()).device.type == "cpu"
    assert tokenizer.eos_token_id is not None
    del model, tokenizer
    gc.collect()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = vllm_utils.VLLMServer(
        model_id=model_id,
        host="127.0.0.1",
        port=port,
        gpu=gpu,
        seed=42,
        gpu_memory_utilization=0.2,
        startup_timeout=180,
    )
    try:
        server.start()
        completions = server.generate_completions(
            prompts=["2 + 2 ="],
            sampling_params={
                "temperature": 0.0,
                "max_tokens": 8,
                "n": 1,
                "seed": 42,
            },
            batch_size=1,
        )
    finally:
        server.stop()

    assert len(completions) == 1
    assert isinstance(completions[0].text, str)
    assert completions[0].token_ids
    assert completions[0].finish_reason in {"stop", "length", None}
