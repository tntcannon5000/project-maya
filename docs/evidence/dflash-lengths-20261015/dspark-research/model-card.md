---
library_name: speculators
base_model:
- zai-org/GLM-5.3-Flash
license: mit
tags:
- speculative-decoding
- dspark
- speculators
- long-context
- preview
---

# RedHatAI/GLM-5.3-Flash-speculator.dspark

> **Preview release:** This checkpoint is the end-of-epoch-2 snapshot from the completed three-epoch training run. Its configuration and serving behavior have been validated.

## Model Overview

- **Model Architecture:** `DSparkDraftModel`
  - **Target Model Architecture:** `Glm5NextForConditionalGeneration`
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Speculative Decoding Algorithm:** DSpark
  - **Draft Model:** 5-layer Qwen3-style backbone
  - **Maximum Draft Length:** 8 tokens
  - **Explicit Auxiliary Target Layers:** 20, 28, 32, 36, 40, and 44
  - **Implicit Final Target Layer:** 45
  - **Markov Head:** Vanilla, rank 256
  - **Confidence Head:** Enabled with Markov features
- **Configured Maximum Context Length:** 1,048,576 tokens
- **Release Date:** 2026-09-24
- **Model Developers:** Red Hat AI

This is a DSpark speculator model for [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). It was trained using the [Speculators](https://github.com/vllm-project/speculators) library (version `0.7.0.dev174`).

DSpark extends DFlash with a Markov head for modeling intra-block token dependencies and a confidence head for predicting per-position acceptance. The 5-layer draft model consumes six explicitly selected auxiliary hidden states from the target model, plus its implicit final hidden state, and proposes up to 8 tokens per decoding step.

## Training Details

This checkpoint was initialized from a common fresh 5-layer DSpark initialization and trained on a GLM-5.3-Flash-regenerated version of [Open PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend). Regeneration used temperature `1.0` with `low`, `high`, and `max` reasoning efforts.

The source JSONL contained 1,731,254 rows. After excluding 7,521 rows with empty loss masks, the prepared dataset contained 1,723,733 structurally valid examples, truncated to at most 8,192 tokens. Training used a 99/1 train-validation split. Hidden-state extraction ran across two GB300 trays, while a third tray provided four FSDP training ranks.

The preview checkpoint is from global step 270,979, at the end of epoch 2 of the completed three-epoch run. It uses the full verifier vocabulary, `sample_from_anchor=true`, and up to 512 anchors per sequence.

<details>
  <summary>Key training configuration</summary>

```yaml
train:
  speculator_type: dspark
  seed: 53
  verifier:
    verifier_name_or_path: zai-org/GLM-5.3-Flash
  draft:
    target_layer_ids: [20, 28, 32, 36, 40, 44]
    implicit_final_layer_id: 45
  data:
    total_seq_len: 8192
    train_data_ratio: 0.99
    max_anchors: 512
  loss:
    loss_fn: '{"ce":0.1,"tv":0.9}'
  optimizer:
    optimizer: muon
    lr: 0.001
    muon_lr: 0.001
  scheduler:
    scheduler_type: linear
  trainer:
    epochs: 3
    released_epoch: 2
    released_global_step: 270979
    checkpoint_freq: 0.05
    fsdp_shard: true
  dspark:
    block_size: 8
    sample_from_anchor: true
    markov_head_type: vanilla
    markov_rank: 256
    confidence_head_with_markov: true
```

Hidden-state extraction used vLLM `0.28.1rc1.dev580+g385dce36b`. Training used Speculators commit [`b9c51bc`](https://github.com/vllm-project/speculators/commit/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f).

</details>

## Model Specifications

| | |
|---|---|
| **Base Model** | `zai-org/GLM-5.3-Flash` |
| **Target Architecture** | `Glm5NextForConditionalGeneration` |
| **Draft Architecture** | `DSparkDraftModel` |
| **Format** | Safetensors, bfloat16 |
| **License** | MIT |
| **Draft Layers** | 5 |
| **Explicit Target Layer IDs** | 20, 28, 32, 36, 40, 44 |
| **Implicit Final Target Layer** | 45 |
| **Draft Vocabulary Size** | 154,880 |
| **Mask Token ID** | 154,856 |
| **Training Sequence Length** | 8,192 |
| **Configured Maximum Context Length** | 1,048,576 |
| **Maximum Anchors** | 512 |
| **Markov Head** | Vanilla, rank 256 |
| **Confidence Head** | Enabled with Markov features |
| **Released Checkpoint** | Epoch 2, global step 270,979 |
| **Training Hardware** | Three NVIDIA GB300 trays: two for extraction and one for four-rank FSDP training |

## Deployment

The checkpoint contains the complete draft backbone, Markov head, confidence head, and serving configuration. Auxiliary hidden-state extraction infrastructure is required for training only and is not needed to serve the finished speculator.

Use a vLLM build with compatible DSpark support and the Speculators plugin:

```bash
vllm serve zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --kv-cache-dtype fp8 \
  --speculative-config \
  '{"model":"RedHatAI/GLM-5.3-Flash-speculator.dspark-preview","num_speculative_tokens":8,"method":"dspark"}'
```

The preview checkpoint was successfully served with vLLM `0.28.1rc1.dev580+g385dce36b` and Speculators `0.7.0.dev174`. Compatibility with an unmodified stable vLLM release is not yet claimed.

## Acceptance Results

The following end-to-end results use the epoch-2 checkpoint with the `zai-org/GLM-5.3-Flash` verifier. Evaluation used temperature `0`, `top_p=1`, 8 draft tokens, tensor parallelism 4, a 16,384-token server context limit, and up to 200 requests per dataset.

| Dataset | Acceptance Length | Acceptance Rate | Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval | 4.74 | 46.74% | 85.5% | 70.1% | 57.1% | 46.2% | 37.7% | 31.0% | 25.4% | 20.9% |
| math_reasoning | 6.12 | 63.98% | 92.9% | 85.0% | 76.3% | 67.1% | 58.0% | 50.2% | 43.9% | 38.4% |
| qa | 3.11 | 26.39% | 73.1% | 50.1% | 33.3% | 22.0% | 14.3% | 9.0% | 5.8% | 3.5% |
| question | 3.19 | 27.33% | 73.0% | 49.6% | 33.3% | 22.8% | 15.8% | 10.9% | 7.7% | 5.5% |
| rag | 3.80 | 35.01% | 80.1% | 61.5% | 46.3% | 33.5% | 23.7% | 16.4% | 11.2% | 7.4% |
| summarization | 3.73 | 34.11% | 81.5% | 62.2% | 45.7% | 32.5% | 22.3% | 14.4% | 9.0% | 5.2% |
| tool_call | 3.40 | 30.03% | 76.8% | 55.2% | 38.9% | 26.5% | 17.9% | 12.0% | 7.8% | 5.1% |
| translation | 3.79 | 34.86% | 79.6% | 61.7% | 45.6% | 33.2% | 23.9% | 16.2% | 11.2% | 7.5% |
| writing | 3.13 | 26.59% | 72.1% | 48.6% | 32.5% | 21.7% | 14.9% | 10.4% | 7.3% | 5.2% |
| **Weighted aggregate** | **3.77** | **34.67%** | **78.3%** | — | — | — | — | — | — | **10.4%** |

## Citation

Please cite the target model and the Speculators project when using this checkpoint:

```bibtex
@software{speculators,
  title = {Speculators: Efficient Speculative Decoding Training and Inference},
  author = {vLLM Project Contributors},
  url = {https://github.com/vllm-project/speculators}
}
```
