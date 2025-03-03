---
title: "Flops Profiler"
excerpt: "Measure the parameters, latency, and floating-point operations of your model"
tags: profiling performance-tuning
---

In this tutorial, we introduce the DeepSpeed flops profiler and provide examples of its usage.

  - [Overview](#overview)
  - [Supported Models](#supported-models)
  - [Multi-GPU, Multi-node Runs](#multi-gpu-multi-node-runs)
  - [Usage](#usage)

## Overview

Effective use of hardware resources is critical to good performance, but performance inefficiency in existing implementations for large-scale model training and inference are often hard to spot and attribute to specific module components. DeepSpeed Flops Profiler helps users easily measure both the model training/inference speed (latency, throughput) and efficiency (floating-point operations per second, i.e., FLOPS) of a model and its submodules, with an eye towards eliminating inefficiencies in existing implementations.

The output profile is computed for each batch of input and printed to the `stdout`. For each module, the measured profile is annotated after the name and is listed in the order of `number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency of the module, percentage of the total latency, floating point operations per second (FLOPS)`. Note that the number of floating point operations is estimated as `2 * MACs` in the profiler (each MAC operation is counted as 2 floating point operations).

Below is an example output for LeNet5 with batch size 1024:

```shell
-------------------------- DeepSpeed Flops Profiler --------------------------
Profile Summary at step 10:
Notations:
data parallel size (dp_size), model paralel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

world size:                                                   1
data parallel size:                                           1
model paralel size:                                           1
batch size per GPU:                                           80
params per gpu:                                               336.23 M
params of model = params per GPU * mp_size:                   336.23 M
fwd MACs per GPU:                                             3139.93 G
fwd flops per GPU:                                            6279.86 G
fwd flops of model = fwd flops per GPU * mp_size:             6279.86 G
fwd latency:                                                  76.67 ms
bwd latency:                                                  108.02 ms
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:          81.9 TFLOPS
bwd FLOPS per GPU = 2 * fwd flops per GPU / bwd latency:      116.27 TFLOPS
fwd+bwd FLOPS per GPU = 3 * fwd flops per GPU / (fwd+bwd latency):   102.0 TFLOPS
step latency:                                                 34.09 us
iter latency:                                                 184.73 ms
samples/second:                                               433.07

------------------------------ Detailed Profile ------------------------------
Each module profile is listed after its name in the following order:
number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency, percentage of total latency, number of floating point operations per second (FLOPS, computed as 2 * MACs / latency).
Note:
1. A module can have torch.nn.functional (e.g. to compute logits) along with submodules, thus making the difference between the parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.

LeNet5(
  61.71 k, 100.00% Params, 439.56 MMACs, 100.00% MACs, 25.7 ms, 100.00% latency, 34.2 GFLOPS,
  (feature_extractor): Sequential(
    50.69 k, 82.15% Params, 428.37 MMACs, 97.45% MACs, 20.12 ms, 78.27% latency, 42.59 GFLOPS,
    (0): Conv2d(156, 0.25% Params, 125.24 MMACs, 28.49% MACs, 9.8 ms, 38.12% latency, 25.56 GFLOPS, 1, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 2.85 ms, 11.08% latency, 0.0 FLOPS, )
    (2): AvgPool2d(0, 0.00% Params, 4.82 MMACs, 1.10% MACs, 4.01 ms, 15.59% latency, 2.4 GFLOPS, kernel_size=2, stride=2, padding=0)
    (3): Conv2d(2.42 k, 3.92% Params, 247.4 MMACs, 56.28% MACs, 924.83 us, 3.60% latency, 535.02 GFLOPS, 6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 672.1 us, 2.62% latency, 0.0 FLOPS, )
    (5): AvgPool2d(0, 0.00% Params, 1.64 MMACs, 0.37% MACs, 1.01 ms, 3.95% latency, 3.23 GFLOPS, kernel_size=2, stride=2, padding=0)
    (6): Conv2d(48.12 k, 77.98% Params, 49.27 MMACs, 11.21% MACs, 647.31 us, 2.52% latency, 152.25 GFLOPS, 16, 120, kernel_size=(5, 5), stride=(1, 1))
    (7): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 82.02 us, 0.32% latency, 0.0 FLOPS, )
  )
  (classifier): Sequential(
    11.01 k, 17.85% Params, 11.18 MMACs, 2.54% MACs, 5.41 ms, 21.06% latency, 4.13 GFLOPS,
    (0): Linear(10.16 k, 16.47% Params, 10.32 MMACs, 2.35% MACs, 2.47 ms, 9.60% latency, 8.37 GFLOPS, in_features=120, out_features=84, bias=True)
    (1): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 90.12 us, 0.35% latency, 0.0 FLOPS, )
    (2): Linear(850, 1.38% Params, 860.16 KMACs, 0.20% MACs, 2.8 ms, 10.91% latency, 613.62 MFLOPS, in_features=84, out_features=10, bias=True)
  )
)
------------------------------------------------------------------------------
```

In the summary profile, the DeepSpeed Flops Profiler outputs the number of parameters, floating-point operations (flops), FLOPS, latency, and throughput in samples/second of the model. This profile shows how much performance gap (compared to the peak hardware performance) the current model execution has and helps users tune the training or inference setup (e.g., hyperparameters, data parallelism, model parallelism, system configurations, etc.) for better performance.

The DeepSpeed Flops Profiler also measures significant modules at different model depths (aggregated profile) and module-specific profile in the model architecture (detailed profile). Using these profiles, DeepSpeed users can understand how each layer or submodule contributes to the overall model complexity/performance. Then users can adjust or refactor the model design to improve performance. For example, using the profiler, DeepSpeed users can quantitatively tell if stacking smaller layers is lighter or more performant than having bigger ones. The aggregated and detailed profiles also allow users to quickly identify bottleneck modules. In the BERT-Large example above, using the DeepSpeed Flops Profiler, we find that BertLayer is the most significant layer and contains quite a few dropout, softmax, and layer norm along with linear modules. These modules are not heavy in flops and would trigger many GPU kernel invocations and create excessive read/write requests to memory. The pattern shown in the detailed profile suggests this is a perfect match for kernel fusion, and we developed fused transformer-kernels to reduce data movement (see [DeepSpeedBert](/tutorials/bert-pretraining)). After applying our optimizations, we see a 25% improvement in FLOPS per GPU and overall training samples/second in the DeepSpeed Flops Profiler output.

The DeepSpeed Flops Profiler can be used with the DeepSpeed runtime without any user code change or be used independently from DeepSpeed as a standalone package. When using DeepSpeed for model training, the profiler can be enabled in the DeepSpeed [configuration file](/docs/config-json/#flops-profiler). As a standalone package, the profiler API can be used in both training and inference code. The DeepSpeed profiler is still under active development and includes just initial features.  Stay connected for more exciting features to be added soon.

For models running on multi-GPU or multi-node, only the model parallelism (e.g. ```--model-parallel-size``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)) affects the number of flops and parameters profiled, i.e.,
`model_parallel_size * flops = total_flops` and `model_parallel_size * parameters = total_parameters`. The number of GPUs or nodes does not affect the output profile.

Similar to existing flops calculation tools or methods, the DeepSpeed Flops Profiler measures the flops of the forward pass of a module and the flops of the backward pass is estimated as `2` times of that of the forward pass.
Different from the PyTorch profiler which calculates the flops of PyTorch operators, the DeepSpeed Flops Profiler measures the flops within modules in a model and provides more insights to the users about the model execution.
The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that the DeepSpeed Flops Profiler not only supports flops computation directly at module level, but can also capture ```torch.nn.functional``` invoked in a module to estimate the flops.
Thus the DeepSpeed Flops Profiler allows for customized modules in the model, e.g., `ParallelTransformerLayerworks`, `ParallelSelfAttention`, `RowParallelLinear`, etc. in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). This is in contrast to ptflops which requires users to write customized flops calculation functions for each customized module.

## Multi-GPU, Multi-node, Data Parallelism, and Model Parallelism

The DeepSpeed Flops Profiler outputs the per GPU profile as well as the world size, data parallel size, and model parallel size.

For models running on multi-GPU or multi-node, only change of the model parallelism (e.g., `--model-parallel-size` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)) affects the number of flops and parameters profiled, i.e.,
`model_parallel_size * flops = total_flops` and `model_parallel_size * parameters = total_parameters`. The data parallel size or world size (related to the number of GPUs or nodes) does not affect the per GPU profile.

## Usage

The DeepSpeed Flops Profiler can be used with the DeepSpeed runtime or as a standalone package. When using DeepSpeed for model training, the profiler can be configured in the deepspeed [configuration file](/docs/config-json/#flops-profiler) without user code changes. To use the flops profiler outside the DeepSpeed runtime, install DeepSpeed and import the `flops_profiler` package to use the APIs directly. Examples of each usage are given below.

  - [Usage With the DeepSpeed Runtime](#usage-with-the-deepspeed-runtime)
    - [Example: Megatron-LM](#example-megatron-lm)
  - [Usage Outside the DeepSpeed Runtime](#usage-outside-the-deepspeed-runtime)
    - [In Model Inference](#in-model-inference)
      - [Example: AlexNet](#example-alexnet)
      - [Example: Bert](#example-bert)
    - [In Model Training Workflow](#in-model-training-workflow)
      - [Example Training Workflow](#example-training-workflow)

### Usage With the DeepSpeed Runtime

When using DeepSpeed for model training, the profiler can be configured in the deepspeed [configuration file](/docs/config-json/#flops-profiler). No explict API calls are needed to use the profiler. The profiler can be enabled by adding the following field to deepspeed's configuration json file. Refer to [flops profiler](/docs/config-json/#flops-profiler) for details.

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    }
}
```

#### Example: Megatron-LM

For information on running Megatron-LM with DeepSpeed, please refer to our tutorial [Megatron-LM](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/megatron/Megatron-LM).

An example output of 12-layer Megatron-LM model (`hidden_size = 8192, num_attention_heads = 32, batch_size = 1024, seq_length = 1024`) is shown below.

```shell
-------------------------- DeepSpeed Flops Profiler --------------------------
Profile Summary at step 10:
Notations:
data parallel size (dp_size), model paralel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

world size:                                                   1
data parallel size:                                           1
model paralel size:                                           1
batch size per GPU:                                           1024
params per gpu:                                               1.29 M
params of model = params per GPU * mp_size:                   1.29 M
fwd MACs per GPU:                                             41271.95 G
fwd flops per GPU:                                            82543.9 G
fwd flops of model = fwd flops per GPU * mp_size:             82543.9 G
fwd latency:                                                  1.89 s
bwd latency:                                                  5.38 s
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:          43.68 TFLOPS
bwd FLOPS per GPU = 2 * fwd flops per GPU / bwd latency:      30.7 TFLOPS
fwd+bwd FLOPS per GPU = 3 * fwd flops per GPU / (fwd+bwd latency):   34.07 TFLOPS
step latency:                                                 34.12 s
iter latency:                                                 41.39 s
samples/second:                                               24.74

------------------------------ Detailed Profile ------------------------------
Each module profile is listed after its name in the following order:
number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency, percentage of total latency, number of floating point operations per second (FLOPS, computed as 2 * MACs / latency).
Note:
1. A module can have torch.nn.functional (e.g. to compute logits) along with submodules, thus making the difference between the parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs(or latency) and the sum of its submodules'.
1. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
2. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

GPT2Model(
  1.29 M, 100.00% Params, 41271.95 GMACs, 100.00% MACs, 1.84 s, 100.00% latency, 44.78 TFLOPS,
  (language_model): TransformerLanguageModel(
    1.29 M, 100.00% Params, 39584.03 GMACs, 95.91% MACs, 1.83 s, 99.11% latency, 43.34 TFLOPS,
    (embedding): Embedding(
      2, 0.00% Params, 0 MACs, 0.00% MACs, 18.1 ms, 0.98% latency, 0.0 FLOPS,
      (word_embeddings): VocabParallelEmbedding(1, 0.00% Params, 0 MACs, 0.00% MACs, 164.75 us, 0.01% latency, 0.0 FLOPS, )
      (position_embeddings): Embedding(1, 0.00% Params, 0 MACs, 0.00% MACs, 489.23 us, 0.03% latency, 0.0 FLOPS, 1024, 8192)
      (embedding_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 93.94 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
    )
    (transformer): ParallelTransformer(
      1.29 M, 100.00% Params, 39584.03 GMACs, 95.91% MACs, 1.81 s, 98.11% latency, 43.78 TFLOPS,
      (layers): ModuleList(
        1.28 M, 98.73% Params, 39584.03 GMACs, 95.91% MACs, 1.3 s, 70.66% latency, 60.79 TFLOPS,
        (0): ParallelTransformerLayerPart1(
          49.15 k, 3.80% Params, 1099.65 GMACs, 2.66% MACs, 23.5 ms, 1.27% latency, 93.6 TFLOPS,
          (input_layernorm): FusedLayerNorm(16.38 k, 1.27% Params, 0 MACs, 0.00% MACs, 128.75 us, 0.01% latency, 0.0 FLOPS, torch.Size([8192]), eps=1e-05, elementwise_affine=True)
          (attention): ParallelSelfAttention(
            32.77 k, 2.53% Params, 1099.65 GMACs, 2.66% MACs, 22.8 ms, 1.24% latency, 96.46 TFLOPS,
            (query_key_value): ColumnParallelLinear(24.58 k, 1.90% Params, 824.63 GMACs, 2.00% MACs, 8.93 ms, 0.48% latency, 184.7 TFLOPS, )
            (scale_mask_softmax): FusedScaleMaskSoftmax(0, 0.00% Params, 134.22 MMACs, 0.00% MACs, 151.16 us, 0.01% latency, 1.78 TFLOPS, )
            (attention_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 79.63 us, 0.00% latency, 0.0 FLOPS, p=0.1, inplace=False)
            (dense): RowParallelLinear(8.19 k, 0.63% Params, 274.88 GMACs, 0.67% MACs, 2.67 ms, 0.14% latency, 205.81 TFLOPS, )
          )
        )
        (transformer): ParallelTransformer(
          12.61 M, 32.43% Params, 103.62 GMACs, 32.94% MACs, 5.0 ms, 14.78% latency, 41.49 TFLOPS,
          (layers): ModuleList(
            12.61 M, 32.42% Params, 103.62 GMACs, 32.94% MACs, 4.4 ms, 13.01% latency, 47.13 TFLOPS,
            (0): ParallelTransformerLayer(
              3.15 M, 8.11% Params, 25.9 GMACs, 8.23% MACs, 1.36 ms, 4.02% latency, 38.09 TFLOPS,
              (input_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 92.51 us, 0.27% latency, 0.0 FLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
              (attention): ParallelSelfAttention(
                1.05 M, 2.70% Params, 8.72 GMACs, 2.77% MACs, 754.59 us, 2.23% latency, 23.12 TFLOPS,
                (query_key_value): ColumnParallelLinear(787.97 k, 2.03% Params, 6.44 GMACs, 2.05% MACs, 182.87 us, 0.54% latency, 70.46 TFLOPS, )
                (scale_mask_softmax): FusedScaleMaskSoftmax(0, 0.00% Params, 134.22 MMACs, 0.04% MACs, 120.4 us, 0.36% latency, 2.23 TFLOPS, )
                (attention_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 47.45 us, 0.14% latency, 0.0 FLOPS, p=0.1, inplace=False)
                (dense): RowParallelLinear(262.66 k, 0.68% Params, 2.15 GMACs, 0.68% MACs, 81.78 us, 0.24% latency, 52.52 TFLOPS, )
              )
              (post_attention_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 57.22 us, 0.17% latency, 0.0 FLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
              (mlp): ParallelMLP(
                2.1 M, 5.40% Params, 17.18 GMACs, 5.46% MACs, 224.83 us, 0.67% latency, 152.83 TFLOPS,
                (dense_h_to_4h): ColumnParallelLinear(1.05 M, 2.70% Params, 8.59 GMACs, 2.73% MACs, 64.13 us, 0.19% latency, 267.87 TFLOPS, )
                (dense_4h_to_h): RowParallelLinear(1.05 M, 2.70% Params, 8.59 GMACs, 2.73% MACs, 90.36 us, 0.27% latency, 190.13 TFLOPS, )
              )
            )
            ...
            (3): ParallelTransformerLayer(...)
          (final_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 52.69 us, 0.16% latency, 0.0 TFLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
)
```

###  Usage Outside the DeepSpeed Runtime

The flops profiler can be used as a standalone package outside of the DeepSpeed runtime.
One can simply install DeepSpeed and import the `flops_profiler` package to use the APIs directly.
Refer to [installation of DeepSpeed](https://www.deepspeed.ai/getting-started/#installation) for installing DeepSpeed.

#### In Model Inference

To profile a trained model in inference, use the `get_model_profile` function.
Examples are given below.

##### Example: AlexNet

The following example shows how to profile AlexNet using the DeepSpeed flops profiler.

```python
import torchvision.models as models
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

with get_accelerator().device(0):
    model = models.alexnet()
    batch_size = 256
    flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling
```

An example output:

```shell
-------------------------- DeepSpeed Flops Profiler --------------------------
Summary of forward pass:
Profile step:                   10
Number of parameters:           61.1 M
Number of multiply-accumulate operations (MACs):   183.18 G
Number of floating point operations ( = 2 * MACs):   366.36 G
Latency:                        22.13 ms
Floating point operations per second(FLOPS):   16.56 TFLOPS

----------------------------- Aggregated Profile -----------------------------
Top 3 modules in MACs at depth 2 are {'Conv2d': '167.95 GMACs', 'Linear': '15.01 GMACs', 'ReLU': '126.26 MMACs'}
Top 3 modules in params at depth 2 are {'Linear': '58.63 M', 'Conv2d': '2.47 M', 'ReLU': '0'}
Top 3 modules in latency at depth 2 are {'Conv2d': '13.96 ms', 'Linear': '6.23 ms', 'ReLU': '730.75 us'}

------------------------------ Detailed Profile ------------------------------
Each module profile is listed after its name in the following order:
number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency, percentage of total latency, number of floating point operations per second (FLOPS, computed as 2 * MACs / latency).
Note:
1. A module can have torch.nn.functional (e.g. to compute logits) along with submodules, thus making the difference between the parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.

AlexNet(
  61.1 M, 100.00% Params, 183.18 GMACs, 100.00% MACs, 22.13 ms, 100.00% latency, 16.56 TFLOPS,
  (features): Sequential(
    2.47 M, 4.04% Params, 168.17 GMACs, 91.81% MACs, 15.17 ms, 68.57% latency, 22.17 TFLOPS,
    (0): Conv2d(23.3 k, 0.04% Params, 18.04 GMACs, 9.85% MACs, 633.0 us, 2.86% latency, 57.0 TFLOPS, 3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(0, 0.00% Params, 49.56 MMACs, 0.03% MACs, 163.79 us, 0.74% latency, 605.17 GFLOPS, inplace=True)
    (2): MaxPool2d(0, 0.00% Params, 49.56 MMACs, 0.03% MACs, 159.26 us, 0.72% latency, 622.38 GFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(307.39 k, 0.50% Params, 57.37 GMACs, 31.32% MACs, 6.15 ms, 27.81% latency, 18.64 TFLOPS, 64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(0, 0.00% Params, 35.83 MMACs, 0.02% MACs, 185.01 us, 0.84% latency, 387.34 GFLOPS, inplace=True)
    (5): MaxPool2d(0, 0.00% Params, 35.83 MMACs, 0.02% MACs, 134.23 us, 0.61% latency, 533.89 GFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(663.94 k, 1.09% Params, 28.72 GMACs, 15.68% MACs, 389.58 us, 1.76% latency, 147.47 TFLOPS, 192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(0, 0.00% Params, 16.61 MMACs, 0.01% MACs, 76.53 us, 0.35% latency, 434.15 GFLOPS, inplace=True)
    (8): Conv2d(884.99 k, 1.45% Params, 38.29 GMACs, 20.90% MACs, 6.38 ms, 28.82% latency, 12.01 TFLOPS, 384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 104.43 us, 0.47% latency, 212.12 GFLOPS, inplace=True)
    (10): Conv2d(590.08 k, 0.97% Params, 25.53 GMACs, 13.94% MACs, 405.79 us, 1.83% latency, 125.83 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 65.57 us, 0.30% latency, 337.85 GFLOPS, inplace=True)
    (12): MaxPool2d(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 122.07 us, 0.55% latency, 181.46 GFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(0, 0.00% Params, 2.36 MMACs, 0.00% MACs, 259.4 us, 1.17% latency, 18.19 GFLOPS, output_size=(6, 6))
  (classifier): Sequential(
    58.63 M, 95.96% Params, 15.01 GMACs, 8.19% MACs, 6.54 ms, 29.54% latency, 4.59 TFLOPS,
    (0): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 42.68 us, 0.19% latency, 0.0 FLOPS, p=0.5, inplace=False)
    (1): Linear(37.75 M, 61.79% Params, 9.66 GMACs, 5.28% MACs, 301.36 us, 1.36% latency, 64.13 TFLOPS, in_features=9216, out_features=4096, bias=True)
    (2): ReLU(0, 0.00% Params, 1.05 MMACs, 0.00% MACs, 79.39 us, 0.36% latency, 26.41 GFLOPS, inplace=True)
    (3): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.58 us, 0.18% latency, 0.0 FLOPS, p=0.5, inplace=False)
    (4): Linear(16.78 M, 27.46% Params, 4.29 GMACs, 2.34% MACs, 234.37 us, 1.06% latency, 36.65 TFLOPS, in_features=4096, out_features=4096, bias=True)
    (5): ReLU(0, 0.00% Params, 1.05 MMACs, 0.00% MACs, 56.03 us, 0.25% latency, 37.43 GFLOPS, inplace=True)
    (6): Linear(4.1 M, 6.71% Params, 1.05 GMACs, 0.57% MACs, 5.69 ms, 25.72% latency, 368.42 GFLOPS, in_features=4096, out_features=1000, bias=True)
  )
)
------------------------------------------------------------------------------
```

##### Example: Bert

```python
from functools import partial
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


with get_accelerator().device(0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 4
    seq_len = 128
    enable_profile = True
    if enable_profile:
      flops, macs, params = get_model_profile(
          model,
          kwargs=bert_input_constructor(batch_size, seq_len, tokenizer),
          print_profile=True,
          detailed=True,
      )
    else:
      inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
      outputs = model(inputs)
```

An example output:

```
-------------------------- DeepSpeed Flops Profiler --------------------------
Summary of forward pass:
Profile step:                   1
Number of parameters:           109.48 M
Number of multiply-accumulate operations (MACs):   43.5 G
Number of floating point operations ( = 2 * MACs):   87.0 G
Latency:                        393.7 ms
Floating point operations per second(FLOPS):   220.97 GFLOPS

----------------------------- Aggregated Profile -----------------------------
Top 3 modules in MACs at depth 7 are {'Linear': '14.5 GMACs', 'Dropout': '0 MACs', 'LayerNorm': '0 MACs'}
Top 3 modules in params at depth 7 are {'Linear': '28.35 M', 'LayerNorm': '18.43 k', 'Dropout': '0'}
Top 3 modules in latency at depth 7 are {'Linear': '153.7 ms', 'LayerNorm': '4.74 ms', 'Dropout': '597.95 us'}

------------------------------ Detailed Profile ------------------------------
Each module profile is listed after its name in the following order:
number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency, percentage of total latency, number of floating point operations per second (FLOPS, computed as 2 * MACs / latency).
Note:
1. A module can have torch.nn.functional (e.g. to compute logits) along with submodules, thus making the difference between the parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.

BertForSequenceClassification(
  109.48 M, 100.00% Params, 43.5 GMACs, 100.00% MACs, 393.7 ms, 100.00% latency, 220.97 GFLOPS,
  (bert): BertModel(
    109.48 M, 100.00% Params, 43.5 GMACs, 100.00% MACs, 393.38 ms, 99.92% latency, 221.15 GFLOPS,
    (embeddings): BertEmbeddings(
      23.84 M, 21.77% Params, 0 MACs, 0.00% MACs, 1.79 ms, 0.45% latency, 0.0 FLOPS,
      (word_embeddings): Embedding(23.44 M, 21.41% Params, 0 MACs, 0.00% MACs, 485.18 us, 0.12% latency, 0.0 FLOPS, 30522, 768, padding_idx=0)
      (position_embeddings): Embedding(393.22 k, 0.36% Params, 0 MACs, 0.00% MACs, 111.1 us, 0.03% latency, 0.0 FLOPS, 512, 768)
      (token_type_embeddings): Embedding(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 215.53 us, 0.05% latency, 0.0 FLOPS, 2, 768)
      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 386.95 us, 0.10% latency, 0.0 FLOPS, (768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 20.27 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      85.05 M, 77.69% Params, 43.5 GMACs, 99.99% MACs, 391.03 ms, 99.32% latency, 222.47 GFLOPS,
      (layer): ModuleList(
        85.05 M, 77.69% Params, 43.5 GMACs, 99.99% MACs, 390.82 ms, 99.27% latency, 222.59 GFLOPS,
        (0): BertLayer(
          7.09 M, 6.47% Params, 3.62 GMACs, 8.33% MACs, 31.91 ms, 8.10% latency, 227.21 GFLOPS,
          (attention): BertAttention(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 16.39 ms, 4.16% latency, 147.47 GFLOPS,
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 906.76 MMACs, 2.08% MACs, 15.07 ms, 3.83% latency, 120.36 GFLOPS,
              (query): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 3.66 ms, 0.93% latency, 164.91 GFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 3.72 ms, 0.94% latency, 162.36 GFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 4.52 ms, 1.15% latency, 133.65 GFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 24.08 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 1.29 ms, 0.33% latency, 469.21 GFLOPS,
              (dense): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 504.26 us, 0.13% latency, 1.2 TFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 437.97 us, 0.11% latency, 0.0 FLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 21.93 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 9.57 ms, 2.43% latency, 252.35 GFLOPS,
            (dense): Linear(2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 8.75 ms, 2.22% latency, 276.11 GFLOPS, in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 5.77 ms, 1.47% latency, 418.39 GFLOPS,
            (dense): Linear(2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 5.13 ms, 1.30% latency, 471.15 GFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 310.9 us, 0.08% latency, 0.0 FLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 29.8 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        ...
        (11): BertLayer(...)
      )
    )
    (pooler): BertPooler(
      590.59 k, 0.54% Params, 2.36 MMACs, 0.01% MACs, 337.12 us, 0.09% latency, 14.0 GFLOPS,
      (dense): Linear(590.59 k, 0.54% Params, 2.36 MMACs, 0.01% MACs, 173.57 us, 0.04% latency, 27.19 GFLOPS, in_features=768, out_features=768, bias=True)
      (activation): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 46.01 us, 0.01% latency, 0.0 FLOPS, )
    )
  )
  (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 19.55 us, 0.00% latency, 0.0 FLOPS, p=0.1, inplace=False)
  (classifier): Linear(1.54 k, 0.00% Params, 6.14 KMACs, 0.00% MACs, 56.51 us, 0.01% latency, 217.47 MFLOPS, in_features=768, out_features=2, bias=True)
)
------------------------------------------------------------------------------
```

#### In Model Training Workflow

To profile model forward in a training workflow, use the `FlopsProfiler`class.
The `FlopsProfiler`class provides the follwing methods:
  * `start_profile()` - starts profiling
  * `get_total_flops(as_string=False)` - returns the total number of floating-point operations in the model
  * `get_total_macs(as_string=False)` - returns the total number of MACs in the model
  * `get_total_params(as_string=False)` - returns the total number of parameters in the model
  * `print_model_profile(profile_step=1, module_depth=-1, top_modules=3, detailed=True)` - prints the model profile
  * `end_profile()` - ends profiling and cleans up. This should be invoked at the end of the profiling and AFTER `get_total_flops`, `get_total_params` or `print_model_profile`.

##### Example Training Workflow

Below is an example of this usage in a typical training workflow. Note that the flops profiler only captures the forward pass in a training step. The flops of a backward pass can be roughly estimated from that of the forward pass (~2x).

```python
from deepspeed.profiling.flops_profiler import FlopsProfiler

model = Model()
prof = FlopsProfiler(model)

profile_step = 5
print_profile= True

for step, batch in enumerate(data_loader):
  # start profiling at training step "profile_step"
  if step == profile_step:
    prof.start_profile()

  # forward() method
  loss = model(batch)

  # end profiling and print output
  if step == profile_step: # if using multi nodes, check global_rank == 0 as well
    prof.stop_profile()
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=profile_step)
    prof.end_profile()

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()
```
