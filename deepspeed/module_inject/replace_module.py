# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import tqdm
import deepspeed
import deepspeed.ops.transformer as transformer_inference
from .replace_policy import HFBertLayerPolicy, HFGPT2LayerPolicy, HFGPTJLayerPolicy
from .replace_policy import replace_policies
from ..constants import INFERENCE_GENERIC_MODE, INFERENCE_SPECIALIZED_MODE
from ..runtime.weight_quantizer import WeightQuantization
from torch import nn
from deepspeed import comm as dist

from .load_checkpoint import load_model_with_checkpoint
import time

class LinearAllreduce(nn.Module):
    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.mp_group is not None:
            torch.distributed.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class LinearLayer(nn.Module):
    def __init__(self, weight, bias=None):
        super(LinearLayer, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            output += self.bias
        return output


class ReplaceWithTensorSlicing:
    def __init__(self, mp_group=None):
        if mp_group is not None:
            self.gpu_index = torch.distributed.get_rank(group=mp_group)
        else:
            self.gpu_index = 0

    def merge_assert(self, dim1, dim2):
        assert dim1 > dim2, \
            'Merging tensors is not allowed here! Please use deepspeed load_checkpoint\
            for merging your checkpoints before replacing the transformer layer with\
            inference-kernels'

    def qkv_copy(self, dst, src):
        if src is None:
            return src
        src_shape = src.shape
        dst_shape = dst.shape

        src_split = torch.split(src.data, src.shape[-1] // 3, dim=-1)

        if (len(src_shape) == 2 and len(dst_shape) == 2):
            if src_shape[1] == dst_shape[1]:
                return src

            self.merge_assert(src_shape[1], dst_shape[1])
            qkv_size = dst_shape[1] // 3
            qkv_split = [torch.split(src_s, qkv_size, dim=1) for src_s in src_split]

            weight_split = [
                torch.cat([qkv_s[i] for qkv_s in qkv_split],
                          axis=1) for i in range(len(qkv_split[0]))
            ]
            dst.data.copy_(weight_split[self.gpu_index].to(
                torch.cuda.current_device()).contiguous())
        else:
            if src_shape[0] == dst_shape[0]:
                return src

            qkv_size = dst_shape[0] // 3
            qkv_split = [torch.split(src_s, qkv_size, dim=0) for src_s in src_split]
            bias_split = [
                torch.cat([qkv_s[i] for qkv_s in qkv_split],
                          axis=0) for i in range(len(qkv_split[0]))
            ]
            dst.data.copy_(bias_split[self.gpu_index].to(
                torch.cuda.current_device()).contiguous())

        return dst

    def copy(self, dst, src):
        if src is None:
            return src

        src_shape = src.shape
        dst_shape = dst.shape

        if (len(src_shape) == 2 and len(dst_shape) == 2):

            if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
                return src

            if src_shape[0] != dst_shape[0]:
                self.merge_assert(src_shape[0], dst_shape[0])
                weight_split = torch.split(src, dst_shape[0])
            else:
                self.merge_assert(src_shape[1], dst_shape[1])
                weight_split = torch.split(src.data, dst_shape[1], dim=1)

            dst.data.copy_(weight_split[self.gpu_index].to(
                torch.cuda.current_device()).contiguous())
        else:
            if src_shape[0] == dst_shape[0]:
                return src

            bias_split = torch.split(src.data, dst_shape[-1])
            dst.data.copy_(bias_split[self.gpu_index].to(
                torch.cuda.current_device()).contiguous())

        return dst


def replace_transformer_layer(orig_layer_impl,
                              model,
                              micro_batch_size,
                              bert_config,
                              seed=-1,
                              preln=True,
                              fp16=True,
                              training=True,
                              huggingface=False,
                              local_rank=-1):
    """ Replace bert-style transformer layers with DeepSpeed's transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation to look for,
            e.g., transformers.models.bert.modeling_bert.BertLayer or transformers.BertLayer
        model (torch.nn.Module): user's nn.module representing their model
        micro_batch_size (int): micro batch size per gpu used during training/eval
        bert_config (dict): model config containing hidden size, attention heads, etc.
        seed (int): random seed value
        preln (bool): does the original layer implementation do pre or post layer norm?
        fp16 (bool): fp16 or fp32
        Training (bool): select between training (True) or inference (False) mode
        huggingface (bool): huggingface implementation is unique (supports both encoder/decoder modes)

    Returns:
        Updated nn.module with replaced transformer layers
    """
    def replace_fn(child):
        transformer_config = deepspeed.DeepSpeedTransformerConfig(
            batch_size=micro_batch_size,
            hidden_size=bert_config.hidden_size,
            heads=bert_config.num_attention_heads,
            attn_dropout_ratio=bert_config.attention_probs_dropout_prob,
            hidden_dropout_ratio=bert_config.hidden_dropout_prob,
            num_hidden_layers=bert_config.num_hidden_layers,
            initializer_range=bert_config.initializer_range,
            layer_norm_eps=bert_config.layer_norm_eps,
            seed=seed,
            fp16=fp16,
            pre_layer_norm=preln,
            huggingface=huggingface,
            local_rank=local_rank,
            training=training)
        new_module = deepspeed.DeepSpeedTransformerLayer(transformer_config)

        if inference:
            hidden_size, num_attention_heads = policy.get_hidden_heads()
            assert num_attention_heads % mp_size == 0,\
                "To run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!" +\
                "This is because the attention computation is partitioned evenly among the parallel GPUs."

    mp_replace = ReplaceWithTensorSlicing(mp_group=config.tensor_parallel.tp_group,
                                          mp_size=config.tensor_parallel.tp_size)  #, out_dim=0, in_dim=1)

    def replace_with_policy(child, policy_cls, triangular_masking, inference=False, layer_id=0):
        policy = policy_cls(child, inference=inference)
        if not policy.cuda_graph_supported:
            # policy says cuda graph is not supported raise an error if set
            assert not config.enable_cuda_graph, "cuda graph is not supported with this model, please disable"

        from deepspeed.moe.layer import MoE
        moe = False
        if hasattr(child, 'mlp') and isinstance(child.mlp, MoE):
            num_experts = child.mlp.num_experts
            moe = True

        # 1. Create a model-specific container object using the policy object.
        _container = policy_to_ds_container(policy=policy,
                                            config=config,
                                            model_config=model_config,
                                            layer_id=layer_id,
                                            child=child)
        _container.set_moe(moe)

        # 2. Set the tensor parallelism config
        _container.set_tensor_parallel_config(config.tensor_parallel.tp_size, config.tensor_parallel.tp_group)

        # 3. Initialize tensors
        _container.initialize_tensors()

        # 4. deal with data types -- needs refactor to use dtype instead of fp16
        if config.dtype in [torch.float16, torch.bfloat16, torch.int8]:
            _container.convert_to_required_dtype()

        # 5. Set the quantization config
        quantizer = GroupQuantizer(q_int8=quantize)
        _container.set_quantization_config(quantizer)

        # 6. create a DS Inference config object
        _container.create_ds_model_config()

        # 7. use the config and create the module
        _container.create_module()

        # 8. transpose the weights and bias if needed
        _container.transpose()

        # 9. deal with tensor parallelism.
        _container.apply_tensor_parallelism(mp_replace)

        # 10. copy the tensors from the model-specific container to the new module
        _container.copy_data_to_new_module()

        # 11. set global for generic checkpoint loading
        global container_g

        if container_g is None:
            container_g = _container

        return _container.module

    def replace_wo_policy(module, all_reduce_linears, prefix="", state_dict=None):
        #mp_replace = ReplaceWithTensorSlicing(mp_group=config.tensor_parallel.tp_group)

        # 1. Create AutoTP object
        _autotp = AutoTP(module, all_reduce_linears, prefix, state_dict, linear_layer_setting, orig_layer_impl,
                         config.keep_module_on_host)

        # 2. Set the tensor parallelism config
        _autotp.set_tensor_parallel_config(config.tensor_parallel.tp_size, config.tensor_parallel.tp_group)

        # 3. Try to get num_key_heads from model_config.num_key_value_heads
        if hasattr(model_config, "vision_config"):
            if "MllamaVisionEncoderLayer" in str(module):
                num_kv_heads = _autotp.get_model_num_kv_heads(model_config.vision_config)
            elif hasattr(model_config, "text_config"):
                num_kv_heads = _autotp.get_model_num_kv_heads(model_config.text_config)
            else:
                num_kv_heads = _autotp.get_model_num_kv_heads(model_config)
        else:
            num_kv_heads = _autotp.get_model_num_kv_heads(model_config)

        # 4. When we have num_kv_heads defined, uneven division is possible, otherwise enforce even division
        set_num_kv_heads(num_kv_heads)

        # 4.1 Get n_embd
        n_embd = None
        multi_query_n_embd_names = ['n_embd', 'hidden_size']
        for name in multi_query_n_embd_names:
            if hasattr(model_config, name):
                n_embd = getattr(model_config, name)
            if n_embd != None:
                break

        # 4.2 set n_embd
        set_n_embd(n_embd)

        # 4.3 set attention_heads
        if hasattr(model_config, 'num_attention_heads'):
            set_num_attention_heads(getattr(model_config, 'num_attention_heads'))

        # 4.4 set tp_grain_size
        set_tp_grain_size(config.tensor_parallel.tp_grain_size)

            if quantize and quantize_settings is not None:
                (quantization_scales,
                 merge_count,
                 mlp_extra_grouping,
                 quantize_groups) = quantize_settings
                if moe:
                    new_module = transformer_inference.DeepSpeedMoEInference(
                        transformer_config,
                        mp_group=mp_group,
                        ep_group=None if ep_group is None else ep_group[num_experts],
                        expert_mp_group=None
                        if expert_mp_group is None else expert_mp_group[num_experts],
                        quantize_scales=quantization_scales[layer_id],
                        quantize_groups=quantize_groups,
                        merge_count=merge_count,
                        mlp_extra_grouping=mlp_extra_grouping,
                        qkv_merging=(policy_cls is HFBertLayerPolicy))

        # 6. Replace modules
        if "lm_head" in all_reduce_linears or "embed_out" in all_reduce_linears:
            return _autotp._replace_last_linear_module(module)
        return _autotp._replace_module(module)

                if quantize and qkvw.dtype != torch.int8:
                    quantize_bits = 8
                    quantizer = WeightQuantization()
                    if policy_cls is HFBertLayerPolicy:
                        data_quantized, _ = quantizer.quantize_data(qkvw.data, quantize_bits, quantize_groups * 3)
                    else:
                        data_quantized, _ = quantizer.quantize_data(qkvw.data, quantize_bits, quantize_groups)
                    qkvw.data.copy_(data_quantized)
                    qkvw.data = qkvw.data.to(torch.int8)
            else:

                if moe:
                    new_module = transformer_inference.DeepSpeedMoEInference(
                        transformer_config,
                        mp_group=mp_group,
                        ep_group=None if ep_group is None else ep_group[num_experts],
                        expert_mp_group=None
                        if expert_mp_group is None else expert_mp_group[num_experts],
                    )

                else:
                    new_module = transformer_inference.DeepSpeedTransformerInference(
                        transformer_config,
                        mp_group=mp_group,
                    )
            new_module.config.scale_attention = scale_attention

            # we want the weights in [input, output] shape
            # linear layer is created with [input, output] shape
            # transpose it here to reduce inference cost!
            def transpose(data):
                data.view(-1).copy_(data.transpose(-1, -2).contiguous().view(-1))
                data = data.reshape(data.shape[-1], data.shape[-2])
                return data

            if attn_linear_layer:
                qkvw.data = transpose(qkvw.data)
                dense_w.data = transpose(dense_w.data)

            if mlp_linear_layer:
                _h4h_w = [transpose(moe_w1.data)
                          for moe_w1 in _h4h_w] if moe else transpose(_h4h_w.data)
                _4hh_w = [transpose(moe_w1.data)
                          for moe_w1 in _4hh_w] if moe else transpose(_4hh_w.data)

            if moe and moe_type == 'residual':
                _res_h4h_w.data = transpose(_res_h4h_w.data)
                _res_4hh_w.data = transpose(_res_4hh_w.data)
                _res_coef.data = transpose(_res_coef.data)

            attn_block = new_module.attention
            attn_block.attn_qkvw = mp_replace.qkv_copy(attn_block.attn_qkvw, qkvw)
            attn_block.attn_qkvb = mp_replace.qkv_copy(attn_block.attn_qkvb, qkvb)

                        attn_block.attn_ow = mp_replace.copy(attn_block.attn_ow, dense_w)
                        attn_block.attn_ob = mp_replace.copy(attn_block.attn_ob, dense_b)
            else:
                if bigscience_bloom:
                    attn_block.attn_qkvw = mp_replace.copy(attn_block.attn_qkvw, qkvw)
                    attn_block.attn_qkvb = mp_replace.copy(attn_block.attn_qkvb, qkvb)
                else:
                    attn_block.attn_qkvw = mp_replace.qkv_copy(
                        attn_block.attn_qkvw,
                        qkvw)
                    attn_block.attn_qkvb = mp_replace.qkv_copy(
                        attn_block.attn_qkvb,
                        qkvb)

                attn_block.attn_ow = mp_replace.copy(attn_block.attn_ow, dense_w)
                attn_block.attn_ob = mp_replace.copy(attn_block.attn_ob, dense_b)

            mpl_block = new_module.mlp
            if moe:
                gpu_index = torch.distributed.get_rank()
                gpu_index = 0
                for ep_index in range(local_ep_size):
                    mpl_block[ep_index].inter_w.data = _h4h_w[
                        gpu_index * local_ep_size + ep_index].to(
                            torch.cuda.current_device())
                    mpl_block[ep_index].inter_b.data = _h4h_b[
                        gpu_index * local_ep_size + ep_index].to(
                            torch.cuda.current_device())
                    mpl_block[ep_index].output_w.data = _4hh_w[
                        gpu_index * local_ep_size + ep_index].to(
                            torch.cuda.current_device())
                    mpl_block[ep_index].output_b.data = _4hh_b[
                        gpu_index * local_ep_size + ep_index].to(
                            torch.cuda.current_device())
                new_module.attn_nw.data = attn_nw.to(torch.cuda.current_device())
                new_module.attn_nb.data = attn_nb.to(torch.cuda.current_device())
                if moe_type == 'residual':
                    new_module.res_mlp.inter_w.data = _res_h4h_w.to(
                        torch.cuda.current_device())
                    new_module.res_mlp.inter_b.data = _res_h4h_b.to(
                        torch.cuda.current_device())
                    new_module.res_mlp.output_w.data = _res_4hh_w.to(
                        torch.cuda.current_device())
                    new_module.res_mlp.output_b.data = _res_4hh_b.to(
                        torch.cuda.current_device())
                    new_module.res_coef.data = _res_coef.to(torch.cuda.current_device())
            else:
                mpl_block.inter_w.data = mp_replace.copy(mpl_block.inter_w, _h4h_w)
                mpl_block.inter_b.data = mp_replace.copy(mpl_block.inter_b, _h4h_b)
                mpl_block.output_w.data = mp_replace.copy(mpl_block.output_w, _4hh_w)
                mpl_block.output_b.data = mp_replace.copy(mpl_block.output_b, _4hh_b)
                if attn_nw is None:
                    new_module.mlp.attn_nw = attn_nw
                else:
                    new_module.mlp.attn_nw.data = attn_nw.to(torch.cuda.current_device())
                if attn_nb is None:
                    new_module.mlp.attn_nb = attn_nb
                else:
                    new_module.mlp.attn_nb.data = attn_nb.to(torch.cuda.current_device())
            new_module.norm_w.data = input_nw.to(torch.cuda.current_device())
            new_module.norm_b.data = input_nb.to(torch.cuda.current_device())
        else:
            transformer_config = deepspeed.DeepSpeedTransformerConfig(
                batch_size=micro_batch_size,
                hidden_size=config.hidden_size,
                heads=config.num_attention_heads,
                attn_dropout_ratio=config.attention_probs_dropout_prob,
                hidden_dropout_ratio=config.hidden_dropout_prob,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                layer_norm_eps=config.layer_norm_eps if hasattr(
                    config,
                    'layer_norm_eps') else 1e-12,
                seed=seed,
                fp16=fp16,
                pre_layer_norm=(False if policy_cls is HFBertLayerPolicy else preln),
                return_tuple=return_tuple,
                local_rank=local_rank,
                stochastic_mode=stochastic_mode,
                normalize_invertible=True,
                training=training)
            new_module = deepspeed.DeepSpeedTransformerLayer(transformer_config)
            new_module.attn_qkvw.data = qkvw
            new_module.attn_qkvb.data = qkvb
            new_module.attn_ow.data = dense_w
            new_module.attn_ob.data = dense_b

            new_module.attn_nw.data = attn_nw
            new_module.attn_nb.data = attn_nb
            new_module.norm_w.data = input_nw
            new_module.norm_b.data = input_nb

            new_module.inter_w.data = _h4h_w
            new_module.inter_b.data = _h4h_b
            new_module.output_w.data = _4hh_w
            new_module.output_b.data = _4hh_b
        return new_module

    def replace_wo_policy(module, all_reduce_linears):
        def _replace(child, name, conv_linear_layer):
            mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)
            if name in all_reduce_linears:
                new_weight = torch.empty((
                    weight_shape[1] if conv_linear_layer else weight_shape[0],
                    (weight_shape[0] if conv_linear_layer else weight_shape[1]) //
                    mp_size,
                ),
                                         device=child.weight.device,
                                         dtype=child.weight.dtype)
                if z_inference:
                    with deepspeed.zero.GatheredParameters(child.weight,
                                                           modifier_rank=0):
                        data = child.weight.data.to(new_weight.device)
                        if conv_linear_layer:
                            data = data.transpose(-1, -2).contiguous()
                        data = mp_replace.copy(new_weight, data)
                    child.weight.ds_tensor = torch.empty(1)
                else:
                    if conv_linear_layer:
                        child.weight.data = child.weight.data.transpose(-1,
                                                                        -2).contiguous()
                    data = mp_replace.copy(new_weight, child.weight.data)
                new_bias = torch.empty((weight_shape[0]),
                                       device=child.weight.device,
                                       dtype=child.weight.dtype)
                if z_inference:
                    with deepspeed.zero.GatheredParameters(child.bias, modifier_rank=0):
                        new_bias.data.copy_(child.bias.data)
                elif child.bias is not None:
                    new_bias.data.copy_(child.bias.data)
                return LinearAllreduce(data, child.bias if child.bias is None else \
                            child.bias.to(torch.cuda.current_device()), mp_group)
            else:
                new_weight = torch.empty(
                    (child.weight.shape[0] //
                     mp_size if conv_linear_layer else child.weight.shape[1],
                     child.weight.shape[1]
                     if conv_linear_layer else child.weight.shape[0] // mp_size),
                    device=child.weight.device,
                    dtype=torch.half if fp16 else torch.float)
                if not conv_linear_layer:
                    child.weight.data.view(-1).copy_(
                        child.weight.data.transpose(-1,
                                                    -2).contiguous().view(-1))
                    child.weight.data = child.weight.data.reshape(
                        child.weight.data.shape[-1],
                        child.weight.data.shape[-2])
                data = mp_replace.copy(new_weight, child.weight.data)
                new_bias = torch.empty((child.weight.shape[1] // mp_size),
                                       device=child.weight.device,
                                       dtype=child.weight.dtype)
                if z_inference:
                    with deepspeed.zero.GatheredParameters(child.bias, modifier_rank=0):
                        bias_data = None if child.bias is None else mp_replace.copy(
                            new_bias,
                            child.bias.data).to(torch.cuda.current_device())
                else:
                    bias_data = None if child.bias is None else mp_replace.copy(
                        new_bias,
                        child.bias.data).to(torch.cuda.current_device())
                return LinearLayer(weight=data.to(torch.cuda.current_device()),
                                   bias=bias_data)

        def _slice_embedding(child, name, conv_linear_layer):
            mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)
            new_weight = torch.empty((child.weight.shape[0],
                                      child.weight.shape[1] // mp_size),
                                     device=child.weight.device,
                                     dtype=child.weight.dtype)
            data = mp_replace.copy(new_weight,
                                   child.weight.ds_tensor.data if hasattr(child.weight, 'ds_tensor') else \
                                   child.weight.data)
            new_embedding = nn.Embedding(child.weight.shape[0],
                                         child.weight.shape[1] // mp_size)
            new_embedding.weight.data.copy_(data)
            return new_embedding

        def update_mp_params(child):
            if hasattr(child, 'n_heads'):
                child.n_heads = child.n_heads // mp_size
            if hasattr(child, 'inner_dim'):
                child.inner_dim = child.inner_dim // mp_size
            if hasattr(child, 'num_heads'):
                child.num_heads = child.num_heads // mp_size
            if hasattr(child, 'num_attention_heads'):
                child.num_attention_heads = child.num_attention_heads // mp_size
            if hasattr(child, 'all_head_size'):
                child.all_head_size = child.all_head_size // mp_size
            if hasattr(child, 'embed_dim'):
                child.embed_dim = child.embed_dim // mp_size

        conv_linear_layer = False
        if linear_layer_setting is not None:
            linear_policies = {linear_layer_setting[0]: _replace}
            if len(linear_layer_setting) == 2:
                linear_policies.update({linear_layer_setting[1]: _slice_embedding})
        else:
            if orig_layer_impl is HFGPT2LayerPolicy._orig_layer_class:
                try:
                    import transformers
                    conv_linear_layer = True
                    linear_policies = {transformers.model_utils.Conv1D: _replace}
                except ImportError:
                    linear_policies = {nn.Linear: _replace}
            else:
                linear_policies = {nn.Linear: _replace, nn.Embedding: _slice_embedding}

        def _replace_module(r_module, prev_name=''):
            for name, child in r_module.named_children():
                if child.__class__ in linear_policies:
                    setattr(
                        r_module,
                        name,
                        linear_policies[child.__class__](child,
                                                         prev_name + '.' + name,
                                                         conv_linear_layer))
                else:
                    update_mp_params(child)
                    _replace_module(child, name)
            return r_module

        return _replace_module(module)

    def replace_fn(child, _policy, layer_id=0):
        if training:
            # copy relevant state from child -> new module
            new_module = replace_with_policy(child, _policy, config.triangular_masking)

        else:
            # copy relevant state from child -> new module
            if not is_autotp_training_mode() and config.replace_with_kernel_inject:
                new_module = replace_with_policy(child,
                                                 _policy,
                                                 config.triangular_masking,
                                                 inference=True,
                                                 layer_id=layer_id)
            else:
                new_module = replace_wo_policy(child, _policy, prefix=prefix, state_dict=state_dict)

        return new_module

    replaced_module = replace_module(model=model,
                                     orig_class=orig_layer_impl,
                                     replace_fn=replace_fn,
                                     _replace_policy=policy)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if checkpoint_dict is not None:
        start_time = time.time()
        checkpoint = checkpoint_dict['checkpoints']
        ckpt_type = checkpoint_dict.get('parallelization', 'pp')
        ckpt_mp_size = checkpoint_dict.get('mp_size', mp_size)
        base_dir = checkpoint_dict.get('base_dir', '')

        if ckpt_type == 'pp':
            pbar = tqdm.tqdm(total=len(checkpoint),
                             desc=f"Loading {len(checkpoint)} checkpoint shards")
            for i in range(len(checkpoint)):
                sd = [
                    torch.load(os.path.join(base_dir1,
                                            checkpoint[i]),
                               map_location='cpu')
                ]
                load_model_with_checkpoint(
                    replaced_module,
                    sd,
                    mp_replace,
                    ckpt_type,
                    quantizer,
                )
                pbar.update(1)
        else:
            num_checkpoints = len(checkpoint) // ckpt_mp_size
            assert world_size >= ckpt_mp_size,\
                "Currently, merging checkpoints is not supported (when world_size is smaller than #checkpoints)!"
            checkpoint_stride = world_size // ckpt_mp_size
            if not deepspeed.comm.is_initialized() or deepspeed.comm.get_rank() == 0:
                pbar = tqdm.tqdm(total=num_checkpoints,
                                 desc=f"Loading {num_checkpoints} checkpoint shards")
            for i in range(num_checkpoints):
                if not deepspeed.comm.is_initialized() or deepspeed.comm.get_rank() == 0:
                    pbar.update(1)

                ckpt_index = i * ckpt_mp_size + (rank // checkpoint_stride)
                ckpt_file = os.path.join(
                    base_dir,
                    checkpoint[ckpt_index]) if base_dir else checkpoint[ckpt_index]
                sd = torch.load(ckpt_file, map_location='cpu')
                load_model_with_checkpoint(replaced_module,
                                           sd,
                                           mp_replace,
                                           ckpt_type,
                                           rank % (world_size // ckpt_mp_size))
        print(f"checkpoint loading time at rank {rank}: {time.time()-start_time} sec")

    if save_mp_checkpoint_path is not None:
        from collections import OrderedDict
        import json

        if checkpoint_dict is None:
            ckpt_name = "ds_model"
            try:
                from transformers.models.bloom.modeling_bloom import BloomForCausalLM
                if isinstance(model, BloomForCausalLM):
                    ckpt_name = "bloom"
            except ImportError:
                ckpt_name = "ds_model"
        else:
            ckpt_name = checkpoint_dict['type']
        if dist.is_initialized():
            dist.barrier()
        transformer_name = get_transformer_name(replaced_module)
        non_tp_ckpt_name = f'{ckpt_name}-non-tp.pt'
        ckpt_files = [non_tp_ckpt_name] * world_size
        os.makedirs(save_mp_checkpoint_path, exist_ok=True)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Saving tp-sharded checkpoints")
            torch.save(
                OrderedDict({
                    k: v
                    for k,
                    v in dict(replaced_module.state_dict()).items()
                    if transformer_name not in k
                }),
                f'{save_mp_checkpoint_path}/{non_tp_ckpt_name}')
            ckpt_files += [f'{ckpt_name}-tp_{r:0>2d}.pt' for r in range(world_size)]
            config = json.dumps({
                'type': ckpt_name,
                'base_dir': f'{save_mp_checkpoint_path}',
                'checkpoints': ckpt_files,
                'version': 1.0,
                'parallelization': 'tp',
                'mp_size': world_size
            })
            with open(f"{save_mp_checkpoint_path}/{ckpt_name}_ds-inference_config.json",
                      "w") as cfg:
                cfg.write(config)
        torch.save(
            OrderedDict({
                k: v
                for k,
                v in dict(replaced_module.state_dict()).items() if transformer_name in k
            }),
            f'{save_mp_checkpoint_path}/{ckpt_name}-tp_{rank:0>2d}.pt')

    if checkpoint is not None:
        pbar = tqdm.tqdm(total=len(checkpoint),
                         desc=f"Loading {len(checkpoint)} checkpoint shards")
        for i in range(len(checkpoint)):
            if not deepspeed.comm.is_initialized() or deepspeed.comm.get_rank() == 0:
                pbar.update(1)
            sd = torch.load(checkpoint[i], map_location='cpu')
            load_model_with_checkpoint(replaced_module, sd, mp_replace)
    return replaced_module


def revert_transformer_layer(orig_layer_impl, model, config, preln=False):
    """ Revert DeepSpeed's transformer layer back to original bert-style transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation that was replaced,
            e.g., transformers.models.bert.modeling_bert.BertLayer or transformers.BertLayer
        model (torch.nn.Module): user's nn.module representing their model
        config (dict): model config containing hidden size, attention heads, etc.
    Returns:
        Updated nn.module with original bert-style transformer layers
    """

    def replace_fn(child, _replace_policy, layer_id):
        #from turing.nvidia_modelingpreln import BertLayer
        orig_module = orig_layer_impl(config)

        # copy relevant state from child -> original module
        qkvw = child.attn_qkvw.data
        qkvb = child.attn_qkvb.data

        qw, kw, vw = torch.chunk(qkvw, 3, axis=0)
        qb, kb, vb = torch.chunk(qkvb, 3, axis=0)

        orig_module.attention.self.query.weight.data = qw
        orig_module.attention.self.query.bias.data = qb
        orig_module.attention.self.key.weight.data = kw
        orig_module.attention.self.key.bias.data = kb
        orig_module.attention.self.value.weight.data = vw
        orig_module.attention.self.value.bias.data = vb

        orig_module.attention.output.dense.weight.data = child.attn_ow.data
        orig_module.attention.output.dense.bias.data = child.attn_ob.data

        attn_ln_w = child.attn_nw.data
        attn_ln_b = child.attn_nb.data
        if preln:
            orig_module.PostAttentionLayerNorm.weight.data = attn_ln_w
            orig_module.PostAttentionLayerNorm.bias.data = attn_ln_b
        else:
            orig_module.attention.output.LayerNorm.weight.data = attn_ln_w
            orig_module.attention.output.LayerNorm.bias.data = attn_ln_b

        inter_ff_w = child.inter_w.data
        inter_ff_b = child.inter_b.data
        if preln:
            orig_module.intermediate.dense_act.weight.data = inter_ff_w
            orig_module.intermediate.dense_act.bias.data = inter_ff_b
        else:
            orig_module.intermediate.dense.weight.data = inter_ff_w
            orig_module.intermediate.dense.bias.data = inter_ff_b

        orig_module.output.dense.weight.data = child.output_w.data
        orig_module.output.dense.bias.data = child.output_b.data

        transformer_ln_w = child.norm_w.data
        transformer_ln_b = child.norm_b.data
        if preln:
            orig_module.PreAttentionLayerNorm.weight.data = transformer_ln_w
            orig_module.PreAttentionLayerNorm.bias.data = transformer_ln_b
        else:
            orig_module.output.LayerNorm.weight.data = transformer_ln_w
            orig_module.output.LayerNorm.bias.data = transformer_ln_b
        return orig_module

    return replace_module(model=model,
                          orig_class=deepspeed.DeepSpeedTransformerLayer,
                          replace_fn=replace_fn,
                          _replace_policy=None)


def replace_module(model, orig_class, replace_fn, _replace_policy, checkpoint=None):
    """ Scan the model for instances of ``orig_clas:`` to replace using ``replace_fn``.
    Arguments:
        model (torch.nn.Module): the model to augment
        orig_class (torch.nn.Module): the module to search for
        replace_fn (method): a method to convert instances of ``orig_class`` to the
                             desired type and return a new instance.
    Returns:
        A modified ``model``.
    """
    sd = None
    if checkpoint is not None:
        if checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(checkpoint)
        else:
            sd = torch.load(checkpoint, map_location='cpu', weights_only=False)

    policy = {}
    if orig_class is not None:
        policy.update({orig_class: (replace_fn, _replace_policy)})
    else:
        for plcy in replace_policies:
            # instantiate a throw-away policy in order to populate the _orig_layer_class
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    policy.update({orig_layer_class: (replace_fn, plcy)})
            elif plcy._orig_layer_class is not None:
                policy.update({plcy._orig_layer_class: (replace_fn, plcy)})
    assert len(policy.items()) > 0,\
        "No default policy found! Please specify your policy injection_policy (like {BertLayer:HFBEertLayerPolicy})." +\
        "You can find some samples here: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py"

    replaced_module, _ = _replace_module(model, policy, state_dict=sd)
    return replaced_module


from ..pipe import PipelineModule

import re


def skip_level_0_prefix(model, state_dict):
    model = str(model)
    key = re.search(r": (.*?)Model", model)
    if key is None:
        key = re.search(r": (.*?)Stack", model)
    if key is None:
        key = re.match(r"(.*?)Model", model)
    # if keys start with 'model.', don't skip level 0 prefix
    if state_dict is not None:
        for item in state_dict.keys():
            if re.match("^model[.]", item):
                return False
    if key is not None and key.group(1).lower() in ["bloom", "opt"]:
        return True
    return False


def _replace_module(model, policies, prefix='', layer_id=0, level_id=0, state_dict=None):
    """ Traverse model's children recursively and apply any transformations in ``policies``.
    Arguments:
        model (torch.nn.Module): model to augment
        policies (dict): Mapping of source class to replacement function.
    Returns:
        Modified ``model``.
    """
    for name, child in model.named_children():
        if child.__class__ in policies:
            replaced_module = policies[child.__class__][0](child,
                                                           policies[child.__class__][-1],
                                                           layer_id,
                                                           prefix=prefix + name,
                                                           state_dict=state_dict)
            setattr(model, name, replaced_module)
            if isinstance(model, PipelineModule):
                assert hasattr(model, 'forward_funcs'),\
                    "we require pipe-module to have the list of fwd_functions"
                model.forward_funcs[model.fwd_map[name]] = replaced_module
            layer_id += 1
        else:
            checking_key = prefix + name + '.'
            if Loading.is_load_module(child) and state_dict is not None:
                if any(checking_key in item for item in state_dict):
                    Loading.load(
                        child,
                        state_dict,
                        checking_key,
                    )
                else:
                    continue
            if len(child._buffers) != 0 and state_dict is not None:
                Loading.load_buffer(child, state_dict, checking_key)
            _, layer_id = _replace_module(child,
                                          policies,
                                          prefix if level_id == 0 and skip_level_0_prefix(model, state_dict) else \
                                          prefix + name + '.',
                                          layer_id=layer_id,
                                          level_id=level_id + 1,
                                          state_dict=state_dict)

    # Add the reset_cache func to the model, so that it can be called in the beginning of text-generation.
    model.reset_cache = transformer_inference.DeepSpeedTransformerInference.reset_cache
    return model, layer_id
