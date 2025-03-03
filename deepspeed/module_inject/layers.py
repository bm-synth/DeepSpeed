# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from deepspeed.accelerator import get_accelerator


class LinearAllreduce(nn.Module):

    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        output = RowParallel.apply(self.mp_group, output, not self.is_training_mode())
        if self.bias is not None:
            output += self.bias
        return output

    @torch.no_grad()
    def gather_params(self, params_list):

        for idx, param in enumerate(params_list):
            if param is None or idx > 0:
                # don't gather bias
                return
            params_list[idx].data_partition = param.data
            param = param.transpose(0, 1).contiguous()
            output_param = torch.empty(self.tp_world_size * param.shape[0],
                                       param.shape[1],
                                       dtype=param.dtype,
                                       device=param.device)
            dist.all_gather_into_tensor(output_param, param, group=self.mp_group)
            params_list[idx].data = output_param.transpose(0, 1).contiguous()
        return

    @torch.no_grad()
    def _tp_partition(self, params_list):

        if not self.is_training_mode():
            self.uneven_partition(params_list)
            return

        else:
            for idx, param in enumerate(params_list):
                if param is None or idx > 0:
                    # don't slipt bias
                    return
                _partition = torch.chunk(param, self.tp_world_size, dim=-1)[self.tp_index]

                _partition = move(_partition, get_accelerator().current_device_name()).detach()

                params_list[idx].data = _partition

    def uneven_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None or idx > 0:
                # don't slipt bias
                return
            assert self.name is not None, "The module name must be provided in the initialization."
            _partition = params_list[idx].split(get_shard_size_list(params_list[idx].shape[1], self.tp_world_size,
                                                                    self.name),
                                                dim=1)[self.tp_index]

            _partition = move(_partition, get_accelerator().current_device_name()).detach()
            params_list[idx].data = _partition


#remove kwargs from partition.
class LinearLayer(TensorParallel_Layer):

    def __init__(self, module, mp_group=None, skip_partition=False, **kwargs):
        super(LinearLayer, self).__init__(mp_group, **kwargs)
        self.weight = module.weight
        self.bias = module.bias
        if not skip_partition:
            self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            self.config_tp_params(self.bias)

    def forward(self, input):
        if getattr(self, 'mp_group', None) is not None:
            input = ColumnParallel.apply(self.mp_group, input)
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output

    @torch.no_grad()
    def gather_params(self, params_list):
        #  Does not support uneven shard.
        for idx, param in enumerate(params_list):

            params_list[idx].data_partition = param.data
            output_param = torch.empty(self.tp_world_size * param.shape[0],
                                       param.shape[1],
                                       dtype=param.dtype,
                                       device=param.device)
            dist.all_gather_into_tensor(output_param, param, group=self.mp_group)
            params_list[idx].data = output_param.contiguous()

    @torch.no_grad()
    def _tp_partition(self, params_list):

        if not self.is_training_mode():
            self.uneven_partition(params_list)
            return
        for idx, param in enumerate(params_list):
            if param is None:
                return
            #split bias if provide
            _partition = torch.chunk(param, self.tp_world_size, dim=0)[self.tp_index]

            _partition = move(_partition, get_accelerator().current_device_name()).detach()

            params_list[idx].data = _partition

    def uneven_partition(self, params_list):

        for idx, param in enumerate(params_list):
            if param is None:
                #split bias if provide
                return
            assert self.name is not None, "The module name must be provided in the initialization."
            _partition = params_list[idx].split(get_shard_size_list(params_list[idx].shape[0], self.tp_world_size,
                                                                    self.name),
                                                dim=0)[self.tp_index]

            _partition = move(_partition, get_accelerator().current_device_name()).detach()

            params_list[idx].data = _partition

    # for bwc
    @classmethod
    def from_weights(cls, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        if weight is not None:
            in_features = weight.shape[1]
            out_features = weight.shape[0]
            linear = nn.Linear(in_features, out_features, bias=(bias is not None))
            linear.weight.data = weight
            if bias is not None:
                linear.bias.data = bias
        else:
            in_features = weight_shape[1]
            out_features = weight_shape[0]
            linear = nn.Linear(in_features, out_features, bias=(bias is not None))
        return cls(linear, skip_partition=True)


class FusedModuleWrapper:

    def __init__(self, fused_module: nn.Module):
        self.fused_module = fused_module

    def __getattr__(self, module):
        return self.fused_module


class fused_LinearLayer(LinearLayer):

    def __init__(self, module, mp_group, skip_partition=False, **kwargs):
        assert kwargs.get('fused_module') is not None, "'fused_module' is required but not provided"
        # Use the warp class to avoid module circular references.
        self.fused_module = FusedModuleWrapper(kwargs.get('fused_module'))
        super().__init__(module, mp_group, skip_partition, **kwargs)

    @torch.no_grad()
    def _tp_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None:
                return

            _partition = prepare_tp_fused_qkvw(self.fused_module.module, param, self.tp_world_size, self.tp_index)

            _partition = move(_partition, get_accelerator().current_device_name()).detach()

            params_list[idx].data = _partition


class conv_LinearLayer(LinearLayer):

    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight = None
        bias = None
        if len(params_list) == 1:
            weight = params_list[0]
        elif len(params_list) == 2:
            weight, bias = params_list[0], params_list[1]
        _partition = weight.data.split(get_shard_size_list(weight.shape[0], self.tp_world_size, self.name),
                                       dim=1)[self.tp_index]
        _partition = move(_partition, get_accelerator().current_device_name()).detach()
        weight.data = _partition

        if bias is not None:
            _partition = bias.data.split(get_shard_size_list(weight.shape[1], self.tp_world_size, self.name),
                                         dim=0)[self.tp_index]
            _partition = move(_partition, get_accelerator().current_device_name()).detach()

            bias.data = _partition


#override the subclasses related to weight splitting.
class Yuan_LinearAllreduce(LinearAllreduce):

    #Yuan2
    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight, bias = shard_value_with_share_qk(params_list[0].data, params_list[1], self.tp_index,
                                                 self.tp_world_size, False)
        params_list[0].data = weight
        if bias is not None:
            params_list[1].data = bias


class Yuan_LinearLayer(LinearLayer):
    #Yuan2
    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight, bias = shard_value_with_share_qk(params_list[0].data, params_list[1], self.tp_index,
                                                 self.tp_world_size, True)
        params_list[0].data = move(weight, get_accelerator().current_device_name()).detach()
        if bias is not None:
            params_list[1].data = move(bias, get_accelerator().current_device_name()).detach()


class GateUpPack_LinearLayer(LinearLayer):
    # chatGLM2, chatGLM2
    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight, bias = shard_chunk_mlp(params_list[0].data, params_list[1], self.tp_index, self.tp_world_size)
        params_list[0].data = move(weight, device=get_accelerator().current_device_name()).detach()
        if bias is not None:
            params_list[1].data = move(bias, device=get_accelerator().current_device_name()).detach()


class Conv_LinearALlreduce(LinearAllreduce):

    @torch.no_grad()
    def _tp_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None:
                return
            param.data = param.data.transpose(-1, -2).contiguous()

            _partition = param.split(get_shard_size_list(param.shape[0], self.tp_world_size, self.name),
                                     dim=1)[self.tp_index]

            _partition = move(_partition, get_accelerator().current_device_name()).detach()

            params_list[idx].data = _partition


#override the subclasses related to fwd/bwd.
class LmHeadLinearAllreduce(LinearAllreduce):

    def __init__(self, module, mp_group, **kwargs):
        # set the fixed name before partition
        self.name = "lm_head"

        # In some tied_embedding cases, only the lm head is sharded, while the word embedding is not.
        # Reinitialization is used to decouple them and prevent the word embedding from being sharded.
        # This should also be effective for cases where both are sharded in tied_embedding scenarios.

        # TODO: Training scenario-related tests, is it necessary to re-implement the vocab parallel module?
        module.weight = nn.Parameter(module.weight.clone().detach())
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias = nn.Parameter(module.bias.clone().detach())
        super().__init__(module, mp_group, **kwargs)

    def forward(self, input):
        input_shard_size = get_shard_size(input.shape[-1], self.tp_world_size, "lm_head")
        input_shard_offset = sum(get_shard_size_list(input.shape[-1], self.tp_world_size, "lm_head")[0:self.tp_index])
        output = torch.matmul(input[:, :, input_shard_offset:input_shard_offset + input_shard_size],
                              self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.inference_all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class LinearLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.weight = Parameter(
                torch.empty(weight_shape, dtype=dtype, device=get_accelerator().current_device_name()))

            self.bias = Parameter(
                torch.empty(weight_shape[0],
                            dtype=dtype,
                            device=get_accelerator().current_device_name())) \
                if bias is not None else None

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output


class Normalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None, bias=None):
        super(Normalize, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.norm = nn.LayerNorm(dim, eps=eps).to(dtype).to(get_accelerator().current_device_name())
            self.weight = self.norm.weight
            self.bias = self.norm.bias

        self.eps = eps

    def forward(self, input):
        return nn.functional.layer_norm(input, input.shape[-1:], self.weight, self.bias, eps=self.eps)


class EmbeddingLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(EmbeddingLayer, self).__init__()
        if weight is None:
            self.weight = Parameter(
                torch.empty(weight_shape[0],
                            weight_shape[1],
                            dtype=dtype,
                            device=get_accelerator().current_device_name()))
        else:
            self.weight = weight

    def forward(self, input):
        return F.embedding(input, self.weight)


class OPTEmbedding(EmbeddingLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, weight_shape=None, weight=None, bias=None):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(weight_shape, weight=weight)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)
