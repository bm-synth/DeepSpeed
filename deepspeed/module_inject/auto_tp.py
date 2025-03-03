# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Automatic Tensor Parallelism
import re

from torch import nn
from .replace_policy import replace_policies


class AutoTP():

    def in_module_list(module, module_list):
        for item in module_list:
            if type(item).__name__ == type(module).__name__:
                return True
        return False

    def get_module_list(model):
        mlist = []
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                for module in child.children():
                    if not mlist:
                        mlist = [module]
                    elif not AutoTP.in_module_list(module, mlist):
                        mlist = mlist + [module]
            else:
                mlist = mlist + AutoTP.get_module_list(child)
        return mlist

    def supported(model):
        unsupported = ['codegen', 'deberta', 'flaubert', 'fsmt', 'gpt2', 'led', 'longformer', 'xlm', 'xlnet']
        model = str(model)
        key = re.search(r": (.*?)Model", model)
        if key is None:
            key = re.search(r": (.*?)Stack", model)
        if key is None:
            key = re.match(r"(.*?)Model", model)
        assert key is not None, "Not able to determine model policy automatically. Please provide policy."
        if key.group(1).lower() in unsupported:
            return False
        return True

    def get_layers(parent, module):
        layer_list = []
        for key, submodule in module._modules.items():
            if isinstance(submodule, nn.Linear):
                layer_list = layer_list + [parent + "." + key]
            elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm' or key == 'layer_norm':
                layer_list = layer_list + ["ln"]
            else:
                layer_list = layer_list + AutoTP.get_layers(key, submodule)
        return layer_list

    def update_policy_list(policy_list, new_module, new_gems):
        if len(policy_list):
            for i, policy in enumerate(policy_list):
                # if module already exists in policy, combine gems and remove duplicates
                if policy[0] == type(new_module):
                    new_gems = set(new_gems + policy[1])
                    policy_list[i] = tuple([type(new_module), new_gems])
                    return policy_list
        policy_list.append(tuple([type(new_module), new_gems]))
        return policy_list

    def kernel_supported(module_list):
        policy = []
        for plcy in replace_policies:
            # instantiate a throw-away policy in order to populate the _orig_layer_class
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    policy.append(orig_layer_class)
            elif plcy._orig_layer_class is not None:
                policy.append(plcy._orig_layer_class)
        for child in module_list:
            if child.__class__ in policy:
                return True
        return False

    def tp_parser(model):
        policy_list = []
        module_list = []
        layer_list = []
        gem_list = []

        module_list = AutoTP.get_module_list(model)
        assert AutoTP.supported(model), "AutoTP not supported for model. Please use kernel injection since container policy for model exists." \
        if AutoTP.kernel_supported(module_list) else "AutoTP not supported for model. Please provide policy."
        for module in module_list:
            for key, submodule in module._modules.items():
                if isinstance(submodule, nn.Linear):
                    layer_list = layer_list + ["." + key]
                elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm' or key == 'layer_norm':
                    layer_list = layer_list + ["ln"]
                else:
                    layer_list = layer_list + AutoTP.get_layers(key, submodule)
            for i, layer in enumerate(layer_list):
                if layer == 'ln':
                    if layer_list[i - 1] != 'ln':
                        gem_list = gem_list + [layer_list[i - 1]]
                elif 'out_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'o_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'down_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'self_attention.dense' in layer and 'falcon' in str(
                        type(module)):  # this is a hack to get the right linear layer for this model!
                    gem_list = gem_list + [layer]

            layer_list = []
            if gem_list != []:
                gem_list = list(set(gem_list))
                policy_list = AutoTP.update_policy_list(policy_list, module, gem_list)
                gem_list = []
        assert len(policy_list), "AutoTP not supported for model. Please use kernel injection since container policy for model exists." \
        if AutoTP.kernel_supported(module_list) else "Not able to determine model policy automatically. Please provide policy."
        return policy_list

    def set_tensor_parallel_config(self, mp_size, mp_group):

        if is_autotp_training_mode():
            self.mp_group = groups.get_tensor_model_parallel_group()
            self.mp_size = groups.get_tensor_model_parallel_world_size()
            return

        self.mp_size = mp_size
        self.mp_group = mp_group

    def _replace(self, child, name, conv_linear_layer):
        # This function should clearly define the routing rules for specific layers
        # and avoid any complex shard-related logic.
        if getattr(child, "replaced", False) == True:
            return
        device_name = 'cpu' if self.keep_module_on_host else get_accelerator().current_device_name()
        # keep_module_on_host is used to keep the module on the host. Checkpoints are loaded to the host first (in some
        # cases it can be done from the disk even to prevent filling host's memory), thus no need to create a new copy.
        return_new_copy = not self.keep_module_on_host
        weight_shape = child.weight.shape
        mp_replace = ReplaceWithTensorSlicing(mp_group=self.mp_group)
        # For TP layer skip, e.g., MoE gate, deepseek low rank layer skip
        if "q_a_proj" in name or "kv_a_proj_with_mqa" in name or name == "block_sparse_moe.gate" or (
            ('mlp.shared_expert_gate' == name or 'mlp.gate' == name) and 'qwen2_moe' in str(type(self.module))):
            return child
        # For Yuan model
        if 'Yuan' in str(self.module):
            if 'v_proj' in name:
                return Yuan_LinearLayer(child, self.mp_group)

            elif 'o_proj' in name:
                return Yuan_LinearAllreduce(child, self.mp_group)

        # For MLP including chunk layer.
        if 'gate_up_proj' in name or ('dense_h_to_4h' in name and 'GLM' in str(self.module)):
            return GateUpPack_LinearLayer(child, self.mp_group)
            # For Arctic model, bypass to all_reduce replacement for w2 weights
        arctic_w2_all_reduce_linear = False
        if 'Arctic' in str(self.module) and 'w2' in name:
            arctic_w2_all_reduce_linear = True
        # For MoE MLP model, e.g., deepseek and jamba
        down_proj = False
        if 'down_proj' in name:
            down_proj = True
        if name in self.all_reduce_linears or arctic_w2_all_reduce_linear or down_proj:

            setattr(child, "replaced", True)
            if self.conv_linear_layer:
                return Conv_LinearALlreduce(child, self.mp_group, name=name)
            elif name == "lm_head" or name == 'embed_out':
                return LmHeadLinearAllreduce(child, self.mp_group)

            return LinearAllreduce(child, self.mp_group, name=name)
        else:

            setattr(child, "replaced", True)
            if self.conv_linear_layer:
                conv_LinearLayer(child, self.mp_group)
            elif require_tp_fused_qkvw(name, self.mp_size):
                #Check and handle fused qkv for TP
                return fused_LinearLayer(child, self.mp_group, fused_module=self.module)

            return LinearLayer(child, self.mp_group, name=name)

    def _slice_embedding(self, child, name, conv_linear_layer):
        if getattr(child, "replaced", False) == True:
            return
        mp_replace = ReplaceWithTensorSlicing(mp_group=self.mp_group)

        if hasattr(child.weight, 'ds_tensor'):
            data = child.weight.ds_tensor.data.split(get_shard_size_list(child.weight.shape[1], self.mp_size), dim=1)
        else:
            data = child.weight.data.split(get_shard_size_list(child.weight.shape[1], self.mp_size, name), dim=1)
        data = data[mp_replace.gpu_index].to(get_accelerator().current_device_name())
        data = torch.nn.parameter.Parameter(data, requires_grad=False)

        new_embedding = nn.Embedding(child.weight.shape[0], get_shard_size(child.weight.shape[1], self.mp_size, name))
        new_embedding.weight.data.copy_(data)
        setattr(child, "replaced", True)
        return new_embedding

    def update_mp_params(self, child):
        if getattr(child, "replaced", False) == True:
            return
        param_list = [
            "n_heads", "inner_dim", "num_heads", "num_kv", "num_attention_heads", "num_attn_heads", "all_head_size",
            "embed_dim", "hidden_size", "num_key_value_heads", "num_kv_heads", "kv_n_heads", "d_model",
            "num_attention_heads_per_partition", "num_multi_query_groups_per_partition", "hidden_size_per_partition"
        ]
        for param in param_list:
            if "Yuan" in str(child) and 'embed_dim' in param_list:
                param_list.remove('embed_dim')
            if hasattr(child, param):
                param_val = getattr(child, param)
                setattr(child, param, get_shard_size(param_val, self.mp_size))
        setattr(child, "replaced", True)

    def update_linear_policies(self):
        self.conv_linear_layer = False
        if self.linear_layer_setting is not None:
            self.linear_policies = {self.linear_layer_setting[0]: self._replace}
            if len(self.linear_layer_setting) == 2:
                self.linear_policies.update({self.linear_layer_setting[1]: self._slice_embedding})
        else:
            import transformers
            if self.orig_layer_impl is transformers.models.gpt2.modeling_gpt2.GPT2Block:
                try:
                    self.conv_linear_layer = True
                    self.linear_policies = {transformers.pytorch_utils.Conv1D: self._replace}
                except ImportError:
                    self.linear_policies = {nn.Linear: self._replace}
            else:
                self.linear_policies = {nn.Linear: self._replace, nn.Embedding: self._slice_embedding}

    def _replace_module(self, r_module, prev_name='', prev_class_name=''):
        for name, child in r_module.named_children():
            if prev_class_name == "":
                class_name = prev_name
            elif prev_name == "":
                class_name = prev_class_name
            else:
                class_name = prev_class_name + '.' + prev_name
            checking_key = self.prefix + '.' + class_name + '.' + name + '.' if class_name != "" else self.prefix + '.' + name + '.'
            if Loading.is_load_module(child) and self.state_dict is not None:
                if any(checking_key in item for item in self.state_dict):
                    Loading.load(child, self.state_dict, checking_key, self.mp_group)
                else:
                    continue
            if len(child._buffers) != 0 and self.state_dict is not None:
                Loading.load_buffer(child, self.state_dict, checking_key)
            if child.__class__ in self.linear_policies:
                setattr(r_module, name, self.linear_policies[child.__class__](child, prev_name + '.' + name,
                                                                              self.conv_linear_layer))
            elif any(isinstance(child, lp) for lp in self.linear_policies):
                # Added for falcon model support
                # Note: isinstance will account for class inheritance, child.__class__ does not
                key = None
                for lp in self.linear_policies:
                    if isinstance(child, lp):
                        key = lp
                        break
                assert key is not None
                setattr(r_module, name, self.linear_policies[key](child, prev_name + '.' + name,
                                                                  self.conv_linear_layer))
            else:
                self.update_mp_params(child)
                self._replace_module(child, name, class_name)
        return r_module

    def get_model_num_kv_heads(self, config):
        num_kv_heads = None
        # multi_query_group_num is for chatglm2 & chatglm3
        kv_head_names = [
            'multi_query_group_num', 'num_kv_heads', 'num_key_value_heads', 'num_attention_heads', 'n_heads',
            'attention_heads'
        ]
        for name in kv_head_names:
            if hasattr(config, name):
                num_kv_heads = getattr(config, name)
                if num_kv_heads is not None:
                    break
        return num_kv_heads

    def _replace_last_linear_module(self, r_module):
        if hasattr(r_module, "lm_head"):
            name = "lm_head"
            child = r_module.lm_head
        elif hasattr(r_module, "embed_out"):
            name = "embed_out"
            child = r_module.embed_out
        else:
            return r_module
        if child.__class__ in self.linear_policies:
            setattr(r_module, name, self.linear_policies[child.__class__](child, name, self.conv_linear_layer))
        return r_module
