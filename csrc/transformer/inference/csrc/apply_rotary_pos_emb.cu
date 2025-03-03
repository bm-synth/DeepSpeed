// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#ifdef __HIP_PLATFORM_AMD__
#include "hip/hip_cooperative_groups.h"
#else
#include "cooperative_groups.h"
#endif
#include "ds_kernel_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

#ifndef __HIP_PLATFORM_AMD__
#include <cuda_profiler_api.h>
#endif

namespace cg = cooperative_groups;

namespace rot_half {
constexpr int threads = 256;
}  // namespace rot_half

template <typename T, int threadsPerHead, int granularity>
__global__ void apply_rotary_pos_half(T* mixed_query,
                                      T* key_layer,
                                      unsigned rotary_dim,
                                      unsigned seq_len,
                                      unsigned seq_offset,
                                      unsigned num_heads,
                                      unsigned head_size,
                                      unsigned total_count,
                                      float rope_theta,
                                      int max_out_tokens)
{
    constexpr int T_per_thread = granularity / sizeof(T);
    constexpr int heads_per_block = rot_half::threads / threadsPerHead;

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<threadsPerHead> head_group = cg::tiled_partition<threadsPerHead>(tb);

    const int head_idx = blockIdx.x * heads_per_block + threadIdx.x / threadsPerHead;
    const int cur_seq_idx = head_idx % seq_len;
    const int offset = head_idx * head_size;
    const int k_offset = (cur_seq_idx + (head_idx / seq_len) * max_out_tokens) * head_size;

    const int seq_idx = cur_seq_idx + seq_offset;
    const int half_dim = rotary_dim >> 1;
    const int half_dim_threads = half_dim / T_per_thread;

    if (head_idx < total_count) {
        const int base_neuron_idx = head_group.thread_rank() * T_per_thread;

        T q[T_per_thread], k[T_per_thread];
        mem_access::load_global<granularity>(q, mixed_query + offset + base_neuron_idx);
        mem_access::load_global<granularity>(k, key_layer + k_offset + base_neuron_idx);

#pragma unroll
        for (int i = 0; i < T_per_thread; i++) {
            const int neuron_idx = base_neuron_idx + i;
            if (neuron_idx < rotary_dim) {
                float inv_freq = (float)((neuron_idx % half_dim) * 2) / (float)rotary_dim;
                inv_freq = 1.0 / powf(rope_theta, inv_freq) * (float)seq_idx;

                float rotary_sign = (neuron_idx > (half_dim - 1) ? -1.0 : 1.0);
                float q_rot = conversion::to<float>(q[i]) * rotary_sign;
                float k_rot = conversion::to<float>(k[i]) * rotary_sign;

                const int target_lane = (neuron_idx < half_dim)
                                            ? head_group.thread_rank() + half_dim_threads
                                            : head_group.thread_rank() - half_dim_threads;

                const float q_rot_temp = head_group.shfl(q_rot, target_lane);
                const float k_rot_temp = head_group.shfl(k_rot, target_lane);

                q[i] = conversion::to<T>(conversion::to<float>(q[i]) * cosf(inv_freq) +
                                         q_rot_temp * sinf(inv_freq));
                k[i] = conversion::to<T>(conversion::to<float>(k[i]) * cosf(inv_freq) +
                                         k_rot_temp * sinf(inv_freq));
            }
        }

        mem_access::store_global<granularity>(mixed_query + offset + base_neuron_idx, q);
        mem_access::store_global<granularity>(key_layer + k_offset + base_neuron_idx, k);
    }
}

__global__ void apply_rotary_pos_emb(__half* mixed_query,
                                     __half* key_layer,
                                     unsigned rotary_dim,
                                     unsigned seq_len,
                                     unsigned seq_offset,
                                     unsigned num_heads,
                                     unsigned head_size,
                                     unsigned total_count)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = (float)mixed_query[offset + lane];
            float k = (float)key_layer[offset + lane];
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = g.shfl_xor(q_rot, 1);
            k_rot = g.shfl_xor(k_rot, 1);
            q = q * cosf(inv_freq) + q_rot * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot * sinf(inv_freq);

            mixed_query[offset + lane] = (__half)q;
            key_layer[offset + lane] = (__half)k;

            lane += WARP_SIZE;
        }
    }
}
__global__ void apply_rotary_pos_emb1(float* mixed_query,
                                      float* key_layer,
                                      unsigned rotary_dim,
                                      unsigned seq_len,
                                      unsigned seq_offset,
                                      unsigned num_heads,
                                      unsigned head_size,
                                      unsigned total_count,
                                      int max_out_tokens)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;
    unsigned seq_index = head_id % seq_len;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = mixed_query[offset + lane];
            float k = key_layer[k_offset + lane];
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = g.shfl_xor(q_rot, 1);
            k_rot = g.shfl_xor(k_rot, 1);
            q = q * cosf(inv_freq) + q_rot * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot * sinf(inv_freq);

            mixed_query[offset + lane] = q;
            key_layer[k_offset + lane] = k;

            lane += WARP_SIZE;
        }
    }
}
__global__ void apply_rotary_pos_emb1(__half* mixed_query,
                                      __half* key_layer,
                                      unsigned rotary_dim,
                                      unsigned seq_len,
                                      unsigned seq_offset,
                                      unsigned num_heads,
                                      unsigned head_size,
                                      unsigned total_count,
                                      int max_out_tokens)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned seq_index = head_id % seq_len;
    unsigned offset = head_id * head_size;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    constexpr unsigned mask[32] = {
        0x1 | 0x1000,     0x2 | 0x2000,     0x4 | 0x4000,     0x8 | 0x8000,     0x10 | 0x10000,
        0x20 | 0x20000,   0x40 | 0x40000,   0x80 | 0x80000,   0x100 | 0x100000, 0x200 | 0x200000,
        0x400 | 0x400000, 0x800 | 0x800000, 0x1000 | 0x1,     0x2000 | 0x2,     0x4000 | 0x4,
        0x8000 | 0x8,     0x10000 | 0x10,   0x20000 | 0x20,   0x40000 | 0x40,   0x80000 | 0x80,
        0x100000 | 0x100, 0x200000 | 0x200, 0x400000 | 0x400, 0x800000 | 0x800, 0x1000000,
        0x2000000,        0x4000000,        0x8000000,        0x10000000,       0x20000000,
        0x40000000,       0x80000000};

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;
    unsigned half_dim = rotary_dim >> 1;
    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane % half_dim) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = (float)mixed_query[offset + lane];
            float k = (float)key_layer[k_offset + lane];
            float rotary_sign = (lane > (half_dim - 1) ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            auto q_rot_tmp = lane < half_dim ? __shfl_sync(mask[lane], q_rot, lane + half_dim)
                                             : __shfl_sync(mask[lane], q_rot, lane - half_dim);
            auto k_rot_tmp = lane < half_dim ? __shfl_sync(mask[lane], k_rot, lane + half_dim)
                                             : __shfl_sync(mask[lane], k_rot, lane - half_dim);
            q = q * cosf(inv_freq) + q_rot_tmp * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot_tmp * sinf(inv_freq);

            mixed_query[offset + lane] = (__half)q;
            key_layer[k_offset + lane] = (__half)k;

            lane += WARP_SIZE;
        }
    }
}

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 float rope_theta,
                                 cudaStream_t stream,
                                 int max_out_tokens)
{
    const int half_dim = rotary_dim >> 1;

    int alignment = sizeof(T);
    if (half_dim % (16 / sizeof(T)) == 0) {
        alignment = 16;
    } else if (half_dim % (8 / sizeof(T)) == 0) {
        alignment = 8;
    } else if (half_dim % (4 / sizeof(T)) == 0) {
        alignment = 4;
    } else {
        assert(false);
    }
    const int T_per_elem = alignment / sizeof(T);

    int total_count = batch * num_heads * seq_len;

    const int padded_head_size = next_pow2(head_size);

    assert(padded_head_size <= hw_warp_size * T_per_elem);

    const int threads_per_head = padded_head_size / T_per_elem;
    const int heads_per_block = rot_half::threads / threads_per_head;

    dim3 block(rot_half::threads);
    dim3 grid((total_count + heads_per_block - 1) / heads_per_block);

    if (alignment == 4) {
        LAUNCH_FOR_ALIGNMENT(4);
    } else if (alignment == 8) {
        LAUNCH_FOR_ALIGNMENT(8);
    } else if (alignment == 16) {
        LAUNCH_FOR_ALIGNMENT(16);
    } else {
        assert(false);
    }
}

#define INSTANTIATE_LAUNCH_ROTARY_POS_EMB(T)                   \
    template void launch_apply_rotary_pos_emb<T>(T*,           \
                                                 T*,           \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 float,        \
                                                 cudaStream_t, \
                                                 int);

INSTANTIATE_LAUNCH_ROTARY_POS_EMB(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_ROTARY_POS_EMB(__nv_bfloat16);
#endif
INSTANTIATE_LAUNCH_ROTARY_POS_EMB(__half);
