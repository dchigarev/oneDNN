/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GRAPH_EXAMPLE_UTILS_HPP
#define GRAPH_EXAMPLE_UTILS_HPP

#include "oneapi/dnnl/dnnl_graph.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

// inline void add_int8_mlp_subgraph(graph_t *agraph,
//         std::vector<graph::dim_t> batch_dims = {1, 384}, graph::dim_t layer = 1,
//         std::vector<graph::dim_t> hidden_size = {1024, 1024},
//         std::vector<op_kind_t> act_type = {op_kind::ReLU},
//         bool mixed_dtype = false, bool dyn_quant = false) {
//     size_t lt_idx = 0;
//     size_t op_idx = 0;

//     auto dtype = mixed_dtype ? graph::data_type::bf16 : graph::data_type::f32;
//     std::vector<graph::dim_t> layer_input_size(batch_dims);
//     layer_input_size.push_back(hidden_size[0]);
//     graph::logical_tensor_t input_desc
//             = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);

//     for (graph::dim_t i = 0; i < layer; ++i) {
//         std::vector<graph::dim_t> layer_weight_size {
//                 hidden_size[i], hidden_size[i + 1]};
//         std::vector<graph::dim_t> layer_bias_size {hidden_size[i + 1]};
//         std::vector<graph::dim_t> layer_dst_size(batch_dims);
//         layer_dst_size.push_back(hidden_size[i + 1]);
//         // creating logical tensors of each layer
//         graph::logical_tensor_t casted_input_desc, quant_input_desc,
//                 dequant_input_desc, casted_dequant_input_desc,
//                 quant_weight_desc, dequant_weight_desc,
//                 casted_dequant_weight_desc, bias_desc, matmul_dst_desc,
//                 act_dst_desc;
//         // logical tensors for dynamic quantization.
//         graph::logical_tensor_t input_scale_desc, input_zp_desc,
//                 weight_scale_desc;
//         casted_input_desc = utils::logical_tensor_init(
//                 lt_idx++, layer_input_size, graph::data_type::f32);
//         quant_input_desc = utils::logical_tensor_init(
//                 lt_idx++, layer_input_size, graph::data_type::u8);
//         dequant_input_desc = utils::logical_tensor_init(
//                 lt_idx++, layer_input_size, graph::data_type::f32);
//         casted_dequant_input_desc
//                 = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);
//         quant_weight_desc = utils::logical_tensor_init(
//                 lt_idx++, layer_weight_size, graph::data_type::s8);
//         quant_weight_desc.property = property_type::constant;
//         dequant_weight_desc = utils::logical_tensor_init(
//                 lt_idx++, layer_weight_size, graph::data_type::f32);
//         casted_dequant_weight_desc = utils::logical_tensor_init(
//                 lt_idx++, layer_weight_size, dtype);
//         bias_desc
//                 = utils::logical_tensor_init(lt_idx++, layer_bias_size, dtype);
//         bias_desc.property = property_type::constant;
//         matmul_dst_desc
//                 = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
//         act_dst_desc
//                 = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
//         if (dyn_quant) {
//             input_scale_desc = utils::logical_tensor_init(
//                     lt_idx++, {1}, graph::data_type::f32);
//             input_zp_desc = utils::logical_tensor_init(
//                     lt_idx++, {1}, graph::data_type::s32);
//             weight_scale_desc = utils::logical_tensor_init(
//                     lt_idx++, {hidden_size[i + 1]}, graph::data_type::f32);
//         }
//         // defining ops of each layer
//         std::string layer_suffix = "_layer" + std::to_string(i);
//         graph::op_t typecast_input_f32 {op_idx++, graph::op_kind::TypeCast,
//                 "typecast_input_f32" + layer_suffix};
//         graph::op_t quant_input {op_idx++,
//                 dyn_quant ? graph::op_kind::DynamicQuantize
//                           : graph::op_kind::Quantize,
//                 "quantize_input" + layer_suffix};
//         if (dyn_quant) {
//             DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(quant_input);
//         } else {
//             DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quant_input);
//         }
//         graph::op_t dequant_input {op_idx++,
//                 dyn_quant ? graph::op_kind::DynamicDequantize
//                           : graph::op_kind::Dequantize,
//                 "dequantize_input" + layer_suffix};
//         if (dyn_quant) {
//             DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequant_input);
//         } else {
//             DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequant_input);
//         }
//         graph::op_t typecast_input_bf16 {op_idx++, graph::op_kind::TypeCast,
//                 "typecast_input_bf16" + layer_suffix};
//         graph::op_t dequant_weight {op_idx++,
//                 dyn_quant ? graph::op_kind::DynamicDequantize
//                           : graph::op_kind::Dequantize,
//                 "dequantize_weight" + layer_suffix};
//         if (dyn_quant) {
//             DEFINE_DEFAULT_PER_CHANNEL_DYN_QUANT_ATTR(
//                     dequant_weight, hidden_size[i + 1], 1);
//         } else {
//             DEFINE_DEFAULT_PER_CHANNEL_QUANT_ATTR(
//                     dequant_weight, hidden_size[i + 1], 1);
//         }
//         graph::op_t typecast_weight_bf16 {op_idx++, graph::op_kind::TypeCast,
//                 "typecast_input_bf16" + layer_suffix};
//         graph::op_t matmul {
//                 op_idx++, graph::op_kind::MatMul, "matmul" + layer_suffix};
//         graph::op_t activation {
//                 op_idx++, act_type[i], "activation" + layer_suffix};
//         // defining op connection of each layer
//         quant_input.add_output(quant_input_desc);
//         dequant_input.add_input(quant_input_desc);
//         dequant_input.add_output(dequant_input_desc);
//         dequant_weight.add_input(quant_weight_desc);
//         dequant_weight.add_output(dequant_weight_desc);
//         if (mixed_dtype) {
//             typecast_input_f32.add_input(input_desc);
//             typecast_input_f32.add_output(casted_input_desc);
//             quant_input.add_input(casted_input_desc);
//             typecast_input_bf16.add_input(dequant_input_desc);
//             typecast_input_bf16.add_output(casted_dequant_input_desc);
//             typecast_weight_bf16.add_input(dequant_weight_desc);
//             typecast_weight_bf16.add_output(casted_dequant_weight_desc);
//             matmul.add_input(casted_dequant_input_desc);
//             matmul.add_input(casted_dequant_weight_desc);
//         } else {
//             quant_input.add_input(input_desc);
//             matmul.add_input(dequant_input_desc);
//             matmul.add_input(dequant_weight_desc);
//         }
//         if (dyn_quant) {
//             quant_input.add_input(input_scale_desc);
//             quant_input.add_input(input_zp_desc);
//             dequant_input.add_input(input_scale_desc);
//             dequant_input.add_input(input_zp_desc);
//             dequant_weight.add_input(weight_scale_desc);
//         }

//         // matmul.add_input(bias_desc);
//         matmul.add_output(matmul_dst_desc);

//         if (act_type[i] == graph::op_kind::Wildcard) {
//             input_desc = matmul_dst_desc;
//         } else {
//             activation.add_input(matmul_dst_desc);
//             activation.add_output(act_dst_desc);
//             input_desc = act_dst_desc;
//         }
//         // adding ops of each layer
//         if (mixed_dtype) {
//             agraph->add_op(&typecast_input_f32);
//             agraph->add_op(&typecast_input_bf16);
//             agraph->add_op(&typecast_weight_bf16);
//         }
//         agraph->add_op(&quant_input);
//         agraph->add_op(&dequant_input);
//         agraph->add_op(&dequant_weight);
//         agraph->add_op(&matmul);
//         if (act_type[i] != graph::op_kind::Wildcard) {
//             agraph->add_op(&activation);
//         }

//         layer_input_size = layer_dst_size;
//     }

//     // defining output layer logical tensors
//     graph::logical_tensor_t casted_output_desc = utils::logical_tensor_init(
//             lt_idx++, layer_input_size, graph::data_type::f32);
//     graph::logical_tensor_t quant_output_desc = utils::logical_tensor_init(
//             lt_idx++, layer_input_size, graph::data_type::u8);
//     graph::logical_tensor_t dequant_output_desc = utils::logical_tensor_init(
//             lt_idx++, layer_input_size, graph::data_type::f32);
//     graph::logical_tensor_t casted_dequant_output_desc
//             = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);
//     // defining logical tensors for dynamic quantization.
//     graph::logical_tensor_t output_scale_desc
//             = utils::logical_tensor_init(lt_idx++, {1}, graph::data_type::f32);
//     graph::logical_tensor_t output_zp_desc
//             = utils::logical_tensor_init(lt_idx++, {1}, graph::data_type::s32);
//     // defining output layer ops
//     graph::op_t typecast_output_f32 {
//             op_idx++, graph::op_kind::TypeCast, "typecast_output_f32"};
//     graph::op_t quant_output {op_idx++,
//             dyn_quant ? graph::op_kind::DynamicQuantize
//                       : graph::op_kind::Quantize,
//             "quantize_output"};
//     if (dyn_quant) {
//         DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(quant_output);
//     } else {
//         DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quant_output);
//     }
//     graph::op_t dequant_output {op_idx++,
//             dyn_quant ? graph::op_kind::DynamicDequantize
//                       : graph::op_kind::Dequantize,
//             "dequantize_output"};
//     if (dyn_quant) {
//         DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequant_output);
//     } else {
//         DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequant_output);
//     }
//     graph::op_t typecast_output_bf16 {
//             op_idx++, graph::op_kind::TypeCast, "typecast_output_bf16"};
//     // defining connection between output ops
//     quant_output.add_output(quant_output_desc);
//     dequant_output.add_input(quant_output_desc);
//     dequant_output.add_output(dequant_output_desc);
//     if (mixed_dtype) {
//         typecast_output_f32.add_input(input_desc);
//         typecast_output_f32.add_output(casted_output_desc);
//         quant_output.add_input(casted_output_desc);
//         typecast_output_bf16.add_input(dequant_output_desc);
//         typecast_output_bf16.add_output(casted_dequant_output_desc);
//     } else {
//         quant_output.add_input(input_desc);
//     }
//     if (dyn_quant) {
//         quant_output.add_input(output_scale_desc);
//         quant_output.add_input(output_zp_desc);
//         dequant_output.add_input(output_scale_desc);
//         dequant_output.add_input(output_zp_desc);
//     }
//     // adding ops
//     agraph->add_op(&quant_output);
//     agraph->add_op(&dequant_output);
//     if (mixed_dtype) {
//         agraph->add_op(&typecast_output_f32);
//         agraph->add_op(&typecast_output_bf16);
//     }
// }

// inline void add_int8_mlp_subgraph(graph_t *agraph, graph::dim_t batch_size = 1,
//         graph::dim_t layer = 1,
//         std::vector<graph::dim_t> hidden_size = {13, 512},
//         std::vector<op_kind_t> act_type = {op_kind::ReLU},
//         bool mixed_dtype = false, bool dyn_quant = false) {
//     add_int8_mlp_subgraph(agraph, std::vector<graph::dim_t> {batch_size}, layer,
//             hidden_size, act_type, mixed_dtype, dyn_quant);
// }

/// Set any layout according to the connection relationship of partitions
///
/// @param partitions a list of partitions
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout
///     type
void set_any_layout(const std::vector<dnnl::graph::partition> &partitions,
        std::unordered_set<size_t> &id_to_set_any_layout) {
    // mapping from output tensor id to the all supported flags of
    // supported partitions, we may only need outputs' supported flags
    std::unordered_map<size_t, std::vector<bool>> output_to_flag_map;
    for (const auto &p : partitions) {
        for (const auto &out : p.get_output_ports()) {
            size_t id = out.get_id();
            if (p.is_supported()
                    && output_to_flag_map.find(id)
                            == output_to_flag_map.end()) {
                output_to_flag_map[id] = {};
            }
        }

        for (const auto &in : p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            if (iter != output_to_flag_map.end()) {
                // collect all of supported flags of this tensor's uses
                // Considering we have such a graph:
                //
                //   partition_A  partition_B
                //        \           |
                //      tensor1    tensor2
                //           \     /     |
                //         partition_C  unsupported partition
                //              |
                //           tensor3
                //              |
                //          framework op
                //
                // so the mapping of partition_A's output will be { true }
                // the mapping of partition_B's output will be { true, false }
                // The mapping of partition_C's output will be { false }
                // Only when all supported flags are true, users can set any
                // layout.
                iter->second.push_back(p.is_supported());
            }
        }
    }

    for (const auto &p : partitions) {
        // no need to set `any` layout if this partition is not supported
        if (!p.is_supported()) continue;
        for (const auto &in : p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            // if this input tensor is not an output of another supported
            // partition, just skip
            if (iter == output_to_flag_map.end()) continue;
            std::vector<bool> flag_vec = iter->second;
            // check if all of uses of this tensor are supported partitions,
            // if not, no need to set ANY layout.
            bool need_set_any = std::all_of(flag_vec.begin(), flag_vec.end(),
                    [](const bool a) { return a; });
            if (!need_set_any) continue;

            /// record the id of logical tensor that will be set to ANY layout
            id_to_set_any_layout.insert(id);
        }
    }
}

struct cpu_deletor {
    cpu_deletor() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

#ifdef DNNL_WITH_SYCL
struct sycl_deletor {
    sycl_deletor() = delete;
    ::sycl::context ctx_;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};

void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    return malloc_shared(size, *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    // Device is not used in this example, but it may be useful for some users
    // application.
    UNUSED(device);
    // immediate synchronization here is for test purpose. For performance,
    // users may need to store the ptr and event and handle them separately
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    free(ptr, *static_cast<const ::sycl::context *>(context));
}
#endif

void allocate_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer,
        const dnnl::engine &eng) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto mem_size = lt.get_mem_size();

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(malloc(mem_size), cpu_deletor {});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);
    }
}

void allocate_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer,
        std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
        const dnnl::engine &eng, bool is_input) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto lt_id = lt.get_id();
        const auto mem_size = lt.get_mem_size();

        // check if the input is an output of another partition
        if (is_input) {
            auto pos = global_outputs_ts_map.find(lt_id);
            if (pos != global_outputs_ts_map.end()) {
                tensors.push_back(pos->second);
                continue;
            }
        }

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(malloc(mem_size), cpu_deletor {});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);

        // record the connection relationship between partitions
        if (!is_input) global_outputs_ts_map[lt_id] = tensors.back();
    }
}

#ifdef DNNL_WITH_SYCL
void allocate_sycl_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer, sycl::queue &q,
        const dnnl::engine &eng) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto mem_size = lt.get_mem_size();

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(::sycl::malloc_shared(mem_size, q.get_device(),
                                         q.get_context()),
                sycl_deletor {q.get_context()});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);
    }
}

void allocate_sycl_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer,
        std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
        sycl::queue &q, const dnnl::engine &eng, bool is_input) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto lt_id = lt.get_id();
        const auto mem_size = lt.get_mem_size();

        // check if the input is an output of another partition
        if (is_input) {
            auto pos = global_outputs_ts_map.find(lt_id);
            if (pos != global_outputs_ts_map.end()) {
                tensors.push_back(pos->second);
                continue;
            }
        }

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(::sycl::malloc_shared(mem_size, q.get_device(),
                                         q.get_context()),
                sycl_deletor {q.get_context()});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);

        // record the connection relationship between partitions
        if (!is_input) global_outputs_ts_map[lt_id] = tensors.back();
    }
}
#endif

#endif
