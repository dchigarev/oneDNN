/*******************************************************************************
 * Copyright 2021-2024 Intel Corporation
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
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "elyzor_partition_impl.hpp"

#include "common/rw_mutex.hpp"
#include "common/verbose.hpp"
#include "graph/interface/graph.hpp"
#include "graph/utils/debug.hpp"
#include "graph/utils/utils.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

graph::status_t elyzor_partition_impl_t::infer_shape(
        std::vector<const graph::logical_tensor_t *> &inputs,
        std::vector<graph::logical_tensor_t *> &outputs) const {
    std::lock_guard<std::mutex> lck(mtx_);
    // construct a temp graph
    copied_ops_ = graph::graph_t::deep_copy(ops_);
    graph::graph_t temp_graph(copied_ops_);
    auto output_ops = temp_graph.get_output_ops();
    auto ret = topo_order_visit(output_ops, [&](op_t *cur_op) {
        const graph::op_schema_t *cur_op_schema
                = graph::op_schema_registry_t::get_op_schema(
                        cur_op->get_kind());
        assertm(cur_op_schema, "Can't infer shape for cur op: no schema");
        auto get_logical_tensor = [&](const std::shared_ptr<value_t> &val)
                -> graph::logical_tensor_t {
            logical_tensor_t lt = val->get_logical_tensor();
            auto in_pos = std::find_if(inputs.begin(), inputs.end(),
                    [&](const graph::logical_tensor_t *alt) -> bool {
                        return alt->id == lt.id;
                    });
            if (in_pos != inputs.end()) { return **in_pos; }
            return lt;
        };
        graph::op_t temp_node = graph::op_t(cur_op->get_kind());
        temp_node.merge_attributes(cur_op->get_attributes());
        std::vector<graph::logical_tensor_t> ordered_inputs_holder
                = utils::func_map(
                        cur_op->get_input_values(), get_logical_tensor);
        std::vector<graph::logical_tensor_t> ordered_outputs_holder
                = utils::func_map(
                        cur_op->get_output_values(), get_logical_tensor);
        std::vector<graph::logical_tensor_t *> ordered_inputs;
        ordered_inputs.reserve(ordered_inputs_holder.size());
        for (auto &tsr : ordered_inputs_holder) {
            ordered_inputs.emplace_back(&tsr);
        }
        std::vector<graph::logical_tensor_t *> ordered_outputs;
        ordered_outputs.reserve(ordered_outputs_holder.size());
        for (auto &tsr : ordered_outputs_holder) {
            ordered_outputs.emplace_back(&tsr);
        }
        graph::status_t ret = cur_op_schema->shape_infer(
                &temp_node, ordered_inputs, ordered_outputs);
        if (ret != graph::status::success) return ret;
        for (size_t i = 0; i < cur_op->get_output_values().size(); ++i) {
            auto output_lt = *ordered_outputs[i];
            auto cur_val = cur_op->get_output_values()[i];
            cur_val->set_logical_tensor(output_lt);
            // if layout is any; let's respect it
            // if layout is strided; shape_infer will fill the stride
            // if layout is undef; convert it to strided and fill the stride
            if (output_lt.layout_type == graph::layout_type::undef) {
                // force set strided dense layout
                graph::dims shape(
                        output_lt.dims, output_lt.dims + output_lt.ndims);
                graph::dims strides = utils::get_dense_strides(shape);
                cur_val->set_strides(strides);
            }
            // only write back inferred info to outputs
            // shall not modify the layout type
            auto out_pos = std::find_if(outputs.begin(), outputs.end(),
                    [&](graph::logical_tensor_t *alt) -> bool {
                        return alt->id == ordered_outputs[i]->id;
                    });
            if (out_pos != outputs.end()) {
                auto cur_lt = cur_val->get_logical_tensor();
                // ensure layout type not modified
                if ((**out_pos).layout_type == graph::layout_type::any)
                    cur_lt.layout_type = (**out_pos).layout_type;
                **out_pos = cur_lt;
            }
        }
        return graph::status::success;
    });
    return ret;
}

graph::status_t elyzor_partition_impl_t::compile(
        graph::compiled_partition_t *compiled_partition,
        const std::vector<graph::logical_tensor_t> &inputs,
        const std::vector<graph::logical_tensor_t> &outputs,
        const graph::engine_t *aengine) const {
    try {
        std::cout << "Trying to compile using Elyzor..." << std::endl;
        graph::status_t res = status::success;
        // here we call infer_shape since logical tensor info
        // may be incomplete for the graph corresponding to the
        // partition
        std::vector<const graph::logical_tensor_t *> input_ref;
        std::vector<graph::logical_tensor_t *> output_ref;
        input_ref.reserve(inputs.size());
        output_ref.reserve(outputs.size());
        for (auto &t : inputs) {
            input_ref.push_back(const_cast<graph::logical_tensor_t *>(&t));
        }
        for (auto &t : outputs) {
            output_ref.push_back(const_cast<graph::logical_tensor_t *>(&t));
        }
        res = this->infer_shape(input_ref, output_ref);
        if (res != status::success) { return res; }

        COMPILE_ASSERT(aengine->kind() == graph::engine_kind_t::dnnl_cpu,
                "Graph compiler backend only supports cpu engine");

        // get executor somehow

        auto pimpl = std::make_shared<elyzor_compiled_partition_impl_t>(
                *aengine, inputs, outputs,
                std::vector<graph::inplace_pair_t> {}, /*executor=*/nullptr);
        compiled_partition->init(pimpl);
        return res;
    } catch (const std::exception &e) {
        VERROR(graph, elyzor, "%s", e.what());
        return graph::status::unimplemented;
    }
}

std::shared_ptr<graph::partition_impl_t>
elyzor_partition_impl_t::clone() const {
    auto ret = std::make_shared<elyzor_partition_impl_t>(
            get_engine_kind(), get_fpmath_mode(), get_kind(), get_name());
    ret->ops_ = graph::graph_t::deep_copy(ops_);
    ret->inputs_ = inputs_;
    ret->outputs_ = outputs_;
    ret->id_ = id_;
    ret->is_init_ = is_init_;
    return ret;
}

bool elyzor_partition_impl_t::is_initialized() const {
    return is_init_;
}

elyzor_compiled_partition_impl_t::elyzor_compiled_partition_impl_t(
        const graph::engine_t &engine,
        const std::vector<graph::logical_tensor_t> &inputs,
        const std::vector<graph::logical_tensor_t> &outputs,
        const std::vector<graph::inplace_pair_t> &inplace_pairs,
        const void *executor)
    : graph::compiled_partition_impl_t(engine, inputs, outputs, inplace_pairs)
    , executor_(executor) {}

elyzor_compiled_partition_impl_t::~elyzor_compiled_partition_impl_t() {}

graph::status_t elyzor_compiled_partition_impl_t::execute(
        const graph::stream_t *astream,
        const std::vector<graph::tensor_t> &inputs,
        const std::vector<graph::tensor_t> &outputs) {
    std::cout << "Trying to execute using Elyzor..." << std::endl;
    // prepare args and execute
    // run(executor_, args)
    return status::success;
}
} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl
