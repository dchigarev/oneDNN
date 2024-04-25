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

/// @example cpu_getting_started.cpp
/// @copybrief graph_cpu_getting_started_cpp
/// > Annotated version: @ref graph_cpu_getting_started_cpp

/// @page graph_cpu_getting_started_cpp Getting started on CPU with Graph API
/// This is an example to demonstrate how to build a simple graph and run it on
/// CPU.
///
/// > Example code: @ref cpu_getting_started.cpp
///
/// Some key take-aways included in this example:
///
/// * how to build a graph and get partitions from it
/// * how to create an engine, allocator and stream
/// * how to compile a partition
/// * how to execute a compiled partition
///
/// Some assumptions in this example:
///
/// * Only workflow is demonstrated without checking correctness
/// * Unsupported partitions should be handled by users themselves
///

/// @page graph_cpu_getting_started_cpp
/// @section graph_cpu_getting_started_cpp_headers Public headers
///
/// To start using oneDNN Graph, we must include the @ref dnnl_graph.hpp header
/// file in the application. All the C++ APIs reside in namespace `dnnl::graph`.
///
/// @page graph_cpu_getting_started_cpp
/// @snippet cpu_getting_started.cpp Headers and namespace
//[Headers and namespace]
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <assert.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "example_utils.hpp"
#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
//[Headers and namespace]

void cpu_getting_started_tutorial() {
    //[Create second relu]


    /// Finally, those created ops will be added into the graph. The graph
    /// inside will maintain a list to store all these ops. To create a graph,
    /// #dnnl::engine::kind is needed because the returned partitions
    /// maybe vary on different devices. For this example, we use CPU engine.
    ///
    /// @note The order of adding op doesn't matter. The connection will
    /// be obtained through logical tensors.
    ///
    /// Create graph and add ops to the graph
    /// @snippet cpu_getting_started.cpp Create graph and add ops
    //[Create graph and add ops]
    graph g(dnnl::engine::kind::cpu);

    std::vector<int64_t> src_shape {4, 1, 4096};

    dims smooth_quant_scales_shape;
    auto dtype = data_type::f32;
    logical_tensor mul_in {0, dtype};
    logical_tensor smooth_quant_scale {1, dtype};
    logical_tensor mul_out {2, dtype};
    logical_tensor quant_out {3, data_type::u8};


    op mul {4, op::kind::Multiply, "mul"};
    mul.add_input(mul_in);
    mul.add_input(smooth_quant_scale);
    mul.add_output(mul_out);

    op quantize {5, op::kind::Quantize, "quantize"};
    quantize.set_attr(op::attr::scales, std::vector<float>({0.12f}));
    quantize.set_attr(op::attr::zps, std::vector<int64_t>({2}));
    quantize.set_attr(op::attr::qtype, std::string("per_tensor"));
    quantize.set_attr(op::attr::axis, (int64_t)0);

    quantize.add_input(mul_out);
    quantize.add_output(quant_out);

    g.add_op(mul);
    g.add_op(quantize);

    //[Create graph and add ops]

    /// After adding all ops into the graph, call
    /// #dnnl::graph::graph::get_partitions() to indicate that the
    /// graph building is over and is ready for partitioning. Adding new
    /// ops into a finalized graph or partitioning a unfinalized graph
    /// will both lead to a failure.
    ///
    /// @snippet cpu_getting_started.cpp Finalize graph
    //[Finalize graph]
    g.finalize();
    //[Finalize graph]

    /// After finished above operations, we can get partitions by calling
    /// #dnnl::graph::graph::get_partitions().
    ///
    /// In this example, the graph will be partitioned into two partitions:
    /// 1. conv0 + conv0_bias_add + relu0
    /// 2. conv1 + conv1_bias_add + relu1
    ///
    /// @snippet cpu_getting_started.cpp Get partition
    //[Get partition]
    auto partitions = g.get_partitions();
    //[Get partition]

    // Check partitioning results to ensure the examples works. Users do
    // not need to follow this step.
    std::cout << "part size: " << partitions.size() << std::endl;

    /// @page graph_cpu_getting_started_cpp
    /// @subsection graph_cpu_getting_started_cpp_compile Compile and Execute Partition
    ///
    /// In the real case, users like framework should provide device information
    /// at this stage. But in this example, we just use a self-defined device to
    /// simulate the real behavior.
    ///
    /// Create a #dnnl::engine. Also, set a user-defined
    /// #dnnl::graph::allocator to this engine.
    ///
    /// @snippet cpu_getting_started.cpp Create engine
    //[Create engine]
    allocator alloc {};
    dnnl::engine eng
            = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
    //[Create engine]

    /// Create a #dnnl::stream on a given engine
    ///
    /// @snippet cpu_getting_started.cpp Create stream
    //[Create stream]
    dnnl::stream strm {eng};
    // return;
    //[Create stream]

    // Mapping from logical tensor id to output tensors
    // used to the connection relationship between partitions (e.g partition 0's
    // output tensor is fed into partition 1)
    std::unordered_map<size_t, tensor> global_outputs_ts_map;

    // Memory buffers bound to the partition input/output tensors
    // that helps manage the lifetime of these tensors
    std::vector<std::shared_ptr<void>> data_buffer;

    // Mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with
    // ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    // This is a helper function which helps decide which logical tensor is
    // needed to be set with `dnnl::graph::logical_tensor::layout_type::any`
    // layout.
    // This function is not a part to Graph API, but similar logic is
    // essential for Graph API integration to achieve best performance.
    // Typically, users need implement the similar logic in their code.
    std::unordered_set<size_t> ids_with_any_layout;
    set_any_layout(partitions, ids_with_any_layout);

    // Mapping from logical tensor id to the concrete shapes.
    // In practical usage, concrete shapes and layouts are not given
    // until compilation stage, hence need this mapping to mock the step.

    dims ml1_dims {10};
    dims ml2_dims {10};

    std::unordered_map<size_t, dims> concrete_shapes {{0, ml1_dims}, {1, ml2_dims}};

    // Compile and execute the partitions, including the following steps:
    //
    // 1. Update the input/output logical tensors with concrete shape and layout
    // 2. Compile the partition
    // 3. Update the output logical tensors with queried ones after compilation
    // 4. Allocate memory and bind the data buffer for the partition
    // 5. Execute the partition
    //
    // Although they are not part of the APIs, these steps are essential for
    // the integration of Graph API., hence users need to implement similar
    // logic.
    for (const auto &partition : partitions) {
        if (!partition.is_supported()) {
            std::cout
                    << "cpu_get_started: Got unsupported partition, users need "
                       "handle the operators by themselves."
                    << std::endl;
            continue;
        }

        std::vector<logical_tensor> inputs = partition.get_input_ports();
        std::vector<logical_tensor> outputs = partition.get_output_ports();

        // Update input logical tensors with concrete shape and layout
        for (auto &input : inputs) {
            const auto id = input.get_id();
            // If the tensor is an output of another partition,
            // use the cached logical tensor
            if (id_to_queried_logical_tensors.find(id)
                    != id_to_queried_logical_tensors.end())
                input = id_to_queried_logical_tensors[id];
            else
                // Create logical tensor with strided layout
                input = logical_tensor {id, input.get_data_type(),
                        concrete_shapes[id], layout_type::strided};
        }

        // Update output logical tensors with concrete shape and layout
        for (auto &output : outputs) {
            const auto id = output.get_id();
            output = logical_tensor {id, output.get_data_type(),
                    DNNL_GRAPH_UNKNOWN_NDIMS, // set output dims to unknown
                    ids_with_any_layout.count(id) ? layout_type::any
                                                  : layout_type::strided};
        }

        /// Compile the partition to generate compiled partition with the
        /// input and output logical tensors.
        ///
        /// @snippet cpu_getting_started.cpp Compile partition
        //[Compile partition]
        compiled_partition cp = partition.compile(inputs, outputs, eng);
        //[Compile partition]

        // Update output logical tensors with queried one
        for (auto &output : outputs) {
            const auto id = output.get_id();
            output = cp.query_logical_tensor(id);
            id_to_queried_logical_tensors[id] = output;
        }

        // Allocate memory for the partition, and bind the data buffers with
        // input and output logical tensors
        std::vector<tensor> inputs_ts, outputs_ts;
        allocate_graph_mem(inputs_ts, inputs, data_buffer,
                global_outputs_ts_map, eng, /*is partition input=*/true);
        allocate_graph_mem(outputs_ts, outputs, data_buffer,
                global_outputs_ts_map, eng, /*is partition input=*/false);

        /// Execute the compiled partition on the specified stream.
        ///
        /// @snippet cpu_getting_started.cpp Execute compiled partition
        //[Execute compiled partition]
        cp.execute(strm, inputs_ts, outputs_ts);
        //[Execute compiled partition]
    }

    // Wait for all compiled partition's execution finished
    strm.wait();
}

int main(int argc, char **argv) {
    return handle_example_errors(
            {engine::kind::cpu}, cpu_getting_started_tutorial);
}
