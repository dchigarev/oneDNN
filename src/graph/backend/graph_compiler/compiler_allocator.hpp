/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_COMPILER_ALLOCATOR_HPP
#define BACKEND_GRAPH_COMPILER_COMPILER_ALLOCATOR_HPP

#include <memory>
#include <unordered_set>

#include "common/engine.hpp"
#include "graph/interface/allocator.hpp"
#include "graph/interface/constant_tensor_cache.hpp"

#include "runtime/context.hpp"
#include "runtime/parallel.hpp"
#include "gc_test.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

struct compiler_graph_engine_t;

struct engine_ref_data {
    std::unordered_map<const graph::engine_t *,
            std::weak_ptr<graph::compiler_impl::compiler_graph_engine_t>>
            engine_map_;
    std::mutex global_mutex_;
};

struct compiler_graph_engine_t {
    graph::engine_t *engine_;
    graph_compiler::EngineContext *ctx_;
    constant_tensor_cache_t *cache_;
    compiler_graph_engine_t(graph_compiler::AllocatorsVTable *vtable,
            graph::engine_t *engine);
    ~compiler_graph_engine_t();
};

struct compiler_graph_stream_t : public gc::runtime::stream_t {
    compiler_graph_stream_t(
            compiler_graph_engine_t *eng, const dnnl_stream *stream);
};

extern graph_compiler::AllocatorsVTable graph_engine_vtable;
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
