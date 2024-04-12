/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_INTERFACE_BACKEND_HPP
#define GRAPH_INTERFACE_BACKEND_HPP

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/interface/partition.hpp"
#include "graph/interface/tensor.hpp"

#include "oneapi/dnnl/dnnl_graph_backend.hpp"

#define BACKEND_ID_LENGTH 4
#define MAX_BACKEND_NUMS (1 << BACKEND_ID_LENGTH)

namespace dnnl {
namespace impl {
namespace graph {

// forward declaration
void register_dnnl_backend();
void register_fake_backend();
#ifdef DNNL_ENABLE_COMPILER_BACKEND
// register graph compiler backend
void register_compiler_backend();
#endif

std::vector<const backend_t *> DNNL_API &dnnl_get_registered_backends();
std::string DNNL_API dnnl_print_backend_name(const backend_t* bkd);
void DNNL_API dnnl_register_backend(const backend_t *abackend);

class DNNL_API backend_registry_t {
public:
    static backend_registry_t &get_singleton() {
        static backend_registry_t inst;
        return inst;
    }

    backend_t *register_backend(const backend_t *abackend) {
        auto has_colliding_name = [&](const backend_t *backend) {
            return backend->get_name().compare(abackend->get_name()) == 0;
        };
        auto backend_already_registered = [&]() {
            return std::find_if(sorted_backends_.begin(),
                           sorted_backends_.end(), has_colliding_name)
                    != sorted_backends_.end();
        };

        auto compare_priority = [](const backend_t *l, const backend_t *r) {
            return l->get_priority() > r->get_priority();
        };

        if (backend_already_registered()) {
            throw std::runtime_error(
                    "backend name not unique: " + abackend->get_name());
        }

        std::lock_guard<std::mutex> lock(m_);

        backends_[abackend->get_id()] = abackend;
        sorted_backends_.emplace_back(abackend);
        std::sort(sorted_backends_.begin(), sorted_backends_.end(),
                compare_priority);
        return const_cast<backend_t *>(abackend);
    }

    // This interface will firstly register all available backends and then
    // return sorted backends. The highest priority backend will be at the front
    // of vector
    std::vector<const backend_t *> &get_registered_backends() {
        invoke_backend_registration();
        std::lock_guard<std::mutex> lock(m_);
        return sorted_backends_;
    }

    // This interface will also try to register all available backends.
    // In order to use get_mem_size() API, we need to dispatch to specific
    // backend according to the backend specific layout id.
    // In this function, we will first decode the layout id to a backend id
    // and a native layout id. Then we will use the backend id to get the
    // backend from the backend registry
    const backend_t *get_registered_backend(size_t layout_id) {
        invoke_backend_registration();
        size_t backend_id = extract_backend_id(layout_id);
        std::lock_guard<std::mutex> lock(m_);
        return backends_[backend_id];
    }

    static std::pair<size_t, size_t> decode_layout_id(size_t layout_id);

    static size_t encode_layout_id(size_t layout_idx, size_t backend_id);

    static size_t extract_layout_id(size_t layout_id);

    static size_t extract_backend_id(size_t layout_id);

private:
    backend_registry_t() = default;
    backend_registry_t(const backend_registry_t &) = delete;
    backend_registry_t(backend_registry_t &&) = delete;
    backend_registry_t &operator=(const backend_registry_t &) = delete;
    backend_registry_t &operator=(backend_registry_t &&) = delete;

    inline void invoke_backend_registration() {
        std::call_once(register_flag_, []() {
            register_dnnl_backend();
            register_fake_backend();
#ifdef DNNL_ENABLE_COMPILER_BACKEND
            register_compiler_backend();
#endif
        });
    }

    std::mutex m_;

    std::once_flag register_flag_;

    // sorted backends by priority
    std::vector<const backend_t *> sorted_backends_;

    // the map from backend id to backend shared pointer
    std::unordered_map<size_t, const backend_t *> backends_;
};

} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
