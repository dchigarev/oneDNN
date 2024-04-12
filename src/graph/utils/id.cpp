/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "graph/utils/id.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

std::atomic<id_t::value_type> id_t::counter DNNL_API {100000};

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
