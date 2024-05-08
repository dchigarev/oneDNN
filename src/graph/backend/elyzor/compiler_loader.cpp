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
#include "compiler_loader.hpp"
#include "elyzor_backend.hpp"

#ifdef __APPLE__
#include <dlfcn.h>
#define DNNL_GC_LIB_NAME "libgraph_compiler.dylib"
#elif !defined(_WIN32)
#include <dlfcn.h>
#define DNNL_GC_LIB_NAME "libgraph_compiler.so"
#endif

#ifdef _WIN32
#include <windows.h>
#define DNNL_GC_LIB_NAME "graph_compiler.dll"
#define OPEN_GC_LIB() LoadLibrary(DNNL_GC_LIB_NAME)
#define LOAD_FUNC(handle, funcname) GetProcAddress((HMODULE)handle, funcname)
#define FREE_LIB(handle) FreeLibrary((HMODULE)handle)
#define GET_LAST_LIB_ERROR() \
    ("an error occured when working with " #DNNL_GC_LIB_NAME)
#else
#define OPEN_GC_LIB() dlopen(DNNL_GC_LIB_NAME, RTLD_LAZY | RTLD_DEEPBIND)
#define LOAD_FUNC(handle, funcname) dlsym(handle, funcname)
#define FREE_LIB(handle) dlclose(handle)
#define GET_LAST_LIB_ERROR() dlerror()
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

template <typename func_ptr_type>
func_ptr_type load_func(void *handle, const char *func_name) {
    if (!handle) {
        throw std::runtime_error("Can't load symbols from an invalid handle.");
    }
    func_ptr_type func
            = reinterpret_cast<func_ptr_type>(LOAD_FUNC(handle, func_name));
    if (!func) {
        std::stringstream ss;
        ss << "Failed to load \'" << func_name << "\' function.";
        throw std::runtime_error(ss.str());
    }
    return func;
}

graph_compiler_loader::graph_compiler_loader() {
    handle_ = OPEN_GC_LIB();
    if (!handle_) {
        std::stringstream ss;
        ss << "Failed to load library: " << GET_LAST_LIB_ERROR();
        throw std::runtime_error(ss.str());
    }

    try {
        vtable_.dnnl_graph_compiler_create
                = load_func<dnnl_graph_compiler_create_t>(
                        handle_, "dnnl_graph_compiler_create");
        vtable_.dnnl_graph_compiler_destroy
                = load_func<dnnl_graph_compiler_destroy_t>(
                        handle_, "dnnl_graph_compiler_destroy");
        vtable_.dnnl_graph_compiler_compile
                = load_func<dnnl_graph_compiler_compile_t>(
                        handle_, "dnnl_graph_compiler_compile");
        vtable_.dnnl_graph_compiler_destroy_executable
                = load_func<dnnl_graph_compiler_destroy_executable_t>(
                        handle_, "dnnl_graph_compiler_destroy_executable");
        vtable_.dnnl_graph_compiler_execute
                = load_func<dnnl_graph_compiler_execute_t>(
                        handle_, "dnnl_graph_compiler_execute");
    } catch (...) {
        FREE_LIB(handle_);
        throw;
    }
}

graph_compiler_loader::~graph_compiler_loader() {
    if (handle_) FREE_LIB(handle_);
}

} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl

#define LOAD_AND_CALL(fn_name, ...) \
    dnnl::impl::graph::elyzor::graph_compiler_loader::get_vtable().fn_name( \
            __VA_ARGS__);

DNNL_API dnnl_status_t dnnl_graph_compiler_create(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc) {
    return LOAD_AND_CALL(dnnl_graph_compiler_create, ctx, gc);
}

DNNL_API void dnnl_graph_compiler_destroy(
        const struct dnnl_graph_compiler *gc) {
    return LOAD_AND_CALL(dnnl_graph_compiler_destroy, gc);
}

DNNL_API dnnl_status_t dnnl_graph_compiler_compile(
        const struct dnnl_graph_compiler *gc, const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe) {
    return LOAD_AND_CALL(dnnl_graph_compiler_compile, gc, graph_json, exe);
}

DNNL_API void dnnl_graph_compiler_destroy_executable(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe) {
    LOAD_AND_CALL(dnnl_graph_compiler_destroy_executable, gc, exe);
}

DNNL_API dnnl_status_t dnnl_graph_compiler_execute(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs) {
    return LOAD_AND_CALL(dnnl_graph_compiler_execute, gc, exe, inputs, outputs);
}
