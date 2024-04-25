#ifndef GC_TEST_HPP
#define GC_TEST_HPP

#include "compiler/config/context.hpp"
#include "compiler/ir/graph/driver.hpp"
#include "compiler/ir/graph/dynamic_utils.hpp"
#include "compiler/ir/graph/pass/pass.hpp"
#include "compiler/ir/transform/tensor_inplace_info.hpp"
#include "compiler/jit/compiler_driver.hpp"
#include "compiler_partition_impl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {
namespace graph_compiler{

// returns an empty graph for now
void json_to_sc_graph(const char* json, gc::sc_graph_t &out_graph, std::vector<gc::sc_op_ptr> &out_args) {}

/*=============*/

struct GraphCompiler {
    std::shared_ptr<gc::runtime::engine_t> eng;
};

std::shared_ptr<dnnl::impl::graph::gc::context_t> build_sc_context(GraphCompiler* gc_ptr);
gc::runtime::stream_t* build_sc_stream(GraphCompiler* gc_ptr);

struct Executable;
struct Hint;
struct Hints{
    Hint* h;
};
enum Status{ OK };
struct EngineContext;

struct AllocatorsVTable{
    ///////// allocators:
    // use case:
    // void* my_allocator(EngineContext* ctx, size_t) {
    //     auto onednn_allocator = static_cast<compiler_graph_engine_t*>(ctx->parent_engine)->engine_->get_allocator();
    //     return onednn_allocator.allocate(...);
    // }
    using alloc_t = void *(*)(EngineContext*, size_t);
    using dealloc_t = void (*)(EngineContext*, void *);
    alloc_t persistent_alloc;
    dealloc_t persistent_dealloc;
    alloc_t temp_alloc;
    dealloc_t temp_dealloc;
    // std::shared_ptr<const_cache_proxy> (*alloc_and_register_tensor_cache)(
    //         engine_t *, size_t);
    size_t (*get_tensor_cache_cap)(EngineContext*);
};

struct EngineContext {
    unsigned thread_num;
    void* parent_engine; // meta info for funcs from vtable (may be null)
    AllocatorsVTable* avtable;
};
struct ExecutionArgs {
    void* inputs;
    void* outputs;
    size_t inputs_size, output_size;
};

Status create(GraphCompiler* gc_ptr, EngineContext* ctx) {
    UNUSED(ctx); // ignore for now
    gc_ptr = new GraphCompiler{};
    return Status::OK;
}

Status destroy(GraphCompiler* gc_ptr) { return Status::OK; }
Status compile(GraphCompiler* gc_ptr, const char* graph_json, Executable* exe) {
    static gc::context_ptr ctx = build_sc_context(gc_ptr);

    std::vector<gc::sc_op_ptr> args;
    gc::sc_graph_t backend_graph_obj;
    json_to_sc_graph(graph_json, backend_graph_obj, args);

    std::shared_ptr<gc::jit_function_t> fptr
                = gc::compiler_driver(ctx, backend_graph_obj, args);
    exe = new Executable{fptr};
    return Status::OK;
}
Status compile(GraphCompiler* gc, const char* graph_json, Executable*, Hints*);
Status execute(GraphCompiler* gc_ptr, Executable* exe, ExecutionArgs args) {
    // TODO: build the stream based on gc's context
    auto str = build_sc_stream(gc_ptr);
    std::vector<gc::generic_val> call_args;
    call_args.reserve(args.inputs_size + args.output_size);

    // ??
    for (size_t i=0; i<args.inputs_size; i++){
        call_args.emplace_back(args.inputs + i);
    }
    for (size_t i=0; i<args.output_size; i++){
        call_args.emplace_back(args.outputs + i);
    }

    exe->fptr->call_generic(str, call_args.data());
    return Status::OK;
}

/////////

struct Executable {
    std::shared_ptr<gc::jit_function_t> fptr;
};

std::shared_ptr<gc::context_t> build_sc_context(GraphCompiler* gc_ptr) {
    // TODO: build context based on gc_ptr's info
    return std::make_shared<gc::context_t>(*gc::get_default_context());
}

gc::runtime::stream_t* build_sc_stream(GraphCompiler* gc_ptr) {
    // TODO: build stream based on gc_ptr's info
    return gc::runtime::get_default_stream();
}

struct GraphCompiler {};

}
}
}
}
}

#endif
