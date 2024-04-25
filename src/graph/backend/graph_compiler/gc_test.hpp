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
    // Expose const_cache_proxy structure???
    // std::shared_ptr<const_cache_proxy> (*alloc_and_register_tensor_cache)(
    //         engine_t *, size_t);
    size_t (*get_tensor_cache_cap)(EngineContext*);
};

struct EngineContext {
    unsigned thread_num;
    void* parent_engine; // meta info for funcs from vtable (may be null)
    AllocatorsVTable* avtable;
};

struct GraphCompiler { // this is an impl, it won't be exposed to a user
    EngineContext* ctx;
    std::shared_ptr<gc::runtime::engine_t> eng;
    // 'eng' expects a vtable with the following interface:
    // void* allocator(gc::runtime::engine_t*, size_t);
    // how to go from engine_t* -> EngineContext??
    GraphCompiler(EngineContext* ctx) {
        std::function<void* (gc::runtime::engine_t*, size_t)> persistent_alloc = 
            [ctx](gc::runtime::engine_t* eng, size_t sz) {
                UNUSED(eng);
                return ctx->avtable->persistent_alloc(ctx, sz);
            };
        // define other allocs...

        gc::runtime::engine_vtable_t vtable {
            /*persistent_alloc=*/persistent_alloc
            // other allocators ...
        };
        ctx = ctx;
        eng = gc::runtime::engine_vtable_t{vtable};
    }

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

struct ExecutionArgs {
    void* inputs;
    void* outputs;
    size_t inputs_size, output_size;
};

Status create(GraphCompiler* gc_ptr, EngineContext* ctx) {
    gc_ptr = new GraphCompiler{ctx};
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
    auto str = build_sc_stream(gc_ptr);
    std::vector<gc::generic_val> call_args;
    call_args.reserve(args.inputs_size + args.output_size);

    // what if some of them were dynamic inputs??
    // how can we get the shapes??
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
    auto ctx = std::make_shared<gc::context_t>(*gc::get_default_context());
    ctx->engine_ = gc_ptr->eng.get();
}

gc::runtime::stream_t* build_sc_stream(GraphCompiler* gc_ptr) {
    // TODO: build stream based on gc_ptr's info
    return new gc::runtime::stream_t {
        /*stream_vtable=*/{
            /*parallel_call_cpu_t=*/sc_parallel_call_cpu_with_env_impl,
            /*dnnl_stream=*/nullptr //(unused in case of TBB and OMP)
        },
        /*sc_engine=*/gc_ptr->eng.get()
    };
}

}
}
}
}
}

#endif
