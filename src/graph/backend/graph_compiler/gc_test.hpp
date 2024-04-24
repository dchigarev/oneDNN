

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

struct GraphCompiler;
struct Executable;
struct Hint;
struct Hints{
    Hint* h;
};
enum Status{ OK };
struct EngineContext {
    unsigned thread_num;
    // allocators
};
struct ExecutionArgs {
    void* inputs;
    void* outputs;
    size_t inputs_size, output_size;
};

Status create(GraphCompiler* gc, EngineContext* ctx) {
    UNUSED(ctx); // ignore for now
    gc = new GraphCompiler{};
    return Status::OK;
}

Status destroy(GraphCompiler* gc) { return Status::OK; }
Status compile(GraphCompiler* gc, const char* graph_json, Executable* exe) {
    // TODO: build the context based on gc's context
    static gc::context_ptr ctx
        = std::make_shared<gc::context_t>(*gc::get_default_context());
    std::vector<gc::sc_op_ptr> args;
    gc::sc_graph_t backend_graph_obj;
    json_to_sc_graph(graph_json, backend_graph_obj, args);

    std::shared_ptr<gc::jit_function_t> fptr
                = gc::compiler_driver(ctx, backend_graph_obj, args);
    exe = new Executable{fptr.get()};
    return Status::OK;
}
Status compile(GraphCompiler* gc, const char* graph_json, Executable*, Hints*);
Status execute(GraphCompiler* gc, Executable* exe, ExecutionArgs args) {
    // TODO: build the stream based on gc's context
    auto str = gc::runtime::get_default_stream();
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
    gc::jit_function_t* fptr;
};

struct GraphCompiler {};

}
}
}
}
}
