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
#include "backend/graph_compiler/compiler_backend.hpp"
#include "interface/allocator.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>
#include <runtime/context.hpp>

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
struct gc_env_initializer {
    gc_env_initializer() {
        dnnl::impl::graph::gc::runtime::get_default_stream = []() {
            static auto the_stream = []() {
                dnnl::impl::graph::gc::runtime::stream_t ret
                        = dnnl::impl::graph::gc::runtime::default_stream;
                ret.vtable_.stream = ::get_stream();
                return ret;
            }();
            return &the_stream;
        };
    }
};
static gc_env_initializer gc_test_init;
#endif

namespace impl = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
namespace compiler_utils = dnnl::impl::graph::tests::unit::compiler::utils;

using ltsr_vec = std::vector<impl::logical_tensor_t>;
static void set_mlp_dynamic_parti_ltsrs(int64_t real_batch_size,
        ltsr_vec &parti_inputs, ltsr_vec &parti_outputs) {
    parti_inputs[0].dims[0] = real_batch_size;
    parti_outputs[0].dims[0] = real_batch_size;
}

static void compile_execution_pipeline(impl::graph_t &agraph,
        int expected_part_size,
        std::function<void(ltsr_vec &, ltsr_vec &)> dynamic_callback
        = nullptr, bool measure_time=false, int num_runs=10, bool random_data=false) {
    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), static_cast<size_t>(expected_part_size));
    if (dynamic_callback) { ASSERT_EQ(expected_part_size, 1); }
    // TODO(yifei): generalize the logic here
    // sort partitions to run forward first according to num ops
    std::sort(partitions.begin(), partitions.end(),
            [](std::shared_ptr<impl::partition_impl_t> a,
                    std::shared_ptr<impl::partition_impl_t> b) {
                return a->get_ops().size() < b->get_ops().size();
            });

    std::unordered_map<size_t, impl::logical_tensor_t> lt_info_map;

    for (size_t i = 0; i < partitions.size(); ++i) {
        impl::partition_t p;
        p.init(partitions[i]);
        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();

        // replace partition inputs info if needed
        for (size_t i = 0; i < partition_inputs.size(); ++i) {
            if (lt_info_map.find(partition_inputs[i].id) != lt_info_map.end()) {
                partition_inputs[i] = lt_info_map[partition_inputs[i].id];
            }
        }

        std::vector<const impl::logical_tensor_t *> inputs;
        std::vector<const impl::logical_tensor_t *> outputs;
        for (auto &lt : partition_inputs) {
            inputs.push_back(&lt);
        }
        for (auto &lt : partition_outputs) {
            outputs.push_back(&lt);
        }
        impl::compiled_partition_t cp(p);
        impl::engine_t &eng = *get_engine();
        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

        num_runs = measure_time ? num_runs : 1;
        std::vector<int64_t> times;
        // srand(time(NULL));
        for (size_t rn = 0; rn < num_runs; rn++) {
            std::vector<test_tensor> execution_inputs;
            std::vector<test_tensor> execution_outputs;
            partition_outputs.clear();
            for (auto &lt : outputs) {
                impl::logical_tensor_t compiled_output;
                cp.query_logical_tensor(lt->id, &compiled_output);
                partition_outputs.push_back(compiled_output);
                assert(compiled_output.ndims > -1);
            }
            if (dynamic_callback) {
                dynamic_callback(partition_inputs, partition_outputs);
            }
            for (auto &lt : partition_inputs) {
                assert(lt.ndims > -1);
                lt_info_map[lt.id] = lt;
            }
            for (auto &lt : partition_outputs) {
                assert(lt.ndims > -1);
                lt_info_map[lt.id] = lt;
            }

            for (auto &lt : partition_inputs) {
                test_tensor placeholder(lt, &eng);
                if (random_data) {
                    float LO = 0.0f;
                    float HI = 200.0f;
                    float mean = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                    float div = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                    placeholder.fill<float>(mean, div);
                }
                execution_inputs.push_back(placeholder);
            }
            for (auto &lt : partition_outputs) {
                test_tensor placeholder(lt, &eng);
                execution_outputs.push_back(placeholder);
            }

            impl::stream_t &strm = *get_stream();
            auto inp = test_tensor::to_graph_tensor(execution_inputs);
            auto out = test_tensor::to_graph_tensor(execution_outputs);
            auto start = std::chrono::high_resolution_clock::now();
            auto status = cp.execute(&strm, inp, out);
            strm.wait();
            auto end = std::chrono::high_resolution_clock::now();
            ASSERT_EQ(status, impl::status::success);
            auto res = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            times.push_back(res);
        }
        if (measure_time) {
            std::sort(times.begin(), times.end());
            std::cout << "Min: " << times[0] << "mc | Median: " << times[times.size() / 2] 
                      << "mc | 0.7%: " << times[times.size() * 0.7]
                      << "mc | 0.8%: " << times[times.size() * 0.8] << "mc | 0.9%: " << times[times.size() * 0.9] 
                      << "mc | Max: "  << times[times.size() - 1]  << "mc" << std::endl;
        }
    }
}

// test fp32 get partition + compile + execution of MHA graph
TEST(GCGraphTest, FP32MHACompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_simple_subgraph(&agraph, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

#include <dlfcn.h>
#include "compiler/jit/symbol_resolver.hpp"
#include "runtime/context.hpp"
#include "runtime/managed_thread_pool.hpp"
#include "common/utils.hpp"

TEST(GCGraphTest, CompileMe) {
    int fs = dnnl::impl::getenv_int_user("FS", 1);

    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    int64_t btc = dnnl::impl::getenv_int_user("BATCH_SZ", 256);
    compiler_utils::add_int8_mlp_subgraph(&agraph, 128, 3, {13, 512, 256, 128},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    // compile_execution_pipeline(agraph, 1, nullptr, true, 100, true);

    auto &compiler_backend_ptr
        = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    // ASSERT_EQ(partitions.size(), static_cast<size_t>(expected_part_size));
    // if (dynamic_callback) { ASSERT_EQ(expected_part_size, 1); }
    // TODO(yifei): generalize the logic here
    // sort partitions to run forward first according to num ops
    std::sort(partitions.begin(), partitions.end(),
            [](std::shared_ptr<impl::partition_impl_t> a,
                    std::shared_ptr<impl::partition_impl_t> b) {
                return a->get_ops().size() < b->get_ops().size();
            });

    std::unordered_map<size_t, impl::logical_tensor_t> lt_info_map;

    impl::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    // replace partition inputs info if needed
    for (size_t i = 0; i < partition_inputs.size(); ++i) {
        if (lt_info_map.find(partition_inputs[i].id) != lt_info_map.end()) {
            partition_inputs[i] = lt_info_map[partition_inputs[i].id];
        }
    }

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }
    impl::compiled_partition_t cp(p);
    impl::engine_t &eng = *get_engine();
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    // num_runs = measure_time ? num_runs : 1;
    // std::vector<int64_t> times;
    // srand(time(NULL));
    bool random_data = true;
    // for (size_t rn = 0; rn < num_runs; rn++) {

    // }

    std::string outpath;

    if (fs == 1) {
        outpath = "/localdisk/dchigare/repos/oneDNN/build/dumps/to_compile/fused_mlp.so";
    } else if (fs == 0) {
        outpath = "/localdisk/dchigare/repos/oneDNN/build/dumps/to_compile/seq_mlp.so";
    } else if (fs == 2) {
        outpath = "/localdisk/dchigare/repos/oneDNN/build/dumps/to_compile/mal_seq_mlp.so";
    }
    auto run_func = "int8_mlp_pattern_100004_0wrapper";
    std::cout << "opening: " << outpath << std::endl;

    void *compiled_module = dlopen(outpath.data(), RTLD_LAZY);
    if (!compiled_module) {
        std::ostringstream os;
        os << "dlopen: " << dlerror();
        throw std::runtime_error(os.str());
    }
    for (auto &kv : dnnl::impl::graph::gc::get_runtime_function_map()) {
        void **ptr = reinterpret_cast<void **>(
                dlsym(compiled_module, (kv.first + "_fptr").c_str()));
        if (ptr) { *ptr = kv.second; }
    }
    typedef void (*init_func_t)(void *ctx, void *mod);
    auto deleter = [](char* p) { delete[] p; };

    // Allocate 512KB of memory and wrap it in a shared pointer with the custom deleter
    std::shared_ptr<char> memory(new char[8192 * 1024], deleter);
    auto init_func = reinterpret_cast<init_func_t>(
            dlsym(compiled_module, "__sc_init__"));
    if (init_func) { init_func(nullptr, memory.get()); }

    auto stream = dnnl::impl::graph::gc::runtime::get_default_stream();
    using generic_val = dnnl::impl::graph::gc::generic_val;
    typedef void (*main_func_t)(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args);
    auto func = reinterpret_cast<main_func_t>(dlsym(compiled_module, run_func));

//  * @param logical_tensor_4 [f32 [256, 32, 768, 64] @ ABCD]
//  * @param logical_tensor_0 [f32 [256, 32, 768, 64] @ ABCD]
//  * @param logical_tensor_1 [f32 [256, 32, 64, 768] @ ABCD]
//  * @param logical_tensor_3 [f32 [256, 32, 768, 64] @ ABCD]

    auto lt4_sz = 8192 * 8192;
    auto lt0_sz = 8192 * 8192;
    auto lt1_sz = 8192 * 8192;
    auto lt3_sz = 8192 * 8192;
    int num_runs = 100;
    std::vector<int64_t> times;
    for (int i=0; i<num_runs; i++) {
        std::vector<test_tensor> execution_inputs;
        std::vector<test_tensor> execution_outputs;
        partition_outputs.clear();
        for (auto &lt : outputs) {
            impl::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt->id, &compiled_output);
            partition_outputs.push_back(compiled_output);
            assert(compiled_output.ndims > -1);
        }
        for (auto &lt : partition_inputs) {
            assert(lt.ndims > -1);
            lt_info_map[lt.id] = lt;
        }
        for (auto &lt : partition_outputs) {
            assert(lt.ndims > -1);
            lt_info_map[lt.id] = lt;
        }

        for (auto &lt : partition_inputs) {
            test_tensor placeholder(lt, &eng);
            if (random_data) {
                float LO = 0.0f;
                float HI = 200.0f;
                float mean = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                float div = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                placeholder.fill<float>(mean, div);
            }
            execution_inputs.push_back(placeholder);
        }
        for (auto &lt : partition_outputs) {
            test_tensor placeholder(lt, &eng);
            execution_outputs.push_back(placeholder);
        }
        auto inp = test_tensor::to_graph_tensor(execution_inputs);
        auto out = test_tensor::to_graph_tensor(execution_outputs);

        auto args = new generic_val[inp.size() + out.size()];
        int idx = 0;
        for (auto& o : out) {
            args[idx].v_ptr = o.get_data_handle();
            idx++;
        }
        for (auto& in: inp) {
            args[idx].v_ptr = in.get_data_handle();
            idx++;
        }
        // args[0].v_ptr = lt4;
        // args[1].v_ptr = lt0;
        // args[2].v_ptr = lt1;
        // args[3].v_ptr = lt3;
        auto start = std::chrono::high_resolution_clock::now();

        dnnl::impl::graph::gc::runtime::thread_manager::cur_mgr.run_main_function((dnnl::impl::graph::gc::runtime::thread_manager::main_func_t)func, (dnnl::impl::graph::gc::runtime::stream_t *)stream, memory.get(), args);

        // stream->wait();
        auto end = std::chrono::high_resolution_clock::now();

        auto res = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        times.push_back(res);

        delete[] args;
    }

    std::sort(times.begin(), times.end());
    std::cout << "Min: " << times[0] << "mc | Median: " << times[times.size() / 2] 
                << "mc | 0.7%: " << times[times.size() * 0.7]
                << "mc | 0.8%: " << times[times.size() * 0.8] << "mc | 0.9%: " << times[times.size() * 0.9] 
                << "mc | Max: "  << times[times.size() - 1]  << "mc" << std::endl;

    
    auto print_n = [](int8_t* arr, int n, const char* name) {
        std::cout << name << ": ";
        for (int i = 0; i < n; i++) {
            std::cout << static_cast<int>(arr[i]) << " ";
        }
        std::cout << std::endl;
    };

    int sz = dnnl::impl::getenv_int_user("PR_TEST", 5);

    // print_n(lt4, sz, "lt4");
    // print_n(lt0, sz, "lt0");
    // print_n(lt1, sz, "lt1");
    // print_n(lt3, sz, "lt3");

}

TEST(GCGraphTest, FP32MHACompileExecution2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MHACompileExecution3_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative2(&agraph);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test ITEX pattern ends with StaticReshape
TEST(GCGraphTest, FP32MHACompileExecution4_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, false, false, impl::op_kind::StaticReshape);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHACompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(
            &agraph, false, true, false, 128, 384, 16, 1024, true);
    agraph.finalize();
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(
            &agraph, true, true, false, 128, 384, 16, 1024, true);
    agraph.finalize();
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHACompileExecutionDynamicQuantize2_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, true,
            impl::op_kind::Reorder, 128, 384, 16, 1024, true);
    agraph.finalize();
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecutionDynamicQuantize2_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, true,
            impl::op_kind::Reorder, 128, 384, 16, 1024, true);
    agraph.finalize();
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHACompileExecutionFake_CPU) {
    REQUIRE_AVX512(); // fake int8, so it only requires avx512
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative2(&agraph, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test int8 get partition + compile + execution of MHA graph
TEST(GCGraphTest, INT8MHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, true, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecution2_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, true, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test ITEX pattern ends with StaticReshape
TEST(GCGraphTest, INT8BF16MHACompileExecution3_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, true, true, impl::op_kind::StaticReshape);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MLPCompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test bf16 get partition + compile + execution of MHA graph
TEST(GCGraphTest, BF16MHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test infer shape + compile + execute for fp32 MHA
// the created graph is without intermediate shapes
// if infer shape failed, the compilation will also fail
TEST(GCGraphTest, FP32MHAInfershapeCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_infer_shape(&agraph);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test compile + multithreading execution for fp32 MHA
TEST(GCGraphTest, FP32MHACompileExecutionMultiThreading_CPU) {
    REQUIRE_AVX512();
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
    GTEST_SKIP();
#endif
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false);
    agraph.finalize();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), impl::status::success);

    int thread_num = 2;

    auto thread_func = [&](size_t tid) {
        std::vector<test_tensor> execution_inputs;
        std::vector<test_tensor> execution_outputs;

        for (auto &lt : partition_inputs) {
            test_tensor placeholder(lt, engine);
            execution_inputs.push_back(placeholder);
        }
        for (auto &lt : partition_outputs) {
            graph::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt.id, &compiled_output);
            test_tensor placeholder(compiled_output, engine);
            execution_outputs.push_back(placeholder);
        }
        impl::stream_t &strm = *get_stream();
        ASSERT_EQ(cp.execute(&strm,
                          test_tensor::to_graph_tensor(execution_inputs),
                          test_tensor::to_graph_tensor(execution_outputs)),
                impl::status::success);
    };

    std::vector<std::thread> workers;
    for (int t_num = 0; t_num < thread_num; t_num++) {
        workers.emplace_back(thread_func, t_num);
    }

    for (int t_num = 0; t_num < thread_num; t_num++) {
        workers[t_num].join();
    }
}

TEST(GCGraphTest, FP32MHAAlternativeCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32MHAAlternativeCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHAAlternativeCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16MHAAlternativeCompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MHAAlternative4CompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative4(&agraph, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHAAlternative4CompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative4(&agraph, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32BartMHACompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16BartMHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BartMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16BartMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32DistillBertMHA_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16DistillBertMHA_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32DistillBertMHA_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16DistillBertMHA_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MLPCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// TEST(GCTestSimple, MyTest) {
//     // REQUIRE_AMXBF16();
//     REQUIRE_CPU_ENGINE();
//     impl::graph_t agraph(engine->kind());
//     compiler_utils::add_mlp_subgraph(&agraph, true, 1, 5,
//             {479, 1024, 1024, 512, 256, 1},
//             {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
//                     impl::op_kind::ReLU, impl::op_kind::Sigmoid});
//     agraph.finalize();

//     compile_execution_pipeline(agraph, 1);
//     // REQUIRE_CPU_ENGINE();
//     // impl::graph_t agraph(engine->kind());
//     // // compiler_utils::add_simpmlp_subgraph(&agraph, true, 1, 5,
//     // //         {479, 1024, 1024, 512, 256, 1},
//     // //         {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
//     // //                 impl::op_kind::ReLU, impl::op_kind::Sigmoid});
//     // compiler_utils::add_mlp_subgraph(&agraph, true, 1, 2, {13, 512, 1}, {graph::op_kind::ReLU, graph::op_kind::ReLU});
//     // agraph.finalize();

//     // compile_execution_pipeline(agraph, 1);
// }

// test bf16 get partition + compile + execution of MHA graph
TEST(GCTestSimple, MyTest) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    int64_t size = dnnl::impl::getenv_int_user("PR_TEST", 4096);
    // int64_t size = 1024;
    // compiler_utils::add_simple_mlp_subgraph(&agraph, false, size, 2, {size, size, size});
    compiler_utils::add_simple_mlp_subgraph(&agraph, true, size, 2, {size, size, size}, {impl::op_kind::Wildcard, impl::op_kind::Wildcard, impl::op_kind::Wildcard});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1, nullptr, true, 100, true);
}

TEST(GCTestSimple00, MyTest) {
    // REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    int64_t btc = dnnl::impl::getenv_int_user("BATCH_SZ", 256);
    compiler_utils::add_simple_mats_subgraph(&agraph);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1, nullptr, true, 100, true);
}

TEST(GCTestSimple0, MyTest) {
    // REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    int64_t btc = dnnl::impl::getenv_int_user("BATCH_SZ", 256);
    compiler_utils::add_mlp_subgraph(&agraph, false, btc, 2, {btc, btc, btc},
            {impl::op_kind::Wildcard, impl::op_kind::Wildcard, impl::op_kind::Wildcard});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1, nullptr, true, 100, true);
}

TEST(GCTestSimple2, MyTest) {
    // REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    int64_t btc = dnnl::impl::getenv_int_user("BATCH_SZ", 256);
    compiler_utils::add_int8_mlp_subgraph(&agraph, btc, 3, {13, 512, 256, 128},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1, nullptr, true, 100, true);
}

TEST(GCTestSimple3, MLPSpeed) {
    // REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    int64_t btc = dnnl::impl::getenv_int_user("BATCH_SZ", 256);
    compiler_utils::add_int8_mlp_subgraph(&agraph, /*batch_size=*/btc, /*num_layers=*/5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid}
    );
    agraph.finalize();

    compile_execution_pipeline(agraph, 1, nullptr, true, 100, true);
}

TEST(GCTestSimple4, FP32MHACompileExecution2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    int64_t btc = dnnl::impl::getenv_int_user("BATCH_SZ", 128);
    int64_t sql = dnnl::impl::getenv_int_user("SEQ_LEN", 128);
    int64_t hdn = dnnl::impl::getenv_int_user("HEAD_N", 16);
    int64_t hdd = dnnl::impl::getenv_int_user("HEAD_D", 1024);
    int64_t dty = dnnl::impl::getenv_int_user("DTY", 1024);

    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, /*f16=*/dty == 0, /*int8=*/dty == 1, false, /*batch_size=*/btc,
                                      /*seq_len=*/sql, /*num_head=*/hdn, /*head_dim=*/hdd);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1, nullptr, true, 100, true);
}

TEST(GCGraphTest, BF16MLPCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, true, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MLPDynamicGraphCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, -1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs, static_cast<int64_t>(1),
                    std::placeholders::_1, std::placeholders::_2));
}

TEST(GCGraphTest, INT8MLPDynamicGraphCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, -1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs, static_cast<int64_t>(1),
                    std::placeholders::_1, std::placeholders::_2));
}

TEST(GCGraphTest, BF16MLPDynamicGraphCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, true, -1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs, static_cast<int64_t>(1),
                    std::placeholders::_1, std::placeholders::_2));
}

TEST(GCGraphTest, FP32MLPTrainingGraphCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward});
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32MLPTrainingGraphCompileExecution2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward},
            false, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MLPTrainingGraphCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward},
            true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32MHATrainingGraphCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MHATrainingGraphCompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32MHATrainingGraphCompileExecution2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MHATrainingGraphCompileExecution2_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32IdenticalBottleneckCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32ConvolutionalBottleneckCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8IdenticalBottleneckCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8IdenticalBottleneckCompileExecutionNXC_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}},
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NXC", "XIO");
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8ConvolutionalBottleneckCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_convolutional_bottleneck_resblock(&agraph,
            id_gen, {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32IdenticalBottleneckTrainingCompileExecution_CPU) {
    REQUIRE_AVX512();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32IdenticalBottleneckTrainingCompileExecutionNCX_CPU) {
    REQUIRE_AVX512();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}}, false, false,
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NCX", "OIX");
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32ConvolutionalBottleneckTrainingCompileExecution_CPU) {
    REQUIRE_AVX512();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_training_subgraph(
            &agraph, id_gen, {1, 56, 56, 64},
            {{1, 1, 64, 256}, {1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16IdenticalBottleneckTrainingCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {64, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}}, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16ConvolutionalBottleneckTrainingCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
#if SC_BUILTIN_JIT_ENABLED
    if (::dnnl::impl::graph::gc::get_default_context()->flags_.jit_kind_
            == ::dnnl::impl::graph::gc::jit_kind::xbyak) {
        GTEST_SKIP();
        return;
    }
#endif
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_training_subgraph(
            &agraph, id_gen, {64, 56, 56, 64},
            {{1, 1, 64, 256}, {1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}},
            true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, INT8IdenticalBottleneckCompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}},
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NCX", "OIX",
            true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest,
        INT8IdenticalBottleneckCompileExecutionDynamicQuantizeNXC_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}},
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NXC", "XIO",
            true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest,
        INT8ConvolutionalBottleneckCompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_convolutional_bottleneck_resblock(&agraph,
            id_gen, {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}},
            {{2, 2}, {1, 1}, {2, 2}, {1, 1}}, {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
            "NCX", "OIX", true);

    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MulQuantizeCompileExecution_CPU) {
    REQUIRE_AVX512();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_mul_quantize_subgraph(
            &agraph, id_gen, {4, 1, 4096});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32GPTMHACompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16GPTMHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32GPTMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16GPTMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32LLAMAMHACompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16LLAMAMHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32LLAMAMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16LLAMAMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32GPTMLPCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16GPTMLPCompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32GPTMLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16GPTMLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32LLAMAMLPCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16LLAMAMLPCompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32LLAMAMLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16LLAMAMLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// shall be re-enabled after fp32/bf16 concat patterns are re-enabled
#if 0
TEST(GCGraphTest, GptjBf16Concat_CPU) {
    REQUIRE_AVX512();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_gptj_concat_subgraph(&agraph);
    agraph.finalize();

    // should hit add_to_concat_permute_concat_to pattern
    compile_execution_pipeline(agraph, 1);
}
#endif

TEST(GCGraphTest, GptjInt8Bf16Concat_CPU) {
    REQUIRE_AVX512();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_gptj_concat_subgraph(&agraph, true);
    agraph.finalize();

    // should hit mul_mul_add_concat_permute_concat_quant pattern
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, LlamaInt8Bf16Concat_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_llama_concat_subgraph(&agraph, true);
    agraph.finalize();

    // should hit add_typecast_concat_typecasts_quant pattern
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32STARCODERMHACompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16STARCODERMHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32STARCODERMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16STARCODERMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    SKIP_WHEN_SINGLE_OP_PATTERN_ON();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}
