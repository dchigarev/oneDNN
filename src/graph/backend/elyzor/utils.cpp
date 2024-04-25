#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {
namespace utils {

const std::unordered_map<op_kind_t, std::string, enum_hash_t> &
get_supported_ops() {
    static const std::unordered_map<op_kind_t, std::string, enum_hash_t>
            elyzor_backend_op = {{op_kind::Add, "add"}, {op_kind::Abs, "abs"},
                    {op_kind::AbsBackward, "abs_bwd"},
                    {op_kind::Subtract, "sub"}, {op_kind::Multiply, "mul"},
                    {op_kind::Divide, "div"}, {op_kind::Pow, "pow"},
                    {op_kind::MatMul, "matmul"},
                    {op_kind::Quantize, "quantize"},
                    {op_kind::Dequantize, "dequantize"},
                    {op_kind::DynamicDequantize, "dynamic_dequantize"},
                    {op_kind::DynamicQuantize, "dynamic_quantize"},
                    {op_kind::StaticReshape, "static_reshape"},
                    {op_kind::StaticTranspose, "transpose"},
                    {op_kind::LogSoftmax, "log_softmax"},
                    {op_kind::LogSoftmaxBackward, "log_softmax_bwd"},
                    {op_kind::SoftMax, "softmax"},
                    {op_kind::SoftMaxBackward, "softmax_bwd"},
                    {op_kind::Reorder, "reorder"},
                    {op_kind::SoftPlus, "soft_plus"},
                    {op_kind::SoftPlusBackward, "soft_plus_bwd"},
                    {op_kind::TypeCast, "cast"}, {op_kind::ReLU, "relu"},
                    {op_kind::Sigmoid, "sigmoid"}, {op_kind::GELU, "gelu"},
                    {op_kind::ReLUBackward, "relu_backprop"},
                    {op_kind::Reciprocal, "reciprocal"},
                    {op_kind::SigmoidBackward, "sigmoid_backprop"},
                    {op_kind::GELUBackward, "gelu_backprop"},
                    {op_kind::ReduceSum, "reduce_sum"},
                    {op_kind::BiasAdd, "add"},
                    {op_kind::Convolution, "conv_fwd"},
                    {op_kind::ConvolutionBackwardData, "conv_bwd_data"},
                    {op_kind::ConvolutionBackwardWeights, "conv_bwd_weight"},
                    {op_kind::BatchNormForwardTraining,
                            "batchnorm_forward_training"},
                    {op_kind::BatchNormTrainingBackward,
                            "batchnorm_training_backprop"},
                    {op_kind::Maximum, "max"}, {op_kind::Minimum, "min"},
                    {op_kind::LayerNorm, "layernorm"},
                    {op_kind::Select, "select"}, {op_kind::Square, "square"},
                    {op_kind::Tanh, "tanh"},
                    {op_kind::TanhBackward, "tanh_bwd"}, {op_kind::Exp, "exp"},
                    {op_kind::ReduceL2, "reduce_l2"},
                    {op_kind::ReduceL1, "reduce_l1"},
                    {op_kind::ReduceMax, "reduce_max"},
                    {op_kind::ReduceMean, "reduce_mean"},
                    {op_kind::ReduceProd, "reduce_prod"},
                    {op_kind::ReduceMin, "reduce_min"},
                    {op_kind::Concat, "concat"}, {op_kind::Clamp, "clamp"},
                    {op_kind::ClampBackward, "clamp_bwd"},
                    {op_kind::HardSigmoid, "hardsigmoid"},
                    {op_kind::HardSigmoidBackward, "hardsigmoid_bwd"},
                    {op_kind::LeakyReLU, "leaky_relu"}, {op_kind::Log, "log"},
                    {op_kind::Elu, "elu"}, {op_kind::EluBackward, "elu_bwd"},
                    {op_kind::MaxPool, "pooling_max"},
                    {op_kind::MaxPoolBackward, "pooling_max_backprop"},
                    {op_kind::AvgPoolBackward, "pooling_avg_backprop"},
                    {op_kind::Mish, "mish"},
                    {op_kind::MishBackward, "mish_bwd"},
                    {op_kind::AvgPool, "pooling_avg"},
                    {op_kind::SqrtBackward, "sqrt_bwd"},
                    {op_kind::HardSwish, "hardswish"},
                    {op_kind::HardSwishBackward, "hardswish_bwd"},
                    {op_kind::BatchNormInference, "batchnorm_inference"}};
    return elyzor_backend_op;
}

std::vector<op_kind_t> get_supported_op_kinds() {
    std::vector<op_kind_t> ret;
    for (const auto &pair : get_supported_ops()) {
        ret.emplace_back(pair.first);
    }
    return ret;
}

} // namespace utils
} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl
