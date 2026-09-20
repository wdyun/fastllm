#include "models/qwen3_5.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;

namespace {

void Check(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class DFlashMapModel : public Qwen3_5Model {
public:
    DFlashMapModel(bool quantized) {
        block_cnt = 1;
        num_attention_heads = num_key_value_heads = 2;
        head_dim = 128;
        num_k_heads = 2;
        num_v_heads = 4;
        head_k_dim = head_v_dim = 128;
        dflashEnabled = true;
        dflashQuantizedLinearWeights = quantized;
        dflashLayers = 1;
    }
};

DataType MappedType(DFlashMapModel &model, const std::string &sourceName) {
    auto result = model.GetTensorMap({sourceName});
    auto it = result.find(sourceName);
    Check(it != result.end(), "tensor was not mapped: " + sourceName);
    Check(it->second.size() == 1, "unexpected mapping count for: " + sourceName);
    return it->second[0].second;
}

void Expect(bool quantized, const std::string &sourceName, DataType expected) {
    DFlashMapModel model(quantized);
    DataType actual = MappedType(model, sourceName);
    Check(actual == expected,
          "mapping mismatch for " + sourceName + " (quantized=" +
              (quantized ? "true" : "false") + "): got " +
              std::to_string((int)actual) + " expected " +
              std::to_string((int)expected));
}

}  // namespace

int main() {
    try {
        for (bool quantized : {false, true}) {
            const DataType linearType = quantized
                ? DataType::DATA_AUTO_SOURCE
                : DataType::BFLOAT16;
            // Quantized drafts resolve their source layout per tensor, while
            // floating-point drafts keep the established BF16 mapping.
            Expect(quantized, "fc.weight", linearType);
            Expect(quantized,
                   "candidate_selector.hidden_projection.weight", linearType);
            Expect(quantized, "layers.0.mlp.gate_proj.weight", linearType);
            Expect(quantized, "layers.0.mlp.up_proj.weight", linearType);
            Expect(quantized, "layers.0.mlp.down_proj.weight", linearType);
            Expect(quantized, "layers.0.self_attn.q_proj.weight", linearType);
            Expect(quantized, "layers.0.self_attn.k_proj.weight", linearType);
            Expect(quantized, "layers.0.self_attn.v_proj.weight", linearType);
            Expect(quantized, "layers.0.self_attn.o_proj.weight", linearType);
            Expect(quantized,
                   "layers.0.attention_conv.kernel_projection.weight",
                   linearType);
            Expect(quantized,
                   "layers.0.mlp_conv.kernel_projection.weight", linearType);

            // Norms, convolution kernels and the selector codebooks stay on
            // their fixed floating-point layouts regardless of quantization.
            Expect(quantized, "hidden_norm.weight", DataType::FLOAT32);
            Expect(quantized, "norm.weight", DataType::FLOAT32);
            Expect(quantized, "layers.0.input_layernorm.weight",
                   DataType::FLOAT32);
            Expect(quantized, "layers.0.post_attention_layernorm.weight",
                   DataType::FLOAT32);
            Expect(quantized, "layers.0.self_attn.q_norm.weight",
                   DataType::FLOAT32);
            Expect(quantized, "layers.0.self_attn.k_norm.weight",
                   DataType::FLOAT32);
            Expect(quantized, "layers.0.attention_conv.base_kernel",
                   DataType::BFLOAT16);
            Expect(quantized, "layers.0.mlp_conv.base_kernel",
                   DataType::BFLOAT16);
            Expect(quantized, "candidate_selector.predecessor_codebook",
                   DataType::BFLOAT16);
            Expect(quantized, "candidate_selector.successor_codebook",
                   DataType::BFLOAT16);
        }
        std::cout << "dflash quantized tensor map test passed." << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "dflash quantized tensor map test failed: "
                  << error.what() << std::endl;
        return 1;
    }
}