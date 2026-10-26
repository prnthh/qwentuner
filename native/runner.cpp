#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct hook_state {
    std::map<int, std::map<int, float>> adds;
    std::map<int, std::vector<float>> activations;
    std::map<int, float> residual_norms;
    std::map<int, int> hits;
    bool capture = false;
    bool calibrate = false;
    bool prefill = true;
    std::string error;
};

static int layer_of(const char * name, const std::string & prefix) {
    std::string s(name);
    if (s.rfind(prefix, 0) != 0) return -1;
    auto suffix = s.substr(prefix.size());
    if (suffix.empty() || suffix.find_first_not_of("0123456789") != std::string::npos) return -1;
    return std::stoi(suffix);
}

static bool evaluate(ggml_tensor * t, bool ask, void * data) {
    auto & h = *static_cast<hook_state *>(data);
    const int a = layer_of(t->name, "ffn_swiglu-");
    const int r = layer_of(t->name, "l_out-");
    bool wanted = (a >= 0 && ((h.capture && h.prefill) || h.adds.count(a))) ||
                  (r >= 0 && h.calibrate && h.prefill);
    if (ask) return wanted;
    if (!wanted) return true;
    // The scheduler synchronizes this tensor before the callback and resumes consumers afterwards.
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        h.error = std::string("Expected contiguous F32 hook: ") + t->name;
        return false;
    }
    const size_t width = t->ne[0];
    const size_t rows = ggml_nelements(t) / width;
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
    if (a >= 0) {
        if (h.capture && h.prefill) h.activations[a] = std::vector<float>(values.end() - width, values.end());
        if (h.adds.count(a)) {
            for (const auto & [neuron, amount] : h.adds.at(a)) {
                if (neuron < 0 || size_t(neuron) >= width) {
                    h.error = "Neuron out of range at layer " + std::to_string(a);
                    return false;
                }
                for (size_t row = 0; row < rows; ++row) values[row * width + neuron] += amount;
            }
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
            h.hits[a] += int(rows);
        }
    } else {
        std::vector<float> norms;
        for (size_t row = rows > 1 ? 1 : 0; row < rows; ++row) {
            double norm = 0;
            for (size_t i = 0; i < width; ++i) norm += double(values[row * width + i]) * values[row * width + i];
            norms.push_back(float(std::sqrt(norm)));
        }
        std::sort(norms.begin(), norms.end());
        size_t m = norms.size() / 2;
        h.residual_norms[r] = norms.size() % 2 ? norms[m] : (norms[m - 1] + norms[m]) / 2;
    }
    return true;
}

static std::string piece(const llama_vocab * vocab, llama_token id) {
    std::vector<char> buf(256);
    int n = llama_token_to_piece(vocab, id, buf.data(), buf.size(), 0, false);
    if (n < 0) {
        buf.resize(-n);
        n = llama_token_to_piece(vocab, id, buf.data(), buf.size(), 0, false);
    }
    if (n < 0) throw std::runtime_error("Could not decode token");
    return std::string(buf.data(), n);
}

static void set_batch(llama_batch_ext * batch, const std::vector<llama_token> & tokens, int start, bool all_logits) {
    llama_batch_ext_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos pos = start + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
        llama_batch_ext_set_output_logits(batch, idx, all_logits || i + 1 == tokens.size());
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: qwentuner-runner MODEL.gguf [gpu_layers=99] [context=2048]\n";
        return 1;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 2 ? std::stoi(argv[2]) : 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    const auto * vocab = llama_model_get_vocab(model);
    hook_state hooks;
    auto cp = llama_context_default_params();
    cp.n_ctx = argc > 3 ? std::stoi(argv[3]) : 2048;
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.cb_eval = evaluate;
    cp.cb_eval_user_data = &hooks;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 3; }
    auto * batch = llama_batch_ext_init(ctx);
    std::cout << json({{"ready", true}, {"layers", llama_model_n_layer(model)}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto req = json::parse(line);
            hooks = hook_state{};
            hooks.capture = req.value("capture", false);
            hooks.calibrate = req.value("calibrate", false);
            for (const auto & s : req.value("steering", json::array())) {
                int layer = s.at("layer").get<int>();
                float amount = s.at("amount").get<float>();
                if (layer < 0 || layer >= llama_model_n_layer(model) || !std::isfinite(amount))
                    throw std::runtime_error("Invalid layer or nonfinite amount");
                hooks.adds[layer][s.at("neuron").get<int>()] += amount;
            }
            std::string prompt = req.at("prompt");
            int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
            if (count <= 0) throw std::runtime_error("Empty prompt");
            std::vector<llama_token> tokens(count);
            if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), count, true, true) < 0)
                throw std::runtime_error("Tokenization failed");
            int max_tokens = req.value("max_tokens", 96);
            if (max_tokens < 0 || count + max_tokens > int(llama_n_ctx(ctx)))
                throw std::runtime_error("Prompt plus generation exceeds context");
            if (count > int(llama_n_ubatch(ctx))) throw std::runtime_error("Prompt must fit one microbatch");
            llama_memory_clear(llama_get_memory(ctx), true);
            set_batch(batch, tokens, 0, hooks.calibrate);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0 || !hooks.error.empty())
                throw std::runtime_error("Prefill failed: " + hooks.error);
            for (const auto & [layer, _] : hooks.adds)
                if (!hooks.hits.count(layer)) throw std::runtime_error("Missing ffn_swiglu hook at layer " + std::to_string(layer));
            if (hooks.capture && int(hooks.activations.size()) != llama_model_n_layer(model))
                throw std::runtime_error("Incomplete activation capture: model hook names unsupported");
            if (hooks.calibrate && int(hooks.residual_norms.size()) != llama_model_n_layer(model))
                throw std::runtime_error("Incomplete residual calibration");
            hooks.prefill = false;
            std::string text;
            std::vector<int> generated;
            bool stopped = false;
            for (int step = 0; step < max_tokens; ++step) {
                const float * logits = llama_get_logits_ith(ctx, -1);
                const int nv = llama_vocab_n_tokens(vocab);
                llama_token id = std::max_element(logits, logits + nv) - logits;
                if (llama_vocab_is_eog(vocab, id)) { stopped = true; break; }
                generated.push_back(id);
                text += piece(vocab, id);
                if (step + 1 == max_tokens) break;
                set_batch(batch, {id}, count + step, false);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0 || !hooks.error.empty())
                    throw std::runtime_error("Decode failed: " + hooks.error);
            }
            json result = {{"text", text}, {"token_ids", generated}, {"prompt_tokens", count},
                           {"stopped_eos", stopped}, {"hook_hits", json::object()},
                           {"residual_norms", json::object()}, {"activations", json::object()}};
            for (const auto & [l, v] : hooks.hits) result["hook_hits"][std::to_string(l)] = v;
            for (const auto & [l, v] : hooks.residual_norms) result["residual_norms"][std::to_string(l)] = v;
            for (const auto & [l, v] : hooks.activations) result["activations"][std::to_string(l)] = v;
            std::cout << result.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
        } catch (const std::exception & e) {
            std::cout << json({{"error", e.what()}}).dump() << std::endl;
        }
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
