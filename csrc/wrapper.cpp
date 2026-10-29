#include "../wrapper.h"

#include "s2_audio.h"
#include "s2_log.h"
#include "s2_pipeline.h"
#include "s2_voice.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// Implemented in Go: receive streamed samples (return false to stop), and report cancellation
extern "C" bool goS2Samples(uintptr_t handle, const float* samples, size_t n_samples);
extern "C" bool goS2Cancelled(uintptr_t handle);

struct s2go_model {
    s2::Pipeline pipeline;
    s2::PipelineParams params;
};

struct s2go_voice {
    s2::VoiceProfile profile;
};

static thread_local std::string g_last_error;

const char* s2go_last_error(void) {
    return g_last_error.c_str();
}

void s2go_set_log_level(int32_t level) {
    s2::set_log_level(static_cast<s2::LogLevel>(level));
}

s2go_params s2go_default_params(void) {
    const s2::PipelineParams defaults;
    s2go_params params = {};
    params.max_new_tokens = defaults.gen.max_new_tokens;
    params.temperature = defaults.gen.temperature;
    params.top_p = defaults.gen.top_p;
    params.top_k = defaults.gen.top_k;
    params.min_tokens_before_end = defaults.gen.min_tokens_before_end;
    params.n_threads = defaults.gen.n_threads;
    params.stream_stride_frames = defaults.stream_decode_stride_frames;
    params.stream_holdback_frames = defaults.stream_holdback_frames;
    return params;
}

s2go_model* s2go_model_load(const char* model_path, const char* tokenizer_path, int32_t backend,
                            int32_t gpu_device, int32_t n_gpu_layers) {
    try {
        auto* model = new s2go_model();
        model->params.model_path = model_path;
        model->params.tokenizer_path = tokenizer_path;
        model->params.backend_type = static_cast<s2::BackendType>(backend);
        model->params.gpu_device = gpu_device;
        model->params.n_gpu_layers = n_gpu_layers;
        // Keep the codec on the model's backend rather than benchmarking backends on every load
        model->params.codec_auto_backend = false;
        model->params.codec_follow_backend = true;
        model->params.gen.verbose = false;
        if (!model->pipeline.init(model->params)) {
            delete model;
            g_last_error = std::string("failed to load model from ") + model_path;
            return nullptr;
        }
        return model;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

void s2go_model_free(s2go_model* model) {
    delete model;
}

int32_t s2go_model_sample_rate(const s2go_model* model) {
    return model->pipeline.output_sample_rate();
}

s2go_voice* s2go_voice_load(const char* path) {
    try {
        auto* voice = new s2go_voice();
        voice->profile = s2::VoiceProfile::load(path);
        return voice;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

s2go_voice* s2go_voice_clone(s2go_model* model, const void* audio, size_t audio_size, const char* transcript,
                             int32_t n_threads) {
    try {
        // Decode here, as the pipeline would carry on without a reference it can't read
        s2::AudioData reference;
        if (!s2::load_audio_from_memory(audio, audio_size, reference, model->pipeline.output_sample_rate())) {
            g_last_error = "could not decode the reference audio as WAV or MP3";
            return nullptr;
        }
        std::vector<int32_t> codes;
        int32_t n_frames = 0;
        if (!model->pipeline.encode_prompt_audio_data(reference, n_threads, codes, n_frames) || n_frames <= 0) {
            g_last_error = "failed to encode the reference audio";
            return nullptr;
        }
        // Fields as the pipeline fills them when saving a profile; the codes are frames by codebooks
        auto* voice = new s2go_voice();
        voice->profile.transcript = transcript;
        voice->profile.codes = std::move(codes);
        voice->profile.T_prompt = n_frames;
        voice->profile.num_codebooks = static_cast<int32_t>(voice->profile.codes.size() / n_frames);
        voice->profile.sample_rate = model->pipeline.output_sample_rate();
        return voice;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

bool s2go_voice_save(const s2go_voice* voice, const char* path) {
    try {
        if (!voice->profile.save(path)) {
            g_last_error = std::string("failed to save voice profile to ") + path;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return false;
    }
}

const char* s2go_voice_transcript(const s2go_voice* voice) {
    return voice->profile.transcript.c_str();
}

int32_t s2go_voice_sample_rate(const s2go_voice* voice) {
    return voice->profile.sample_rate;
}

void s2go_voice_free(s2go_voice* voice) {
    delete voice;
}

// request builds the parameters for one synthesis from the model's load parameters
static s2::PipelineParams request(const s2go_model* model, const s2go_voice* voice, const char* text,
                                  const s2go_params& params) {
    s2::PipelineParams p = model->params;
    p.text = text;
    p.prompt_text = voice ? voice->profile.transcript : "";
    p.gen.max_new_tokens = params.max_new_tokens;
    p.gen.temperature = params.temperature;
    p.gen.top_p = params.top_p;
    p.gen.top_k = params.top_k;
    p.gen.min_tokens_before_end = params.min_tokens_before_end;
    p.gen.n_threads = params.n_threads;
    p.stream_decode_stride_frames = params.stream_stride_frames;
    p.stream_holdback_frames = params.stream_holdback_frames;
    return p;
}

float* s2go_synthesize(s2go_model* model, const s2go_voice* voice, const char* text, s2go_params params,
                       size_t* n_samples) {
    try {
        const s2::PipelineParams p = request(model, voice, text, params);
        std::vector<float> audio;
        bool ok;
        if (voice) {
            ok = model->pipeline.synthesize_with_prompt_codes(p, voice->profile.codes.data(), voice->profile.T_prompt,
                                                              audio);
        } else {
            s2::AudioData no_reference;
            ok = model->pipeline.synthesize_raw(p, no_reference, audio);
        }
        if (!ok) {
            g_last_error = "synthesis failed";
            return nullptr;
        }
        // The pipeline applies these only when writing a file, so apply them here in its order
        const int32_t rate = model->pipeline.output_sample_rate();
        if (params.trim_silence && !audio.empty()) {
            // s2.cpp trims only the end, so trim the reversed audio for the start, with the same
            // threshold and minimum duration
            std::vector<float> trimmed = s2::audio_trim_trailing_silence(audio.data(), audio.size(), rate);
            std::reverse(trimmed.begin(), trimmed.end());
            trimmed = s2::audio_trim_trailing_silence(trimmed.data(), trimmed.size(), rate);
            std::reverse(trimmed.begin(), trimmed.end());
            // As s2.cpp does when saving, keep audio that's all silence rather than nothing
            if (!trimmed.empty()) {
                audio = std::move(trimmed);
            }
        }
        if (params.normalize_dynamic && !audio.empty()) {
            audio = s2::audio_normalize_dynamic(audio.data(), audio.size(), rate);
        }
        if (params.normalize_peak && !audio.empty()) {
            float peak = 0.0f;
            for (const float s : audio) {
                peak = std::max(peak, std::fabs(s));
            }
            if (peak > 1e-6f) {
                const float scale = 0.95f / peak;
                for (float& s : audio) {
                    s *= scale;
                }
            }
        }
        auto* samples = static_cast<float*>(std::malloc(audio.size() * sizeof(float)));
        if (!samples && !audio.empty()) {
            g_last_error = "out of memory";
            return nullptr;
        }
        std::memcpy(samples, audio.data(), audio.size() * sizeof(float));
        *n_samples = audio.size();
        return samples;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

void s2go_free_samples(float* samples) {
    std::free(samples);
}

// goSink passes streamed samples to Go, which can stop the stream or cancel it
class goSink : public s2::StreamingSink {
  public:
    explicit goSink(uintptr_t handle) : handle_(handle) {}

    bool on_pcm_data(const float* data, size_t n_samples) override {
        return goS2Samples(handle_, data, n_samples);
    }
    bool on_header(const uint8_t*, size_t) override { return true; }
    void on_done() override {}
    void on_error(const std::string& message) override { error = message; }
    bool is_cancelled() const override { return goS2Cancelled(handle_); }

    std::string error;

  private:
    uintptr_t handle_;
};

bool s2go_synthesize_stream(s2go_model* model, const s2go_voice* voice, const char* text, s2go_params params,
                            uintptr_t handle) {
    try {
        const s2::PipelineParams p = request(model, voice, text, params);
        goSink sink(handle);
        bool ok;
        if (voice) {
            ok = model->pipeline.synthesize_streaming_with_prompt_codes(p, voice->profile.codes.data(),
                                                                       voice->profile.T_prompt, sink);
        } else {
            s2::AudioData no_reference;
            ok = model->pipeline.synthesize_streaming_raw(p, no_reference, sink);
        }
        if (!ok) {
            g_last_error = sink.error.empty() ? "streaming synthesis failed" : sink.error;
        }
        return ok;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return false;
    }
}
