#ifndef S2GO_WRAPPER_H
#define S2GO_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles: a loaded model (s2::Pipeline) and a voice (s2::VoiceProfile)
typedef struct s2go_model s2go_model;
typedef struct s2go_voice s2go_voice;

// Backends, matching s2::BackendType
enum s2go_backend {
    S2GO_BACKEND_CPU = -1,
    S2GO_BACKEND_VULKAN = 0,
    S2GO_BACKEND_CUDA = 1,
    S2GO_BACKEND_METAL = 2,
};

typedef struct {
    int32_t max_new_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    int32_t min_tokens_before_end;
    int32_t n_threads; // 0 uses every core
    // Streaming: decode every this many frames (0 lets s2.cpp choose), holding back this many
    // trailing frames until more arrive (-1 holds back the codec's whole context window)
    int32_t stream_stride_frames;
    int32_t stream_holdback_frames;
    // Post-processing of a whole utterance, applied by s2go_synthesize in this order
    bool trim_silence;
    bool normalize_dynamic;
    bool normalize_peak;
} s2go_params;

// The error from the last failed call on this thread
const char* s2go_last_error(void);

// s2.cpp's log level: 0 error, 1 warning, 2 info, 3 debug
void s2go_set_log_level(int32_t level);

// s2.cpp's defaults, as its CLI uses them
s2go_params s2go_default_params(void);

// Load a model GGUF (weights and codec) with its tokenizer.json, NULL on error
s2go_model* s2go_model_load(const char* model_path, const char* tokenizer_path, int32_t backend,
                            int32_t gpu_device, int32_t n_gpu_layers);
void s2go_model_free(s2go_model* model);
int32_t s2go_model_sample_rate(const s2go_model* model);

// Load a saved .s2voice profile, NULL on error
s2go_voice* s2go_voice_load(const char* path);
// Clone a voice from WAV or MP3 bytes and their transcript, NULL on error
s2go_voice* s2go_voice_clone(s2go_model* model, const void* audio, size_t audio_size, const char* transcript,
                             int32_t n_threads);
bool s2go_voice_save(const s2go_voice* voice, const char* path);
const char* s2go_voice_transcript(const s2go_voice* voice);
int32_t s2go_voice_sample_rate(const s2go_voice* voice);
void s2go_voice_free(s2go_voice* voice);

// Synthesise text in a voice (NULL for a random speaker) into mono float samples at the model's
// sample rate. Returns the samples (free with s2go_free_samples) and their count, NULL on error.
float* s2go_synthesize(s2go_model* model, const s2go_voice* voice, const char* text, s2go_params params,
                       size_t* n_samples);
void s2go_free_samples(float* samples);

// Stream synthesis: samples go to goS2Samples as they're decoded, and goS2Cancelled is polled to
// stop early. The post-processing fields of params don't apply. Returns false on error or
// cancellation.
bool s2go_synthesize_stream(s2go_model* model, const s2go_voice* voice, const char* text, s2go_params params,
                            uintptr_t handle);

#ifdef __cplusplus
}
#endif

#endif
