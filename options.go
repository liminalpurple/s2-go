package s2

/*
#include "wrapper.h"
*/
import "C"

// ModelOption configures LoadModel.
type ModelOption func(*modelConfig)

type modelConfig struct {
	tokenizer string
	backend   Backend
	gpuDevice int
	gpuLayers int
}

// WithTokenizer sets the path of the model's tokenizer.json, by default the one beside the model.
func WithTokenizer(path string) ModelOption {
	return func(c *modelConfig) { c.tokenizer = path }
}

// WithBackend sets the device that runs the model: CUDA by default when built with the cublas
// tag, and the CPU otherwise.
func WithBackend(backend Backend) ModelOption {
	return func(c *modelConfig) { c.backend = backend }
}

// WithGPUDevice selects the GPU by index, 0 by default.
func WithGPUDevice(device int) ModelOption {
	return func(c *modelConfig) { c.gpuDevice = device }
}

// WithGPULayers sets how many of the model's transformer layers run on the GPU: -1, the default,
// puts all of them there.
func WithGPULayers(layers int) ModelOption {
	return func(c *modelConfig) { c.gpuLayers = layers }
}

// SynthesisOption configures Synthesize and SynthesizeStream.
type SynthesisOption func(*synthesisConfig)

type synthesisConfig struct {
	params C.s2go_params
}

// newSynthesisConfig starts from s2.cpp's defaults, as its CLI uses them
func newSynthesisConfig(opts []SynthesisOption) synthesisConfig {
	cfg := synthesisConfig{params: C.s2go_default_params()}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func (c synthesisConfig) postProcessing() bool {
	return bool(c.params.trim_silence || c.params.normalize_dynamic || c.params.normalize_peak)
}

// WithTemperature sets the sampling temperature, 0.8 by default.
func WithTemperature(temperature float32) SynthesisOption {
	return func(c *synthesisConfig) { c.params.temperature = C.float(temperature) }
}

// WithTopP sets nucleus sampling's cumulative probability, 0.8 by default.
func WithTopP(p float32) SynthesisOption {
	return func(c *synthesisConfig) { c.params.top_p = C.float(p) }
}

// WithTopK limits sampling to the k most likely tokens, 30 by default.
func WithTopK(k int) SynthesisOption {
	return func(c *synthesisConfig) { c.params.top_k = C.int32_t(k) }
}

// WithMaxTokens caps the tokens generated, 1024 by default (about 47 seconds of audio).
func WithMaxTokens(n int) SynthesisOption {
	return func(c *synthesisConfig) { c.params.max_new_tokens = C.int32_t(n) }
}

// WithThreads sets the CPU threads used, 0 by default for every core.
func WithThreads(n int) SynthesisOption {
	return func(c *synthesisConfig) { c.params.n_threads = C.int32_t(n) }
}

// WithStreamStride sets how many frames SynthesizeStream decodes at a time, each about 46 ms of
// audio. The default, 0, lets s2.cpp choose.
func WithStreamStride(frames int) SynthesisOption {
	return func(c *synthesisConfig) { c.params.stream_stride_frames = C.int32_t(frames) }
}

// WithStreamHoldback sets how many trailing frames SynthesizeStream holds back until more arrive.
// The default, -1, holds back the codec's whole context window, which for a short sentence means
// everything arrives at the end; lower values start sooner but may click at chunk boundaries.
func WithStreamHoldback(frames int) SynthesisOption {
	return func(c *synthesisConfig) { c.params.stream_holdback_frames = C.int32_t(frames) }
}

// WithTrimSilence trims silence from both ends of the utterance, so gaps between utterances can
// be set exactly. Not for streaming.
func WithTrimSilence() SynthesisOption {
	return func(c *synthesisConfig) { c.params.trim_silence = true }
}

// WithDynamicNormalization evens out loudness across the utterance. Not for streaming.
func WithDynamicNormalization() SynthesisOption {
	return func(c *synthesisConfig) { c.params.normalize_dynamic = true }
}

// WithPeakNormalization scales the utterance so its loudest sample is 0.95. Not for streaming.
func WithPeakNormalization() SynthesisOption {
	return func(c *synthesisConfig) { c.params.normalize_peak = true }
}
