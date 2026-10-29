// Package s2 provides Go bindings for s2.cpp, which runs Fish Audio S2 Pro text-to-speech
// through GGML.
//
// Load a model once, then synthesise speech in a random voice or a voice cloned from a
// reference clip. Text can carry S2's inline tags, such as "[whisper]" or "[laughing]", which
// change how the speech after them sounds.
//
// Built with Fish Audio. s2.cpp and the S2 Pro model are under the Fish Audio Research License,
// which allows research and non-commercial use only.
//
// Example:
//
//	model, err := s2.LoadModel("s2-pro-q4_k_m.gguf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	audio, err := model.Synthesize(context.Background(), "[softly] Hello there.", nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	os.WriteFile("hello.wav", audio.WAV(), 0o644)
package s2

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -ls2go -lstdc++ -lm -lpthread -lgomp
#include <stdlib.h>
#include "wrapper.h"
*/
import "C"

import (
	gocontext "context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// Backend is the device that runs the model.
type Backend int

const (
	BackendCPU    Backend = C.S2GO_BACKEND_CPU
	BackendVulkan Backend = C.S2GO_BACKEND_VULKAN
	BackendCUDA   Backend = C.S2GO_BACKEND_CUDA
	BackendMetal  Backend = C.S2GO_BACKEND_METAL
)

// defaultBackend is CUDA when built with the cublas tag, and the CPU otherwise
var defaultBackend = BackendCPU

func init() {
	// s2.cpp logs progress to stdout at its default level; keep only errors
	C.s2go_set_log_level(0)
}

// SetLogLevel sets s2.cpp's log level: 0 errors (the default), 1 warnings, 2 info, 3 debug.
func SetLogLevel(level int) {
	C.s2go_set_log_level(C.int32_t(level))
}

// Model is a loaded S2 model, with its codec and tokenizer. It's safe for concurrent use, but
// synthesises one request at a time.
type Model struct {
	mu         sync.Mutex
	ptr        *C.s2go_model
	sampleRate int
}

// LoadModel loads an S2 model GGUF, which holds both the transformer and the audio codec. By
// default the tokenizer is the tokenizer.json beside the model, as s2.cpp looks for it.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	cfg := modelConfig{
		tokenizer: filepath.Join(filepath.Dir(path), "tokenizer.json"),
		backend:   defaultBackend,
		gpuLayers: -1,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	for _, file := range []string{path, cfg.tokenizer} {
		if _, err := os.Stat(file); err != nil {
			return nil, fmt.Errorf("failed to load model: %w", err)
		}
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cTokenizer := C.CString(cfg.tokenizer)
	defer C.free(unsafe.Pointer(cTokenizer))

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ptr := C.s2go_model_load(cPath, cTokenizer, C.int32_t(cfg.backend), C.int32_t(cfg.gpuDevice),
		C.int32_t(cfg.gpuLayers))
	if ptr == nil {
		return nil, lastError()
	}
	return &Model{ptr: ptr, sampleRate: int(C.s2go_model_sample_rate(ptr))}, nil
}

// Close frees the model and its GPU memory. Closing twice is safe.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ptr != nil {
		C.s2go_model_free(m.ptr)
		m.ptr = nil
	}
	return nil
}

// SampleRate returns the sample rate of the audio the model produces, in Hz.
func (m *Model) SampleRate() int {
	return m.sampleRate
}

// Voice is a voice to synthesise in, cloned from a reference clip or loaded from a saved
// .s2voice profile.
type Voice struct {
	ptr *C.s2go_voice
}

// LoadVoice loads a voice profile saved by Voice.Save or s2.cpp's --save-voice.
func LoadVoice(path string) (*Voice, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ptr := C.s2go_voice_load(cPath)
	if ptr == nil {
		return nil, lastError()
	}
	return &Voice{ptr: ptr}, nil
}

// CloneVoice clones a voice from a WAV or MP3 reference clip and its transcript. S2 recommends
// 5-30 seconds of clean speech. The clip also sets how the voice speaks by default, so a varied,
// expressive reference gives a more expressive voice.
func (m *Model) CloneVoice(audio []byte, transcript string) (*Voice, error) {
	if len(audio) == 0 {
		return nil, errors.New("reference audio is empty")
	}
	if transcript == "" {
		return nil, errors.New("reference transcript is empty")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}

	cAudio := C.CBytes(audio)
	defer C.free(cAudio)
	cTranscript := C.CString(transcript)
	defer C.free(unsafe.Pointer(cTranscript))

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ptr := C.s2go_voice_clone(m.ptr, cAudio, C.size_t(len(audio)), cTranscript, 0)
	if ptr == nil {
		return nil, lastError()
	}
	return &Voice{ptr: ptr}, nil
}

// Save writes the voice as an .s2voice profile.
func (v *Voice) Save(path string) error {
	if v.ptr == nil {
		return errors.New("voice is closed")
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if !C.s2go_voice_save(v.ptr, cPath) {
		return lastError()
	}
	return nil
}

// Transcript returns the transcript of the voice's reference clip.
func (v *Voice) Transcript() string {
	if v.ptr == nil {
		return ""
	}
	return C.GoString(C.s2go_voice_transcript(v.ptr))
}

// SampleRate returns the sample rate the voice was encoded at, in Hz.
func (v *Voice) SampleRate() int {
	if v.ptr == nil {
		return 0
	}
	return int(C.s2go_voice_sample_rate(v.ptr))
}

// Close frees the voice. Closing twice is safe.
func (v *Voice) Close() error {
	if v.ptr != nil {
		C.s2go_voice_free(v.ptr)
		v.ptr = nil
	}
	return nil
}

// Synthesize speaks text in a voice, or in a random speaker when voice is nil, and returns the
// whole utterance. S2's quality fades after about 800 tokens (around 37 seconds of audio), so
// synthesise long text a sentence or two at a time.
func (m *Model) Synthesize(ctx gocontext.Context, text string, voice *Voice, opts ...SynthesisOption) (*Audio, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cfg := newSynthesisConfig(opts)
	m.mu.Lock()
	defer m.mu.Unlock()
	cVoice, err := m.check(voice)
	if err != nil {
		return nil, err
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var n C.size_t
	samples := C.s2go_synthesize(m.ptr, cVoice, cText, cfg.params, &n)
	if samples == nil {
		return nil, lastError()
	}
	defer C.s2go_free_samples(samples)
	out := make([]float32, int(n))
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(samples)), int(n)))
	return &Audio{Samples: out, SampleRate: m.sampleRate}, nil
}

// SynthesizeStream speaks text like Synthesize, but passes the audio to fn as it's decoded.
// Returning an error from fn stops synthesis and returns that error, and cancelling ctx stops it
// with ctx's error. The post-processing options need the whole utterance, so they're rejected.
func (m *Model) SynthesizeStream(ctx gocontext.Context, text string, voice *Voice, fn func(samples []float32) error,
	opts ...SynthesisOption) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	cfg := newSynthesisConfig(opts)
	if cfg.postProcessing() {
		return errors.New("trimming and normalisation need the whole utterance, so they don't apply to streaming")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	cVoice, err := m.check(voice)
	if err != nil {
		return err
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	state := &streamState{ctx: ctx, fn: fn}
	handle := cgo.NewHandle(state)
	defer handle.Delete()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ok := C.s2go_synthesize_stream(m.ptr, cVoice, cText, cfg.params, C.uintptr_t(handle))
	// The callback's error or cancellation explains a stop better than s2.cpp's message
	if state.err != nil {
		return state.err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !ok {
		return lastError()
	}
	return nil
}

// check validates the model and voice for synthesis; the caller holds m.mu
func (m *Model) check(voice *Voice) (*C.s2go_voice, error) {
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}
	if voice == nil {
		return nil, nil
	}
	if voice.ptr == nil {
		return nil, errors.New("voice is closed")
	}
	if rate := voice.SampleRate(); rate != m.sampleRate {
		return nil, fmt.Errorf("voice was encoded at %d Hz, but the model produces %d Hz", rate, m.sampleRate)
	}
	return voice.ptr, nil
}

type streamState struct {
	ctx gocontext.Context
	fn  func([]float32) error
	err error
}

//export goS2Samples
func goS2Samples(handle C.uintptr_t, samples *C.float, n C.size_t) C.bool {
	state := cgo.Handle(handle).Value().(*streamState)
	if state.err != nil {
		return false
	}
	// Copy, as the buffer belongs to s2.cpp
	out := make([]float32, int(n))
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(samples)), int(n)))
	if err := state.fn(out); err != nil {
		state.err = err
		return false
	}
	return true
}

//export goS2Cancelled
func goS2Cancelled(handle C.uintptr_t) C.bool {
	state := cgo.Handle(handle).Value().(*streamState)
	return C.bool(state.ctx.Err() != nil)
}

// lastError returns the wrapper's error; the caller must still be on the thread that failed
func lastError() error {
	return errors.New(C.GoString(C.s2go_last_error()))
}
