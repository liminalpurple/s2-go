package s2_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"

	s2 "github.com/liminalpurple/s2-go"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// peak returns the loudest sample in a stretch of audio
func peak(samples []float32) float64 {
	p := 0.0
	for _, s := range samples {
		p = math.Max(p, math.Abs(float64(s)))
	}
	return p
}

// save writes audio to TEST_S2_OUTPUT, when it's set, for listening to
func save(name string, audio *s2.Audio) {
	if dir := os.Getenv("TEST_S2_OUTPUT"); dir != "" {
		Expect(os.WriteFile(filepath.Join(dir, name+".wav"), audio.WAV(), 0o644)).To(Succeed())
	}
}

var _ = Describe("Audio", func() {
	audio := &s2.Audio{Samples: []float32{0, 0.5, -0.5, 1.5, -1.5}, SampleRate: 44100}

	It("should convert to 16-bit PCM, clipping out-of-range samples", func() {
		pcm := audio.PCM16()
		Expect(pcm).To(HaveLen(10))
		values := make([]int16, 5)
		for i := range values {
			values[i] = int16(binary.LittleEndian.Uint16(pcm[2*i:]))
		}
		Expect(values).To(Equal([]int16{0, 16384, -16384, math.MaxInt16, math.MinInt16}))
	})

	It("should write a mono 16-bit WAV file", func() {
		wav := audio.WAV()
		Expect(wav).To(HaveLen(44 + 10))
		Expect(string(wav[0:4])).To(Equal("RIFF"))
		Expect(string(wav[8:16])).To(Equal("WAVEfmt "))
		Expect(binary.LittleEndian.Uint16(wav[22:])).To(Equal(uint16(1)))
		Expect(binary.LittleEndian.Uint32(wav[24:])).To(Equal(uint32(44100)))
		Expect(binary.LittleEndian.Uint16(wav[34:])).To(Equal(uint16(16)))
		Expect(binary.LittleEndian.Uint32(wav[40:])).To(Equal(uint32(10)))
	})

	It("should report its duration", func() {
		second := &s2.Audio{Samples: make([]float32, 22050), SampleRate: 44100}
		Expect(second.Duration().Seconds()).To(BeNumerically("~", 0.5, 1e-9))
	})
})

var _ = Describe("Loading", func() {
	It("should report a missing model", func() {
		_, err := s2.LoadModel("does-not-exist.gguf")
		Expect(err).To(MatchError(ContainSubstring("does-not-exist.gguf")))
	})

	It("should report a missing voice", func() {
		_, err := s2.LoadVoice("does-not-exist.s2voice")
		Expect(err).To(MatchError(ContainSubstring("cannot open voice profile")))
	})
})

// TEST_S2_MODEL names an S2 model GGUF with its tokenizer.json beside it. TEST_S2_VOICE optionally
// names a saved voice to synthesise with too.
var _ = Describe("Synthesis", Label("integration"), Ordered, func() {
	var model *s2.Model
	const text = "Well, hello there. It's lovely to meet you."

	BeforeAll(func() {
		path := os.Getenv("TEST_S2_MODEL")
		if path == "" {
			Skip("TEST_S2_MODEL not set - skipping integration tests")
		}
		var err error
		model, err = s2.LoadModel(path)
		Expect(err).NotTo(HaveOccurred())
		Expect(model.SampleRate()).To(Equal(44100))
	})

	AfterAll(func() {
		if model != nil {
			Expect(model.Close()).To(Succeed())
		}
	})

	It("should speak in a random voice", func() {
		audio, err := model.Synthesize(context.Background(), text, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(audio.SampleRate).To(Equal(44100))
		Expect(audio.Duration().Seconds()).To(BeNumerically(">", 1))
		Expect(peak(audio.Samples)).To(BeNumerically(">", 0.01))
		save("random", audio)
	})

	It("should trim silence from both ends", func() {
		audio, err := model.Synthesize(context.Background(), text, nil, s2.WithTrimSilence())
		Expect(err).NotTo(HaveOccurred())
		// s2.cpp leaves up to 80 ms of silence, plus 10 ms beside the speech
		edge := audio.SampleRate / 10
		Expect(peak(audio.Samples[:edge])).To(BeNumerically(">", 0.01))
		Expect(peak(audio.Samples[len(audio.Samples)-edge:])).To(BeNumerically(">", 0.01))
	})

	It("should normalise the peak", func() {
		audio, err := model.Synthesize(context.Background(), text, nil, s2.WithPeakNormalization())
		Expect(err).NotTo(HaveOccurred())
		Expect(peak(audio.Samples)).To(BeNumerically("~", 0.95, 1e-3))
	})

	It("should stream the audio as it's decoded", func() {
		// By default s2.cpp holds back the codec's whole window, so a short sentence arrives at once
		var chunks, samples int
		err := model.SynthesizeStream(context.Background(), text, nil, func(chunk []float32) error {
			chunks++
			samples += len(chunk)
			return nil
		}, s2.WithStreamStride(4), s2.WithStreamHoldback(4))
		Expect(err).NotTo(HaveOccurred())
		Expect(chunks).To(BeNumerically(">", 1))
		Expect(samples).To(BeNumerically(">", model.SampleRate()))
	})

	It("should stop streaming when cancelled or the callback fails", func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		err := model.SynthesizeStream(ctx, text, nil, func([]float32) error {
			cancel()
			return nil
		})
		Expect(err).To(MatchError(context.Canceled))

		stop := errors.New("stop here")
		err = model.SynthesizeStream(context.Background(), text, nil, func([]float32) error { return stop })
		Expect(err).To(MatchError(stop))
	})

	It("should refuse post-processing while streaming", func() {
		err := model.SynthesizeStream(context.Background(), text, nil, func([]float32) error { return nil },
			s2.WithTrimSilence())
		Expect(err).To(MatchError(ContainSubstring("whole utterance")))
	})

	It("should clone a voice, save it and load it back", func() {
		reference, err := model.Synthesize(context.Background(), text, nil, s2.WithTrimSilence())
		Expect(err).NotTo(HaveOccurred())
		voice, err := model.CloneVoice(reference.WAV(), text)
		Expect(err).NotTo(HaveOccurred())
		defer voice.Close()
		Expect(voice.Transcript()).To(Equal(text))
		Expect(voice.SampleRate()).To(Equal(44100))

		path := filepath.Join(GinkgoT().TempDir(), "clone.s2voice")
		Expect(voice.Save(path)).To(Succeed())
		loaded, err := s2.LoadVoice(path)
		Expect(err).NotTo(HaveOccurred())
		defer loaded.Close()
		Expect(loaded.Transcript()).To(Equal(text))

		audio, err := model.Synthesize(context.Background(), "[softly] And now in the same voice.", loaded)
		Expect(err).NotTo(HaveOccurred())
		Expect(audio.Duration().Seconds()).To(BeNumerically(">", 1))
		save("reference", reference)
		save("cloned", audio)
	})

	It("should reject a reference it can't decode", func() {
		_, err := model.CloneVoice([]byte("not audio"), "Hello.")
		Expect(err).To(MatchError(ContainSubstring("could not decode")))
	})

	It("should speak in a saved voice", func() {
		path := os.Getenv("TEST_S2_VOICE")
		if path == "" {
			Skip("TEST_S2_VOICE not set")
		}
		voice, err := s2.LoadVoice(path)
		Expect(err).NotTo(HaveOccurred())
		defer voice.Close()
		audio, err := model.Synthesize(context.Background(),
			"[low voice] There you are. [laughing] I was starting to think you'd forgotten me.", voice,
			s2.WithTrimSilence())
		Expect(err).NotTo(HaveOccurred())
		Expect(audio.Duration().Seconds()).To(BeNumerically(">", 1))
		save("saved-voice", audio)
	})

	It("should refuse to work once closed", func() {
		closed, err := s2.LoadModel(os.Getenv("TEST_S2_MODEL"))
		Expect(err).NotTo(HaveOccurred())
		Expect(closed.Close()).To(Succeed())
		Expect(closed.Close()).To(Succeed())
		_, err = closed.Synthesize(context.Background(), text, nil)
		Expect(err).To(MatchError(ContainSubstring("model is closed")))
	})
})
