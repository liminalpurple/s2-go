package s2

import (
	"encoding/binary"
	"math"
	"time"
)

// Audio is synthesised mono speech.
type Audio struct {
	Samples    []float32 // in [-1, 1]
	SampleRate int       // in Hz
}

// Duration returns how long the audio plays for.
func (a *Audio) Duration() time.Duration {
	if a.SampleRate == 0 {
		return 0
	}
	return time.Duration(len(a.Samples)) * time.Second / time.Duration(a.SampleRate)
}

// PCM16 returns the audio as 16-bit little-endian PCM, clipping samples outside [-1, 1].
func (a *Audio) PCM16() []byte {
	return PCM16(a.Samples)
}

// WAV returns the audio as a 16-bit PCM WAV file.
func (a *Audio) WAV() []byte {
	pcm := a.PCM16()
	out := make([]byte, 44, 44+len(pcm))
	copy(out[0:], "RIFF")
	binary.LittleEndian.PutUint32(out[4:], uint32(36+len(pcm)))
	copy(out[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(out[16:], 16)                     // fmt chunk size
	binary.LittleEndian.PutUint16(out[20:], 1)                      // PCM
	binary.LittleEndian.PutUint16(out[22:], 1)                      // mono
	binary.LittleEndian.PutUint32(out[24:], uint32(a.SampleRate))   // sample rate
	binary.LittleEndian.PutUint32(out[28:], uint32(a.SampleRate)*2) // bytes per second
	binary.LittleEndian.PutUint16(out[32:], 2)                      // bytes per frame
	binary.LittleEndian.PutUint16(out[34:], 16)                     // bits per sample
	copy(out[36:], "data")
	binary.LittleEndian.PutUint32(out[40:], uint32(len(pcm)))
	return append(out, pcm...)
}

// PCM16 converts samples to 16-bit little-endian PCM, clipping samples outside [-1, 1]. Useful
// for the chunks SynthesizeStream passes on.
func PCM16(samples []float32) []byte {
	out := make([]byte, 2*len(samples))
	for i, s := range samples {
		v := math.Round(float64(s) * math.MaxInt16)
		v = math.Max(math.MinInt16, math.Min(math.MaxInt16, v))
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(v)))
	}
	return out
}
