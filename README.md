# s2-go

Go bindings for [s2.cpp](https://github.com/rodrigomatta/s2.cpp), the C++/GGML inference engine
for Fish Audio S2 Pro text-to-speech. Load a model, design or clone a voice, and synthesise
expressive speech directed by inline tags, on the CPU or a GPU.

Built with Fish Audio. s2.cpp and the S2 Pro model are under the Fish Audio Research License, which
allows research and non-commercial use only (see [License](#license)).

## Requirements

- Go, CMake and a C++17 compiler
- For NVIDIA GPUs: the CUDA Toolkit 12.4 or later (s2.cpp also supports Vulkan and Metal, which
  these bindings haven't been tested with)

## Building

Clone with the s2.cpp submodule, then build `libs2go.a`, which merges s2.cpp, its ggml and the
cgo wrapper into a single archive:

```bash
git clone --recurse-submodules https://github.com/liminalpurple/s2-go.git
cd s2-go

make libs2go.a                                            # CPU
BUILD_TYPE=cublas CUDA_ARCHITECTURES=86 make libs2go.a    # CUDA, here for an RTX 3090
```

Build Go code with the `cublas` tag for CUDA, which also makes CUDA the default backend:

```bash
go build -tags cublas ./...
```

When using s2-go from another module, put the directory holding `libs2go.a` on `LIBRARY_PATH`. As
everything s2.cpp needs is inside that archive, it can sit alongside other GGML-based libraries,
such as llama.cpp's, without their archives clashing.

## Getting a model

Quantised GGUFs, each holding the transformer and the audio codec, are at
[rodrigomt/s2-pro-gguf](https://huggingface.co/rodrigomt/s2-pro-gguf). Download one with
`tokenizer.json`, and keep them in the same folder: s2-go looks for the tokenizer beside the model.

`s2-pro-q4_k_m.gguf` is a good default. On an RTX 3090 it loads in about 1.5 seconds and speaks
faster than real time (a real-time factor of about 0.8), using about 7 GB of VRAM, and in our
listening it sounded the same as Q8_0.

## Quick start

```go
model, err := s2.LoadModel("models/s2-pro-q4_k_m.gguf")
if err != nil {
    log.Fatal(err)
}
defer model.Close()

audio, err := model.Synthesize(context.Background(),
    "[warm and relaxed, like talking to an old friend] It's been a while.", nil,
    s2.WithTrimSilence())
if err != nil {
    log.Fatal(err)
}
os.WriteFile("hello.wav", audio.WAV(), 0o644)
```

Without a voice, S2 chooses a different speaker every time.

## Finding a voice

S2 takes who is speaking from reference audio, and without one it picks a speaker to suit the
whole text. To settle on one voice, describe the delivery you want in a tag at the start of the
text, generate several candidates, and keep the one you like best:

```go
const delivery = "[warm, cheerful, slightly husky voice, relaxed and informal, not reading, " +
    "just talking] "
const script = "[cheerful] Morning! You're up early. [laughing] Don't tell me you actually slept " +
    "last night. [very softly, almost whispering] Here, sit yourself down, the kettle's just " +
    "boiled. [whisper] Between you and me, I've hidden the good biscuits behind the flour. [sigh] " +
    "Right, go on then. What's the plan for today?"

for i := range 4 {
    audio, err := model.Synthesize(ctx, delivery+script, nil, s2.WithTrimSilence())
    // ... save each as candidate-1.wav to candidate-4.wav, and listen
}
```

Then clone the one you chose, using the script without its tags as the transcript, and save it:

```go
voice, err := model.CloneVoice(candidate, "Morning! You're up early. Don't tell me you actually ...")
voice.Save("voices/mine.s2voice")

// From then on
voice, err := s2.LoadVoice("voices/mine.s2voice")
audio, err := model.Synthesize(ctx, "[low voice] There you are.", voice)
```

What we learned finding a voice this way:

- S2 was trained on captions of vocal features (emotion, prosody, laughter and the like), not
  descriptions of speakers, so it follows words about delivery well and words about who is
  speaking loosely. Gender and age shifted the odds without deciding them, and no wording we tried
  produced an Irish or Scottish accent
- The script pulls as hard as the description: a homely script gave homely voices, and a very
  English one gave English voices even when the description asked for another accent. Naming a
  place, whatever accent came out, seemed to give more natural-sounding voices
- Speakers vary a lot from one generation to the next, so generate three or four per prompt
- Use words that can only mean one thing: "low" can mean pitch, volume or mood, so say
  "low-pitched"; "strong" gave a voice more body
- Describe a situation rather than a style: "conversational" still sounded like a narrator, while
  "not reading, just talking" sounded like someone talking
- A clone takes its default delivery from the reference, so clone from an expressive clip like the
  one above: a calm reading gives a monotone voice

For a particular kind of speaker, Fish Audio's
[Voice Design](https://docs.fish.audio/features/voice-design) generates candidates from a
description of age, gender and accent: clone one of those with `CloneVoice` to use it locally.

## Inline tags

A tag describes how the text after it should sound, and lasts until the next tag. S2 accepts
free-form descriptions, and its model card lists common tags such as `[laughing]`, `[whisper]`,
`[low voice]`, `[sigh]`, `[emphasis]`, `[pause]` and `[pitch up]`. From our testing:

| Tag | Result |
| --- | --- |
| `[laughing]` | Reliable: speaks the following words while laughing. `[laughs]` did nothing |
| `[whisper]`, `[whispering intimately]` | Works, but give it six words or more to be heard |
| `[very softly, almost whispering]` | A clear step down; `[softly]` alone is subtle |
| `[low voice]` | Quieter and closer, rather than lower in pitch |
| `[sigh]` | Usually a sigh before the words |
| `[playfully teasing]` | Works; `[amused]` tended to sound indignant |
| `[slowly]` | Nothing: S2 doesn't control pace |
| `[looking up, a slow smile]` | Nothing: describe what can be heard, not what can be seen |

S2 doesn't pause between sentences, so to control gaps, synthesise sentences separately with
`WithTrimSilence`, which trims both ends, and add silence between them yourself.

## Streaming

`SynthesizeStream` passes audio to a callback as it's decoded, and stops when its context is
cancelled. By default s2.cpp holds back the codec's whole context window, so a short sentence
arrives all at once at the end. Set `WithStreamHoldback` (and `WithStreamStride`) lower for audio
sooner, at the risk of clicks at chunk boundaries:

```go
err := model.SynthesizeStream(ctx, text, voice, func(samples []float32) error {
    _, err := out.Write(s2.PCM16(samples))
    return err
}, s2.WithStreamStride(4), s2.WithStreamHoldback(4))
```

Trimming and normalisation need the whole utterance, so they aren't available while streaming.

## Testing

Tests that need a model skip unless it's set:

- `TEST_S2_MODEL` - an S2 GGUF, with `tokenizer.json` beside it
- `TEST_S2_VOICE` - optionally, a saved `.s2voice` to synthesise with
- `TEST_S2_OUTPUT` - optionally, a folder to save the generated audio to for listening

```bash
TEST_S2_MODEL=models/s2-pro-q4_k_m.gguf go test -tags cublas ./...
```

## License

The Go code here is MIT licensed. s2.cpp and the S2 Pro model are under the
[Fish Audio Research License](LICENSE-FISH-AUDIO.md), which allows research and non-commercial use
only: anything built with s2-go inherits that restriction. Commercial use needs a license from
[Fish Audio](https://fish.audio). See [NOTICE](NOTICE) for attribution.
