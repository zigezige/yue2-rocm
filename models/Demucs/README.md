---
tags:
- audio
- music
- music-source-separation
- demucs
---

# HTDemucs

Weights for the `htdemucs` pretrained model of
[Demucs](https://github.com/adefossez/demucs), music source separation in the
waveform domain. See the [demucs repository](https://github.com/adefossez/demucs)
for how to use them.

This is a bag of 1 model(s), whose outputs are averaged (see `htdemucs.yaml`
for the per source weights). Each `.safetensors` file contains the weights of one
model, with its class and init arguments as json in the safetensors metadata, and
the full training metadata in the matching `.json` sidecar.

- `955717e8` (`955717e8.safetensors`)
