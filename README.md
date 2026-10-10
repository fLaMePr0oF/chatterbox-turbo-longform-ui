# Chatterbox Long-Form UI

A Gradio front end for **Resemble AI Chatterbox** that adds practical long-form narration workflows on top of Chatterbox Turbo and Multilingual V3.

This repository contains two standalone front ends:

- `longform_turbo_gui.py` — Chatterbox Turbo long-form narrator
- `longform_v3_gui.py` — Chatterbox Multilingual V3 long-form narrator

## Demo and screenshots

The Turbo demo and screenshots are available in the [media assets package](https://github.com/fLaMePr0oF/chatterbox-turbo-longform-ui). Once uploaded to `docs/images/`, they can be embedded here.

## Features

### Shared long-form features

- Automatic sentence-aware chunking for scripts longer than the model's normal short-form input
- Adjustable maximum characters per chunk
- Live highlighting of paragraphs that exceed the current chunk limit
- Separate pauses between generated chunks and between paragraphs
- Multiple blank lines create proportionally longer paragraph pauses
- Reference-audio upload / microphone recording
- Display of the active reference-audio file path
- Seed control and display of the seed used for the last generation
- Chunk preview before / after generation
- WAV output assembled from all generated chunks
- CUDA use when available

### Turbo-specific features

- Non-verbal event-tag soundboard:
  - `[clear throat]`
  - `[sigh]`
  - `[shush]`
  - `[cough]`
  - `[groan]`
  - `[sniff]`
  - `[gasp]`
  - `[chuckle]`
  - `[laugh]`
- Event tags count as **one character** for the front end's chunk-length calculations
- Temperature, Top P, Top K, repetition penalty and reference loudness controls
- Reference voice required

### Multilingual V3-specific features

- 23-language selector
- Exaggeration control
- CFG / pace control
- Optional reference voice, with fallback to the model's default conditioning
- Pronunciation replacements retained in the current V3 build

## Requirements

Chatterbox itself is not vendored in this repository.

Resemble AI currently recommends Python 3.11 for Chatterbox. Install Chatterbox first, either from PyPI:

```bash
pip install chatterbox-tts
```

or from the upstream source repository.

Then install the small front-end dependency set:

```bash
pip install -r requirements.txt
```

### PyTorch / CUDA note

The correct PyTorch build depends on your GPU and CUDA environment. Newer NVIDIA GPUs may require a newer CUDA-enabled PyTorch build than the version selected by Chatterbox's default dependency pins. If Chatterbox installs a CPU-only or incompatible build, install the appropriate PyTorch / TorchAudio / TorchVision packages from the official PyTorch index for your system.

## Running

Turbo:

```bash
python longform_turbo_gui.py
```

Multilingual V3:

```bash
python longform_v3_gui.py
```

Each app launches a local Gradio interface in your browser.

## Long-form pause behaviour

The editor preserves repeated line breaks.

With a paragraph pause of 550 ms:

```text
Paragraph one.
Paragraph two.
```

creates one paragraph pause.

```text
Paragraph one.

Paragraph two.
```

creates two paragraph pauses.

Additional blank lines continue to multiply the configured paragraph pause.

## Project status

This is an independent front-end project built around Resemble AI's Chatterbox models. It is not an official Resemble AI project.

Contributions, bug reports and UI / workflow improvements are welcome.

## Upstream project

Chatterbox is developed by Resemble AI:

https://github.com/resemble-ai/chatterbox

Chatterbox is distributed under the MIT License. This project retains attribution to the upstream project and is also distributed under the MIT License.
