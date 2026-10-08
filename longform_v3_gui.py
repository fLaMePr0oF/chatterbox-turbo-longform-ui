import re
import copy
import random
import logging
import numpy as np
import torch
import soundfile as sf
import gradio as gr

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


# ------------------------------------------------------------
# General setup
# ------------------------------------------------------------

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRONUNCIATIONS = {
    "AI": "Ā eye",
    "OpenAI": "Open Ā eye",
    "ChatGPT": "Chat G P T",
    "GPT": "G P T",
    "LLM": "L L M",
    "LLMs": "L L M's",
    "GPU": "G P U",
    "GPUs": "G P U's",
    "CPU": "C P U",
    "CPUs": "C P U's",
    "API": "Ā P eye",
    "APIs": "Ā P eyes",
    "RTX": "R T X",
    "DLSS": "D L S S",
    "VRAM": "V ram",
    "NVIDIA": "en-VID-ee-uh",
    "GeForce": "Gee Force",
    "CUDA": "coo-duh",
}

def apply_pronunciations(text):
    replacements = sorted(
        PRONUNCIATIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for original, spoken in replacements:
        pattern = (
            r"(?<!\w)"
            + re.escape(original)
            + r"(?!\w)"
        )

        text = re.sub(
            pattern,
            spoken,
            text,
            flags=re.IGNORECASE
        )

    return text

# ------------------------------------------------------------
# Narration editor CSS / JavaScript
# ------------------------------------------------------------

EDITOR_CSS = """
#narration-script-hidden {
    display: none !important;
}

#narration-editor {
    min-height: 360px;
    max-height: 600px;
    overflow-y: auto;
    padding: 12px;

    border: 1px solid #666;
    border-radius: 8px;

    background: #27272a;
    color: white;

    font-size: 16px;
    line-height: 1.5;

    white-space: pre-wrap;
    overflow-wrap: break-word;

    outline: none;
}

#narration-editor:focus {
    border-color: #888;
}

#narration-editor:empty::before {
    content: "Paste your full YouTube narration here...";
    color: #888;
    pointer-events: none;
}

#narration-editor-status {
    margin-top: 6px;
    font-size: 13px;
    color: #aaa;
}

::highlight(too-long) {
    color: #ff7088;
}

/* ---------------------------------------------------------
   Consistent application panels
   --------------------------------------------------------- */

.app-panel {
    border: 1px solid #555 !important;
    border-radius: 10px !important;
    background: #27272a !important;
    padding: 16px 18px !important;
    margin-bottom: 14px !important;
}

.app-header {
    padding: 18px 20px !important;
}

.app-title {
    margin: 0 0 4px 0 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    color: white !important;
}

.app-subtitle {
    margin: 0 0 14px 0 !important;
    font-size: 15px !important;
    color: #d4d4d8 !important;
}

.app-instructions {
    margin: 0 !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
    color: #bdbdc7 !important;
}

.section-title {
    margin: 0 0 4px 0 !important;
    font-size: 19px !important;
    font-weight: 650 !important;
    color: white !important;
}

.section-help {
    margin: 0 0 12px 0 !important;
    font-size: 13px !important;
    color: #aaa !important;
    line-height: 1.45 !important;
}
"""


EDITOR_JS = r"""
function getChunkLimit() {
    const root = document.querySelector("#chunk-size-slider");

    if (!root) {
        return 260;
    }

    const input = root.querySelector("input");

    if (!input) {
        return 260;
    }

    const value = parseInt(input.value, 10);

    return Number.isFinite(value)
        ? value
        : 260;
}


function getHiddenTextarea() {
    const root = document.querySelector(
        "#narration-script-hidden"
    );

    if (!root) {
        return null;
    }

    return root.querySelector("textarea");
}


function getTextNodes(root) {
    const walker = document.createTreeWalker(
        root,
        NodeFilter.SHOW_TEXT
    );

    const nodes = [];
    let node;

    while ((node = walker.nextNode())) {
        nodes.push(node);
    }

    return nodes;
}


function getPlainText(editor) {
    return editor.innerText
        .replace(/\r/g, "")
        .replace(/\u00a0/g, " ");
}


function syncToGradio(text) {
    const textarea = getHiddenTextarea();

    if (!textarea) {
        return;
    }

    const nativeSetter =
        Object.getOwnPropertyDescriptor(
            HTMLTextAreaElement.prototype,
            "value"
        ).set;

    nativeSetter.call(textarea, text);

    textarea.dispatchEvent(
        new Event("input", { bubbles: true })
    );

    textarea.dispatchEvent(
        new Event("change", { bubbles: true })
    );
}


function clearHighlight() {
    if (window.CSS && CSS.highlights) {
        CSS.highlights.delete("too-long");
    }
}


function updateStatus(text, limit, overLimitCount) {
    const status = document.querySelector(
        "#narration-editor-status"
    );

    if (!status) {
        return;
    }

    const paragraphs = text.length
        ? text.split("\n").filter(
            paragraph => paragraph.length > 0
        ).length
        : 0;

    let message =
        text.length.toLocaleString() +
        " characters | " +
        paragraphs +
        (
            paragraphs === 1
                ? " paragraph"
                : " paragraphs"
        );

    if (overLimitCount > 0) {
        message +=
            " | " +
            overLimitCount +
            " over " +
            limit +
            " characters";

        status.style.color = "#ff7088";
    } else {
        message +=
            " | All paragraphs within " +
            limit +
            " characters";

        status.style.color = "#aaa";
    }

    status.textContent = message;
}


function updateHighlight() {
    const editor = document.querySelector(
        "#narration-editor"
    );

    if (!editor) {
        return;
    }

    clearHighlight();

    const limit = getChunkLimit();
    const text = getPlainText(editor);

    syncToGradio(text);

    /*
    Each newline is preserved as a paragraph-break unit.
    */
    const paragraphs = text.split("\n");

    const targets = [];

    let offset = 0;
    let overLimitCount = 0;

    for (const paragraph of paragraphs) {
        const start = offset;
        const end = start + paragraph.length;

        if (paragraph.length > limit) {
            targets.push({
                start: start,
                end: end
            });

            overLimitCount++;
        }

        /*
        +1 accounts for the newline.
        */
        offset = end + 1;
    }

    updateStatus(
        text,
        limit,
        overLimitCount
    );

    if (targets.length === 0) {
        return;
    }

    const nodes = getTextNodes(editor);
    const ranges = [];

    for (const target of targets) {
        let running = 0;

        let startNode = null;
        let startOffset = 0;

        let endNode = null;
        let endOffset = 0;

        for (const node of nodes) {
            const length = node.textContent.length;
            const nodeStart = running;
            const nodeEnd = running + length;

            if (
                startNode === null &&
                target.start >= nodeStart &&
                target.start <= nodeEnd
            ) {
                startNode = node;

                startOffset = Math.min(
                    target.start - nodeStart,
                    length
                );
            }

            if (
                target.end >= nodeStart &&
                target.end <= nodeEnd
            ) {
                endNode = node;

                endOffset = Math.min(
                    target.end - nodeStart,
                    length
                );

                break;
            }

            running = nodeEnd;
        }

        if (startNode && endNode) {
            const range = new Range();

            range.setStart(
                startNode,
                startOffset
            );

            range.setEnd(
                endNode,
                endOffset
            );

            ranges.push(range);
        }
    }

    if (
        ranges.length > 0 &&
        window.Highlight
    ) {
        const highlight = new Highlight(
            ...ranges
        );

        CSS.highlights.set(
            "too-long",
            highlight
        );
    }
}


function insertOneLineBreak() {
    document.execCommand(
        "insertLineBreak",
        false
    );
}


function installNarrationEditor() {
    const editor = document.querySelector(
        "#narration-editor"
    );

    const chunkRoot = document.querySelector(
        "#chunk-size-slider"
    );

    const hiddenTextarea = getHiddenTextarea();

    if (
        !editor ||
        !chunkRoot ||
        !hiddenTextarea
    ) {
        setTimeout(
            installNarrationEditor,
            250
        );

        return;
    }

    if (editor.dataset.installed === "true") {
        return;
    }

    editor.dataset.installed = "true";

    /*
    Native typing, backspace, delete and cursor movement
    are left alone. Only Enter is normalised to one line break.
    */
    editor.addEventListener(
        "input",
        updateHighlight
    );

    editor.addEventListener(
        "keydown",
        event => {
            if (event.key === "Enter") {
                event.preventDefault();

                insertOneLineBreak();
                updateHighlight();
            }
        }
    );

    const chunkInput = chunkRoot.querySelector(
        "input"
    );

    if (chunkInput) {
        chunkInput.addEventListener(
            "input",
            updateHighlight
        );

        chunkInput.addEventListener(
            "change",
            updateHighlight
        );
    }

    updateHighlight();
}


setTimeout(
    installNarrationEditor,
    500
);
"""


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def set_seed(seed: int):
    if seed == 0:
        seed = random.SystemRandom().randint(
            1,
            2**32 - 1
        )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    return seed


def split_long_sentence(sentence, max_chars):
    words = sentence.split()

    chunks = []
    current = ""

    for word in words:
        candidate = (
            word
            if not current
            else current + " " + word
        )

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = word

    if current:
        chunks.append(current)

    return chunks


def split_text(text, max_chars=260):
    """
    Returns tuples:
        (chunk_text, paragraph_breaks_after)

    A normal chunk inside a paragraph has:
        paragraph_breaks_after = 0

    The final chunk of a paragraph stores the number of newline
    characters that followed that paragraph in the original text.

    This means:
        one Enter   = 1 paragraph pause
        two Enters  = 2 paragraph pauses
        three Enters = 3 paragraph pauses

    Empty lines do not need any text of their own.
    """

    text = re.sub(r"\r\n?", "\n", text)

    if not text.strip():
        return []

    # Ignore leading/trailing blank space, but preserve every
    # internal newline so repeated Enters can control pause length.
    text = text.strip()

    sections = re.split(r"(\n+)", text)
    chunks = []

    for section_index in range(0, len(sections), 2):
        paragraph = sections[section_index].strip()

        if not paragraph:
            continue

        newline_count = 0

        if section_index + 1 < len(sections):
            newline_count = len(
                sections[section_index + 1]
            )

        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z0-9"“‘\(])',
            paragraph
        )

        paragraph_chunks = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current:
                    paragraph_chunks.append(
                        current.strip()
                    )
                    current = ""

                paragraph_chunks.extend(
                    split_long_sentence(
                        sentence,
                        max_chars
                    )
                )

                continue

            candidate = (
                sentence
                if not current
                else current + " " + sentence
            )

            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    paragraph_chunks.append(
                        current.strip()
                    )

                current = sentence

        if current:
            paragraph_chunks.append(
                current.strip()
            )

        for index, part in enumerate(
            paragraph_chunks
        ):
            is_last_in_paragraph = (
                index == len(paragraph_chunks) - 1
            )

            breaks_after = (
                newline_count
                if is_last_in_paragraph
                else 0
            )

            chunks.append(
                (
                    part,
                    breaks_after
                )
            )

    return chunks


# ------------------------------------------------------------
# Load Chatterbox V3
# ------------------------------------------------------------

def load_model():
    print()
    print(
        f"Loading Chatterbox Multilingual V3 on {DEVICE}..."
    )

    model = (
        ChatterboxMultilingualTTS
        .from_pretrained(
            device=DEVICE,
            t3_model="v3"
        )
    )

    print("Multilingual V3 loaded.")
    print()

    return model


MODEL = load_model()

# Save the model's original voice conditioning so clearing
# a reference recording really returns to the default voice.
DEFAULT_CONDS = copy.deepcopy(
    MODEL.conds
)


# ------------------------------------------------------------
# Generate long-form narration
# ------------------------------------------------------------

def generate_longform(
    text,
    audio_prompt_path,
    language_id,
    sentence_pause_ms,
    paragraph_pause_ms,
    chunk_size,
    exaggeration,
    temperature,
    cfg_weight,
    seed_num,
    progress=gr.Progress()
):
    if not text or not text.strip():
        raise gr.Error(
            "Please enter some text."
        )

    if DEVICE != "cuda":
        print(
            "WARNING: CUDA is not active. "
            "Generation will run on CPU."
        )

    chunk_size = int(chunk_size)

    chunks = split_text(
        text,
        max_chars=chunk_size
    )

    if not chunks:
        raise gr.Error(
            "No text chunks were created."
        )

    used_seed = set_seed(
        int(seed_num)
    )

    print(
        f"Seed used: {used_seed}"
    )

    if not audio_prompt_path:
        MODEL.conds = copy.deepcopy(
            DEFAULT_CONDS
        )

    sample_rate = MODEL.sr

    sentence_pause_samples = int(
        sample_rate
        * (
            sentence_pause_ms
            / 1000.0
        )
    )

    paragraph_pause_samples = int(
        sample_rate
        * (
            paragraph_pause_ms
            / 1000.0
        )
    )

    sentence_silence = np.zeros(
        sentence_pause_samples,
        dtype=np.float32
    )

    paragraph_silence = np.zeros(
        paragraph_pause_samples,
        dtype=np.float32
    )

    generated_parts = []

    print()
    print(
        f"Generating {len(chunks)} chunks..."
    )
    print()

    for index, (
        chunk,
        paragraph_breaks_after
    ) in enumerate(chunks):

        progress(
            (index + 1) / len(chunks),
            desc=(
                f"Generating section "
                f"{index + 1} "
                f"of {len(chunks)}"
            )
        )

        print(
            f"[{index + 1}/"
            f"{len(chunks)}] "
            f"{len(chunk)} chars"
        )

        print(chunk)
        print()

        generate_kwargs = {
            "language_id": language_id,
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
        }

        if (
            audio_prompt_path
            and index == 0
        ):
            generate_kwargs[
                "audio_prompt_path"
            ] = audio_prompt_path

        spoken_chunk = apply_pronunciations(
            chunk
        )
        
        print(
            "Spoken as:",
            spoken_chunk
        )

        wav = MODEL.generate(
            spoken_chunk,
            **generate_kwargs
        )
        audio = (
            wav
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        generated_parts.append(
            audio
        )

        if index < len(chunks) - 1:
            if paragraph_breaks_after > 0:
                for _ in range(
                    paragraph_breaks_after
                ):
                    generated_parts.append(
                        paragraph_silence
                    )
            else:
                generated_parts.append(
                    sentence_silence
                )

    complete_audio = np.concatenate(
        generated_parts
    )

    output_file = (
        "longform-v3-output.wav"
    )

    sf.write(
        output_file,
        complete_audio,
        sample_rate
    )

    duration = (
        len(complete_audio)
        / sample_rate
    )

    status = (
        f"Finished: {len(chunks)} chunks | "
        f"{duration / 60:.1f} minutes | "
        f"{len(text):,} characters | "
        f"Language: {language_id}"
    )

    chunk_preview_text = ""

    for index, (
        chunk,
        paragraph_breaks_after
    ) in enumerate(
        chunks,
        start=1
    ):
        chunk_preview_text += (
            f"--- CHUNK {index} ---\n"
            f"{chunk}\n"
        )

        if paragraph_breaks_after > 0:
            chunk_preview_text += (
                f"[PARAGRAPH BREAK x"
                f"{paragraph_breaks_after}]\n"
            )

        chunk_preview_text += "\n"

    return (
        output_file,
        status,
        chunk_preview_text,
        str(used_seed)
    )


# ------------------------------------------------------------
# User interface
# ------------------------------------------------------------

def show_reference_audio_path(audio_path):
    if audio_path:
        return str(audio_path)

    return "No reference audio selected"

with gr.Blocks(
    title=(
        "Chatterbox Multilingual V3 "
        "Long-Form Narrator"
    )
) as demo:

    gr.HTML(
        """
<div class="app-panel app-header">
    <div class="app-title">
        Chatterbox Multilingual V3 — Long-Form Narrator
    </div>

    <div class="app-subtitle">
        Long-form multilingual voice cloning and narration using
        Chatterbox Multilingual V3.
    </div>

    <div class="app-instructions">
        Paste or type your narration below. Paragraphs that exceed the
        current chunk limit are highlighted in pink/red. Each press of
        <strong>Enter</strong> adds one paragraph pause, so additional
        blank lines create proportionally longer pauses.
        <br><br>
        The script is split into short sections automatically and
        combined into a single WAV file. Choose the required language
        under <strong>Voice &amp; Accent Settings</strong>.
        <br><br>
        <strong>Reference voice optional:</strong> upload or record a
        clean voice clip to clone a voice, or leave it empty to use the
        model's default conditioning.
    </div>
</div>
"""
    )

    with gr.Row():

        with gr.Column(scale=2):
            with gr.Group(                elem_classes=["app-panel"]
            ):
                gr.HTML(
                    """
<div class="section-title">
    Narration script
</div>
<div class="section-help">
    Type or paste narration here. Pink/red text indicates a paragraph
    that exceeds the current chunk-character limit.
</div>
"""
                )

                gr.HTML(
                    """
<div
    id="narration-editor"
    contenteditable="plaintext-only"
    spellcheck="true"
></div>

<div
    id="narration-editor-status"
>
    0 characters
</div>
"""
                )

                # Hidden transport textbox: JavaScript keeps this
                # synchronised with the custom narration editor.
                text = gr.Textbox(
                    value="",
                    lines=1,
                    elem_id=(
                        "narration-script-hidden"
                    ),
                    container=False,
                    interactive=True
                )

            reference_audio = gr.Audio(
                sources=[
                    "upload",
                    "microphone"
                ],
                type="filepath",
                label=(
                    "Reference voice "
                    "(optional)"
                )
            )

            reference_audio_path_display = gr.Textbox(
                label="Reference audio file location",
                value="No reference audio selected",
                interactive=False
            )

            reference_audio.change(
                fn=show_reference_audio_path,
                inputs=reference_audio,
                outputs=reference_audio_path_display
            )

            generate_button = gr.Button(
                "Generate Long-Form Narration",
                variant="primary"
            )

        with gr.Column(scale=1):
            output_audio = gr.Audio(
                label="Completed narration",
                type="filepath"
            )

            status = gr.Textbox(
                label="Status",
                interactive=False
            )

            with gr.Accordion(
                "Voice & Accent Settings",
                open=True
            ):
                language_id = gr.Dropdown(
                    choices=[
                        "en",
                        "fr",
                        "de",
                        "es",
                        "it",
                        "pt",
                        "nl",
                        "pl",
                        "ru",
                        "zh",
                        "ja",
                        "ko",
                        "ar",
                        "da",
                        "fi",
                        "el",
                        "he",
                        "hi",
                        "ms",
                        "no",
                        "sv",
                        "sw",
                        "tr"
                    ],
                    value="en",
                    label="Language",
                    interactive=True
                )

                exaggeration = gr.Slider(
                    minimum=0.25,
                    maximum=2.0,
                    value=0.5,
                    step=0.05,
                    label="Exaggeration",
                    interactive=True
                )

                cfg_weight = gr.Slider(
                    minimum=0.2,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="CFG / Pace",
                    interactive=True
                )

                temperature = gr.Slider(
                    minimum=0.05,
                    maximum=5.0,
                    value=0.8,
                    step=0.05,
                    label="Temperature",
                    interactive=True
                )

            with gr.Accordion(
                "Long-Form Settings",
                open=True
            ):
                chunk_size = gr.Slider(
                    minimum=150,
                    maximum=300,
                    value=260,
                    step=10,
                    label=(
                        "Maximum characters per chunk"
                    ),
                    elem_id=(
                        "chunk-size-slider"
                    ),
                    interactive=True
                )

                sentence_pause_ms = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    value=150,
                    step=25,
                    label=(
                        "Pause between chunks "
                        "(milliseconds)"
                    ),
                    interactive=True
                )

                paragraph_pause_ms = gr.Slider(
                    minimum=0,
                    maximum=2000,
                    value=550,
                    step=50,
                    label=(
                        "Pause between paragraphs "
                        "(milliseconds)"
                    ),
                    interactive=True
                )

            with gr.Accordion(
                "Reproducibility",
                open=False
            ):
                seed_num = gr.Number(
                    value=0,
                    label=(
                        "Random seed "
                        "(0 = random)"
                    ),
                    interactive=True
                )

                last_seed_display = gr.Textbox(
                    label="Seed used for last generation",
                    value="No generation yet",
                    interactive=False
                )

    with gr.Accordion(
        "Show automatic text chunks",
        open=False
    ):
        chunk_preview = gr.Textbox(
            lines=18,
            interactive=False,
            label="Generated chunks"
        )

    generate_button.click(
        fn=generate_longform,
        inputs=[
            text,
            reference_audio,
            language_id,
            sentence_pause_ms,
            paragraph_pause_ms,
            chunk_size,
            exaggeration,
            temperature,
            cfg_weight,
            seed_num
        ],
        outputs=[
            output_audio,
            status,
            chunk_preview,
            last_seed_display
        ]
    )


# ------------------------------------------------------------
# Start application
# ------------------------------------------------------------

if __name__ == "__main__":
    print()
    print(
        "Starting Chatterbox Multilingual V3 "
        "Long-Form GUI..."
    )
    print()

    demo.queue(
        max_size=20,
        default_concurrency_limit=1
    ).launch(
        inbrowser=True,
        share=False,
        css=EDITOR_CSS,
        js=EDITOR_JS
    )