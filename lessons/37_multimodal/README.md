# Lesson 37 · Multimodal AI

The frontier LLMs in 2026 are all multimodal. Claude, GPT-4o,
Gemini 2.5 — each ingests **images, audio, video, and PDFs natively**,
and the right multimodal model often replaces an entire pipeline of
OCR + parsing + classification. This lesson is how to wire that into a
LangChain / LangGraph app.

## What you'll learn

| Modality | What | Why |
|---|---|---|
| **Vision input** | Send images to Claude / GPT-4o / Gemini | Replaces OCR + layout parsing for receipts, screenshots, charts, PDFs |
| **VLM-based PDF understanding** | GPT-4o / Claude on each PDF page as an image | Beats text-extraction on tables, scanned docs, mixed content |
| **Image generation** | DALL-E 3, Stability, Flux | Product mockups, marketing, illustration |
| **Audio (input)** | Whisper, Deepgram, AssemblyAI | Transcription, diarisation |
| **Audio (output / TTS)** | ElevenLabs, OpenAI TTS, Cartesia | Voice agents, accessibility |
| **Video** | Gemini 1.5/2.5 (native), TwelveLabs | Sample-the-frames patterns elsewhere |
| **Multimodal RAG** | CLIP embeddings + vector store | Image + text retrieval |

## Part 1 · Vision input — the killer use case

The fastest way to feel the change: send a photo of a receipt to
Claude and ask "Extract the structured data." It works. No OCR, no
template, no regex. The same code on an architectural diagram, a
chart, a handwritten note.

### Claude vision via LangChain

```python
import base64
from langchain_core.messages import HumanMessage
from shared import get_llm

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()

image_b64 = encode_image("data/sample_images/receipt.jpg")

reply = get_llm().invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Extract: merchant, total, date, line items."},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ])
])
print(reply.content)
```

The `content` field is a **list** of blocks: text + image. Each block
is a separate token-cost factor. Image cost is per-tile (Anthropic
charges by image dimensions; OpenAI by detail level).

### GPT-4o vision (same shape)

```python
from langchain_openai import ChatOpenAI

reply = ChatOpenAI(model="gpt-4o").invoke([
    HumanMessage(content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}",
                       "detail": "high"}},  # low / high / auto
    ])
])
```

`detail` is OpenAI-specific: `low` = 85 tokens (256×256 thumbnail);
`high` = ~765 tokens at full resolution. Use `low` for "is there a
cat here?", `high` for "extract this table."

### Gemini vision via langchain-google-genai

```python
# pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
reply = llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "Describe this scene."},
    {"type": "image_url", "image_url": "https://example.com/photo.jpg"},
])])
```

Gemini supports **URL** input as well — it fetches the image
server-side. Saves you a base64 round-trip.

### Cost / quality matrix (rough, 2026 prices)

| Model | Image cost (1024×1024) | Best at |
|---|---|---|
| **Claude Sonnet 4.6** | ~$0.005 | Document understanding, charts, structured extraction |
| **GPT-4o** | ~$0.0036 (high detail) | General; fast |
| **GPT-4o-mini vision** | ~$0.0004 | High-volume classification (10× cheaper) |
| **Gemini 2.5 Pro** | ~$0.003 | Long video, large multi-image batches |
| **Gemini 2.5 Flash** | ~$0.0005 | Cheap multimodal classification |

For high-volume multimodal classification (millions of images),
**Gemini Flash and GPT-4o-mini are the workhorses**.

## Part 2 · VLM-based PDF understanding — the OCR replacement

Lesson 20 chunks PDFs with `pypdf` (text extraction). Two failure
modes:

1. **Scanned PDFs** — pypdf returns gibberish.
2. **Complex layouts** — multi-column papers, tables embedded in text, footnotes — pypdf flattens the order.

The 2024+ alternative: **render each page as an image, send to a
vision LLM**. Two patterns:

### Pattern A — Docling (IBM)

```python
# pip install docling
from docling.document_converter import DocumentConverter
result = DocumentConverter().convert("paper.pdf")
markdown = result.document.export_to_markdown()
```

Docling uses a vision model under the hood (DocLayNet) to detect
layout, then extracts text in reading order. Outputs clean markdown
with tables preserved. **Best general-purpose PDF parser in 2026.**

### Pattern B — direct GPT-4o / Claude on rendered pages

```python
# pip install pypdf2 pdf2image
from pdf2image import convert_from_path

pages = convert_from_path("paper.pdf", dpi=150)
for i, img in enumerate(pages):
    img.save(f"page_{i}.png")
    # ... send each page to Claude/GPT-4o vision ...
```

Slower per page, more flexible (you can ask for specific extractions —
"give me the equations" or "summarise the methodology section").

### When to use which

- **Bulk text extraction** → Docling
- **Targeted extraction with custom rubric** → direct VLM
- **Tables specifically** → Docling table mode, or `unstructured.io`'s table-transformer
- **Scanned legal documents** → AWS Textract or Azure Document Intelligence (specialised; better at forms)

## Part 3 · Audio — Whisper and beyond

### Speech-to-text with Whisper

```python
# pip install openai-whisper      # local model
import whisper
model = whisper.load_model("base")          # tiny / base / small / medium / large
result = model.transcribe("call.mp3")
print(result["text"])
```

Local Whisper runs on CPU (slowly) or GPU (fast). Outputs
transcription + timestamps. Multilingual.

Hosted alternatives:

```python
# OpenAI Whisper API
from openai import OpenAI
client = OpenAI()
with open("call.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(model="whisper-1", file=f)

# Deepgram (typically faster + more accurate on noisy audio, with diarisation)
# pip install deepgram-sdk
```

**Deepgram is the production default** for high-volume STT — speaker
diarisation, real-time streaming, lower latency than Whisper.

### Text-to-speech

```python
# OpenAI TTS
client.audio.speech.create(model="tts-1", voice="alloy", input="Hello, world.")

# ElevenLabs (better voices, voice cloning, more expensive)
# pip install elevenlabs
from elevenlabs.client import ElevenLabs
e = ElevenLabs(api_key="...")
audio = e.text_to_speech.convert(text="Hello, world.", voice_id="...")

# Cartesia (fastest; sub-100ms; great for real-time voice agents)
# pip install cartesia
```

**For voice agents (Vapi, Retell, etc.) the latency budget is brutal:**
STT must be ≤200ms, LLM ≤500ms, TTS ≤200ms = ~1s end-to-end. Cartesia
+ GPT-4o-mini + Deepgram is the current "make a voice agent feel
real-time" stack.

## Part 4 · Image generation

Quick survey:

| Model | API | Strengths |
|---|---|---|
| **DALL-E 3** | OpenAI | Best for following nuanced text prompts |
| **GPT-4o native** | OpenAI | In-conversation generation + editing |
| **Flux.1 (pro/dev/schnell)** | Replicate, fal.ai | OSS, photorealistic, fast |
| **Stable Diffusion XL** | Stability, Replicate | OSS, customisable |
| **Midjourney** | Discord / API | Highest aesthetic quality |
| **Imagen 3 / 4** | Google | Strong text rendering |

Quick recipe:

```python
from openai import OpenAI
img = OpenAI().images.generate(
    model="dall-e-3",
    prompt="A cat astronaut in low-earth orbit, photorealistic",
    size="1024x1024",
    quality="hd",
)
print(img.data[0].url)
```

In an agent: image generation is just another `@tool` that returns a
URL or a base64 string. The pattern is identical to any other API tool.

## Part 5 · Multimodal RAG

Standard RAG indexes text. Multimodal RAG indexes **text + image**
(and audio, in some setups), retrieving by visual similarity or
cross-modal matching.

### Pattern A — CLIP embeddings (joint text+image space)

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
clip = SentenceTransformer("clip-ViT-B-32")

# Embed images and text into the SAME 512-dim space
text_emb = clip.encode("a cat on a sofa")
image_emb = clip.encode(Image.open("photo.jpg"))

# Index both into Chroma; queries can be text or image
```

CLIP was trained contrastively on `(image, caption)` pairs. The
beautiful property: `text("red car")` and `image(red car)` end up
**close in the same vector space**. Text-to-image search just works.

### Pattern B — VLM-summary indexing

Cheaper and often better-quality: **generate a text description of
each image with a VLM, embed the description with a normal text
embedder**.

```python
descriptions = []
for img_path in paths:
    desc = vlm.describe(img_path)         # one-line description
    descriptions.append(desc)

# Now embed descriptions as text, index normally
store.add_texts(descriptions, metadatas=[{"path": p} for p in paths])
```

Trade-offs: VLM call is one-time (at index time), search is normal
text retrieval, results are interpretable. CLIP is faster to index
but its descriptions are "vibe-based" (no human-readable summary).

## Part 6 · Multimodal in LangGraph

The retriever, the tool, the message — every layer of LangGraph
already supports multimodal content. The state field just needs to
accommodate it.

```python
from typing import Annotated, TypedDict
from operator import add

class MultimodalState(TypedDict):
    messages: list                   # can contain image/audio blocks
    extracted_text: str
    extracted_data: dict
    confidence: float

async def vision_extract_node(state):
    """Send the last image in the message to a VLM and extract structured data."""
    last_msg = state["messages"][-1]
    image_blocks = [b for b in last_msg.content if b.get("type") == "image_url"]
    if not image_blocks:
        return {"extracted_text": "", "confidence": 0.0}

    extracted = await vision_llm.ainvoke([
        HumanMessage(content=[
            {"type": "text", "text": "Extract structured data as JSON."},
            *image_blocks,
        ])
    ])
    return {"extracted_text": extracted.content, "confidence": 0.9}
```

Combine with lesson 30's fan-out: one branch per image in a batch
upload, reduce into a combined result. This is the shape of "extract
data from 200 invoices in parallel."

## Run it

```bash
uv add pdf2image pillow         # for the demo
# Optional:
#   uv add openai-whisper        # local STT
#   uv add docling               # PDF + VLM extraction
#   uv add elevenlabs            # TTS

uv run python -m lessons.37_multimodal.example
uv run python -m lessons.37_multimodal.example --vision
uv run python -m lessons.37_multimodal.example --pdf
uv run python -m lessons.37_multimodal.example --audio
```

The script demonstrates:

1. **Vision**: send a sample image (provided) to Claude/GPT-4o; extract structured data with a Pydantic schema.
2. **PDF-to-images**: render a sample PDF to PNGs; send each page to a VLM and reconstruct a structured summary.
3. **Audio (smoke)**: if `openai-whisper` is installed, transcribe a sample WAV. Otherwise prints the API recipe.

## Anti-patterns

| Smell | Fix |
|---|---|
| OCR for everything by default | Try a VLM first — saves a pipeline of OCR + post-processing |
| `detail: "high"` on classification jobs | Use `low` for "is there X here?"; saves 10× tokens |
| Sending uncompressed images | Resize to max-dim 1568px (Claude) or 2048px (GPT-4o). Above that, no quality gain |
| Whisper for production high-volume STT | Deepgram is faster + more accurate at scale |
| `tts-1-hd` for chatbot responses | `tts-1` is fine; HD adds latency for marginal quality |
| CLIP for retrieval when descriptions would work better | VLM-summary indexing is often higher quality; CLIP is fast but vibe-based |
| Reusing one giant VLM call for an unknown number of images | Fan-out (lesson 30) with a semaphore (lesson 27) |
| Forgetting per-image cost in the budget | Image tokens dominate fast on multi-page PDFs |

## Pairs with

- **[Lesson 20 · Chunking + parsing](../20_chunking_and_parsing/README.md)** — the text-only side of document understanding
- **[Lesson 04 · Structured output](../04_structured_output/README.md)** — Pydantic schemas with image extraction
- **[Lesson 30 · Advanced graphs](../30_advanced_graphs/README.md)** — fan-out over a batch of images
- **[Lesson 36 · Library landscape](../36_library_landscape/README.md)** — Firecrawl / Jina Reader also do VLM-based extraction

## References

- [Anthropic vision docs](https://docs.anthropic.com/en/docs/build-with-claude/vision) — image-input shape + best practices
- [OpenAI vision docs](https://platform.openai.com/docs/guides/vision) — detail levels, cost
- [Google Gemini multimodal](https://ai.google.dev/gemini-api/docs/vision) — including video
- [Docling](https://github.com/DS4SD/docling) — IBM's PDF / VLM document understanding
- [unstructured.io](https://docs.unstructured.io/) — alternative document extractor
- [Whisper](https://github.com/openai/whisper) — speech-to-text, local
- [Deepgram docs](https://developers.deepgram.com/) — production STT
- [ElevenLabs docs](https://elevenlabs.io/docs) — production TTS
- [Cartesia](https://docs.cartesia.ai/) — low-latency TTS for voice agents
- [CLIP paper · Radford et al. 2021](https://arxiv.org/abs/2103.00020) — joint text+image embeddings
- [Voyage multimodal embed-3](https://docs.voyageai.com/docs/multimodal-embeddings) — production CLIP successor

## Next →

[Lesson 38 · Reasoning models + routing](../38_reasoning_and_routing/README.md) — when the "thinking" model is the right tool.
