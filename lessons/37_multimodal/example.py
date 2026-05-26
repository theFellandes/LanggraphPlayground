"""Lesson 37 · Multimodal — runnable demos.

Demonstrates vision input to Claude / GPT-4o via LangChain, the
PDF-as-images extraction pattern, and an audio transcription smoke test.

A small placeholder image and PDF are generated programmatically so the
lesson runs without external sample files.

Run:
    uv run python -m lessons.37_multimodal.example
    uv run python -m lessons.37_multimodal.example --vision
    uv run python -m lessons.37_multimodal.example --pdf
    uv run python -m lessons.37_multimodal.example --audio
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

from shared import get_llm, settings
from shared.pretty import console, section

ART_DIR = settings.data_dir / "multimodal"
ART_DIR.mkdir(parents=True, exist_ok=True)


def _make_sample_image() -> Path:
    """Render a tiny synthetic 'receipt' as an image so the demo is self-contained."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        console.print("[yellow]Missing Pillow. Install: uv add pillow[/]")
        raise

    path = ART_DIR / "receipt.png"
    if path.exists():
        return path

    img = Image.new("RGB", (400, 280), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    lines = [
        "ACME COFFEE",
        "123 Market St",
        "2026-05-26  14:23",
        "",
        "Latte             $4.50",
        "Croissant         $3.20",
        "Tip               $1.50",
        "",
        "TOTAL             $9.20",
        "VISA ****1234     APPROVED",
    ]
    y = 12
    for line in lines:
        d.text((20, y), line, fill="black", font=font)
        y += 22
    img.save(path)
    return path


def _encode(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


# --- Demo 1 · vision ---------------------------------------------------------
def demo_vision() -> None:
    section("Vision · extract structured data from an image")
    from langchain_core.messages import HumanMessage

    img_path = _make_sample_image()
    console.print(f"Sample image: {img_path}")

    image_b64 = _encode(img_path)
    msg = HumanMessage(content=[
        {"type": "text",
         "text": ("Extract from this receipt as STRICT JSON: "
                  '{"merchant": str, "date": "YYYY-MM-DD", "total": float, '
                  '"items": [{"name": str, "price": float}]}. '
                  "Reply with JSON only, no commentary.")},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ])
    reply = get_llm().invoke([msg])
    text = reply.content if hasattr(reply, "content") else str(reply)
    console.print(text)


# --- Demo 2 · PDF-as-images --------------------------------------------------
def demo_pdf() -> None:
    section("PDF · render each page as an image, extract via VLM")
    try:
        from langchain_core.messages import HumanMessage
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        console.print("[yellow]Missing Pillow. Install: uv add pillow[/]")
        return

    # Synthesise a tiny "PDF" — for the demo we just render two pages as PNGs
    # (you'd normally use pdf2image to convert a real PDF).
    pages = []
    for i, body in enumerate(
        [
            "POLICY DOC — section 1\nPTO: 20 days per year\nRollover: max 5 days\n",
            "POLICY DOC — section 2\nRefund SLA: 7 business days\nCap: $100 auto-approve\n",
        ]
    ):
        img = Image.new("RGB", (400, 200), "white")
        d = ImageDraw.Draw(img)
        d.text((20, 20), body, fill="black")
        p = ART_DIR / f"policy_page_{i + 1}.png"
        img.save(p)
        pages.append(p)

    # Send each page to the VLM and combine.
    extracted = []
    for p in pages:
        msg = HumanMessage(content=[
            {"type": "text", "text": "Extract every policy rule on this page as a bullet list."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_encode(p)}"}},
        ])
        reply = get_llm().invoke([msg])
        text = reply.content if hasattr(reply, "content") else str(reply)
        extracted.append(text.strip())

    console.print("[bold]Combined extraction:[/]")
    for i, x in enumerate(extracted, 1):
        console.rule(f"[bold]page {i}[/]")
        console.print(x)


# --- Demo 3 · audio (smoke) --------------------------------------------------
def demo_audio() -> None:
    section("Audio · Whisper transcription (smoke)")
    try:
        import whisper
    except ImportError:
        console.print(
            "[dim]openai-whisper not installed. The shape would be:[/]\n\n"
            "    import whisper\n"
            "    model = whisper.load_model('base')\n"
            "    result = model.transcribe('audio.mp3')\n"
            "    print(result['text'])\n\n"
            "[dim]Hosted alternatives:[/] OpenAI Whisper API, Deepgram, AssemblyAI."
        )
        return

    console.print("[yellow]No sample audio bundled with the repo — skipping actual transcription.[/]")
    console.print("Provide a path to a .mp3 / .wav and re-run with --audio --path your.mp3.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--audio", action="store_true")
    args = parser.parse_args()

    selected = []
    if args.vision: selected.append(demo_vision)
    if args.pdf:    selected.append(demo_pdf)
    if args.audio:  selected.append(demo_audio)
    if not selected:
        selected = [demo_vision, demo_pdf, demo_audio]

    for fn in selected:
        try:
            fn()
        except Exception as e:
            console.print(f"[red]demo failed:[/] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
