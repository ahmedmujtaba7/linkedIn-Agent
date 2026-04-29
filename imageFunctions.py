import base64
import datetime
import json
import os
from pathlib import Path
from pyexpat import errors
import re
import time

import cloudinary
from google import genai
from google.genai import types

from config import DEFAULT_GEMINI_IMAGE_MODEL, GEMINI_API_KEY, GEMINI_IMAGE_FALLBACKS, ImageType


cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)


PROMPT_TEMPLATES = {
    ImageType.CODE_SNIPPET: """
Create a high-quality LinkedIn post image of a {language} code snippet.

Topic: {topic}
Code content: {content}

Requirements:
- Dark background (VS Code / GitHub dark theme style)
- Monospace font (JetBrains Mono or Fira Code style)
- Syntax highlighting with distinct colors per token type (keywords, strings, comments, functions)
- Line numbers on the left in muted gray
- Rounded corners, subtle window chrome at the top (macOS-style dots: red, yellow, green)
- File name shown in the top bar: {filename}
- Font size large enough to be clearly readable on LinkedIn
- Clean padding around the code block
- All text must be perfectly legible, accurate, and correctly formatted
- 1:1 square or 4:5 aspect ratio suitable for LinkedIn
""",
    ImageType.ARCHITECTURE_DIAGRAM: """
Create a high-quality LinkedIn post image of a software architecture diagram.

Topic: {topic}
Components to include: {content}

Requirements:
- Clean, modern flat design with a light or dark background
- Each component in a clearly labeled box/icon (use distinct shapes for services, databases, queues, APIs)
- Directional arrows showing data/request flow with brief labels
- Color-coded by layer: frontend (blue), backend (purple), database (green), external services (orange)
- All component names and labels must be sharp, legible, and correctly spelled
- Include a short title at the top
- Professional look suitable for a tech LinkedIn post
- 1:1 or 16:9 aspect ratio
""",
    ImageType.FLOW_CHART: """
Create a high-quality LinkedIn post image of a flow chart / process diagram.

Topic: {topic}
Steps/process: {content}

Requirements:
- Clean minimal design, white or very light gray background
- Standard flowchart shapes: rectangles for steps, diamonds for decisions, rounded rects for start/end
- Clear directional arrows with Yes/No labels on decision branches
- Each node has concise text, large enough to read at a glance
- Color-coded: start/end (green), process steps (blue), decisions (amber/yellow), outcomes (gray)
- Title at the top in bold
- All text perfectly legible and spelled correctly
- 1:1 square or 4:5 aspect ratio
""",
    ImageType.INFOGRAPHIC: """
Create a high-quality LinkedIn post infographic.

Topic: {topic}
Key points / data: {content}

Requirements:
- Bold headline at the top
- Clean modern layout with clear visual hierarchy
- Icons or simple illustrations to complement each point
- Consistent color palette (2-3 accent colors max)
- Each point clearly labeled with legible text
- Source or branding area at the bottom (optional watermark: {author})
- All text must be sharp, correctly spelled, and easy to read
- 4:5 aspect ratio (ideal for LinkedIn feed)
""",
}


IMAGE_TYPE_BY_VALUE = {t.value: t for t in ImageType}

def _normalize_gemini_model_id(name: str) -> str:
    n = (name or "").strip()
    if n.startswith("models/"):
        n = n[len("models/") :]
    return n



def gemini_image_models_to_try() -> list[str]:
    primary = _normalize_gemini_model_id(os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL))
    extra = os.getenv("GEMINI_IMAGE_MODEL_FALLBACKS", "")
    ordered: list[str] = []
    for m in [primary] + [ _normalize_gemini_model_id(x) for x in extra.split(",") if x.strip() ] + list(GEMINI_IMAGE_FALLBACKS):
        if m and m not in ordered:
            ordered.append(m)
    return ordered

def extract_image_bytes(response: types.GenerateContentResponse) -> bytes:
    """Pull first inline image from a google-genai generate_content response."""
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                raw = inline.data
                if isinstance(raw, str):
                    return base64.b64decode(raw)
                return raw
    raise ValueError(
        "No image returned by the model. Check prompt, model name, or quota. "
        f"Text (if any): {getattr(response, 'text', '')!r}"
    )


def generate_linkedin_image(
    image_type: ImageType,
    topic: str,
    content: str,
    output_dir: str = "./generated_images",
    language: str = "Python",
    filename: str = "snippet.py",
    author: str | None = None,
    api_key: str | None = None,
    custom_prompt_suffix: str = "",
) -> str:
    """
    Generate a LinkedIn-ready image with Gemini (all typography inside the image).
    Returns absolute path to the saved PNG.
    """
    resolved_key = api_key or GEMINI_API_KEY
    if not resolved_key:
        raise ValueError(
            "No Gemini API key. Set GEMINI_API_KEY or IMAGE_MODEL_API_KEY in the environment."
        )

    author = author or os.getenv("LINKEDIN_AUTHOR", "Your Name")

    template = PROMPT_TEMPLATES[image_type]
    prompt = template.format(
        topic=topic,
        content=content,
        language=language,
        filename=filename,
        author=author,
    ).strip()

    if custom_prompt_suffix:
        prompt += f"\n\nAdditional requirements:\n{custom_prompt_suffix}"

    client = genai.Client(api_key=resolved_key)
    models_to_try = gemini_image_models_to_try()
    last_error: Exception | None = None
    response = None
    for i, model_id in enumerate(models_to_try):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="4:5"),
                ),
            )
            if i > 0:
                print(f"[✓] Image model fallback succeeded: {model_id}")
            break
        except errors.ClientError as e:
            last_error = e
            if e.code == 404 and i + 1 < len(models_to_try):
                print(
                    f"⚠️  Model {model_id!r} not available for image generation (404). "
                    f"Trying next: {models_to_try[i + 1]!r}…"
                )
                time.sleep(0.5)
                continue
            raise RuntimeError(
                f"Gemini image API call failed ({model_id}): {e}. "
                "Set GEMINI_IMAGE_MODEL to an image-capable model from "
                "https://ai.google.dev/gemini-api/docs/models or list models via the API."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Gemini image API call failed ({model_id}): {e}") from e

    if response is None and last_error is not None:
        raise RuntimeError(f"Gemini image API call failed: {last_error}") from last_error

    image_data = extract_image_bytes(response)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\-]", "_", topic)[:40]
    filename_out = f"{image_type.value}_{safe_topic}_{timestamp}.png"
    save_path = out_dir / filename_out

    with open(save_path, "wb") as f:
        f.write(image_data if isinstance(image_data, bytes) else base64.b64decode(image_data))

    print(f"[✓] Image saved → {save_path.resolve()}")
    return str(save_path.resolve())


def get_image_decision(post: str, groq_client) -> dict | None:
    """
    Decide if an image is needed. If yes, return a JSON spec dict for Gemini templates
    (topic, content, image_type, etc.). All text is rendered inside the image by the model.
    """
    image_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a LinkedIn visual strategist. A separate image model will draw the post image; ALL headlines, labels, code, and bullets must appear inside that image (no empty abstract-only backgrounds).

                If this post does not need an image, return exactly: NO_IMAGE

                If an image helps, return ONLY valid JSON (no markdown fences) with:
                - "image_type": one of "code_snippet", "architecture_diagram", "flow_chart", "infographic"
                - "topic": short title/headline to show prominently in the image
                - "content": the main material to render IN the image (extract code from fenced blocks for code_snippet; for other types use concise bullets, component lists, or steps derived from the post)
                - "language": for code_snippet only, e.g. "Python", "JavaScript"
                - "filename": for code_snippet only, e.g. "api.ts", "handler.py"
                - "author": optional short watermark for infographic (e.g. name or handle)
                - "custom_prompt_suffix": optional extra style notes for the image model

                Choice guide:
                - code_snippet: post centers on a code example — include the real code (trim if very long but keep it correct).
                - architecture_diagram: services, databases, APIs, data flow.
                - flow_chart: pipelines, decisions, processes, algorithms.
                - infographic: tips, lists, takeaways, or conceptual overview without strict diagram semantics.

                Keep "content" under ~1200 characters if possible. No trailing commentary outside the JSON.""",
                            },
                            {
                                "role": "user",
                                "content": f"""Analyze this LinkedIn post. Return NO_IMAGE or the JSON spec as instructed.

                ```
                {post}
                ```""",
                            },
                        ],
                        temperature=0.3,
                    )

    result = image_response.choices[0].message.content.strip()

    if result.upper() == "NO_IMAGE" or re.match(r"^no_image\b", result, re.I):
        print("ℹ️  No image needed for this post.")
        return None

    spec = _parse_image_spec(result)
    if spec is None:
        return None

    print("🖼️  Image spec received — generating with Gemini (text in-image)...")
    print(f"🎨 Topic: {spec.get('topic', '')!r} | type: {spec.get('image_type', '')!r}")
    return spec


def generate_architecture_diagram(topic: str, components: str, **kwargs) -> str:
    return generate_linkedin_image(
        ImageType.ARCHITECTURE_DIAGRAM,
        topic=topic,
        content=components,
        **kwargs,
    )


def generate_flow_chart(topic: str, steps: str, **kwargs) -> str:
    return generate_linkedin_image(
        ImageType.FLOW_CHART,
        topic=topic,
        content=steps,
        **kwargs,
    )


def generate_infographic(topic: str, key_points: str, **kwargs) -> str:
    return generate_linkedin_image(
        ImageType.INFOGRAPHIC,
        topic=topic,
        content=key_points,
        **kwargs,
    )

def _parse_image_spec(raw: str) -> dict | None:
    t = raw.strip()
    if not t or t.upper() == "NO_IMAGE":
        return None
    if re.match(r"^no_image\b", t, re.I):
        return None
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```\s*$", "", t)
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        print("⚠️  Could not parse image spec as JSON; skipping image.")
        return None
    if data.get("no_image"):
        return None
    return data



def generate_image_from_spec(spec: dict) -> str:
    
    """Map Groq JSON spec to generate_linkedin_image and return local file path."""
    raw_type = (spec.get("image_type") or "infographic").strip().lower()
    image_type = IMAGE_TYPE_BY_VALUE.get(raw_type, ImageType.INFOGRAPHIC)

    topic = (spec.get("topic") or "LinkedIn post").strip()
    
    # Handle content that might be a list or string
    content_raw = spec.get("content") or topic
    if isinstance(content_raw, list):
        content = "\n".join(str(item).strip() for item in content_raw)
    else:
        content = str(content_raw).strip()
    
    language = (spec.get("language") or "Python").strip()
    filename = (spec.get("filename") or f"snippet.{language.lower()[:3]}").strip()
    author = spec.get("author") or os.getenv("LINKEDIN_AUTHOR", "Your Name")
    suffix = (spec.get("custom_prompt_suffix") or "").strip()

    return generate_linkedin_image(
        image_type=image_type,
        topic=topic,
        content=content,
        language=language,
        filename=filename,
        author=author,
        custom_prompt_suffix=suffix,
    )
