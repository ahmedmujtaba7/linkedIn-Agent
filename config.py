import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
# Gemini image API key (prefer GEMINI_API_KEY; IMAGE_MODEL_API_KEY kept for older .env files)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("IMAGE_MODEL_API_KEY")
# Default: model that supports generateContent + image output on the Gemini API (see SDK tests).
# Optional GEMINI_IMAGE_MODEL_FALLBACKS: comma-separated extra IDs to try after 404 (e.g. preview aliases).
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
GEMINI_IMAGE_FALLBACKS = (
    "gemini-2.5-flash-image",
    # "gemini-2.5-flash-image",
    # "gemini-2.5-flash-preview-image",
    # "gemini-3.1-flash-image-preview",
)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
LINKEDIN_ACCESS_TOKEN = os.environ["LINKEDIN_ACCESS_TOKEN"]


class ImageType(Enum):
    CODE_SNIPPET = "code_snippet"
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    FLOW_CHART = "flow_chart"
    INFOGRAPHIC = "infographic"