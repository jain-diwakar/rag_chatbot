import os
import tempfile
import base64
import fitz  # PyMuPDF
from PIL import Image
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import config
import re

def safe_id(text: str) -> str:
    """
    Convert any string into a valid Azure AI Search document key.
    Allowed: letters, digits, _, -, =
    """
    return re.sub(r"[^a-zA-Z0-9_\-=]", "_", text)

# -------------------------------------------------------------------
# Safety checks
# -------------------------------------------------------------------
if not getattr(config, "AZURE_OPENAI_API_KEY", None):
    raise ValueError("AZURE_OPENAI_API_KEY not set in config.py")

if not getattr(config, "AZURE_SEARCH_ADMIN_KEY", None):
    raise ValueError("AZURE_SEARCH_ADMIN_KEY not set in config.py")


# -------------------------------------------------------------------
# Initialize Azure OpenAI Client
# -------------------------------------------------------------------
openai_client = AzureOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    api_version=config.AZURE_OPENAI_API_VERSION,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT
)


# -------------------------------------------------------------------
# Initialize Azure AI Search Client
# -------------------------------------------------------------------
search_client = SearchClient(
    endpoint=config.AZURE_SEARCH_ENDPOINT,
    index_name=config.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(config.AZURE_SEARCH_ADMIN_KEY)
)


# -------------------------------------------------------------------
# Helper: Encode image as Base64
# -------------------------------------------------------------------
def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


# -------------------------------------------------------------------
# Convert PDF → Images using PyMuPDF (Vision‑Safe)
# -------------------------------------------------------------------
def pdf_to_images(pdf_path: str, output_dir: str, zoom: float = 2.0):
    """
    zoom=1.2 ≈ 90 DPI
    Safe for Azure OpenAI Vision
    """
    doc = fitz.open(pdf_path)
    image_paths = []

    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image_path = os.path.join(output_dir, f"page_{page_index + 1}.png")
        pix.save(image_path)

        image_paths.append((page_index + 1, image_path))

    return image_paths


# -------------------------------------------------------------------
# Convert PNG → JPEG (reduces payload size dramatically)
# -------------------------------------------------------------------
def convert_to_jpeg(image_path: str, quality: int = 95) -> str:
    img = Image.open(image_path).convert("RGB")
    jpg_path = image_path.replace(".png", ".jpg")
    img.save(jpg_path, "JPEG", quality=quality)
    os.remove(image_path)
    return jpg_path


# -------------------------------------------------------------------
# GPT‑4o Vision → Extract FULL page details (GROUND TRUTH)
# -------------------------------------------------------------------
def extract_page_detail(image_path: str) -> str:
    prompt = """
You are analyzing a page from a financial or business PDF document.

STRICT RULES:
- Extract ALL visible information exactly as shown
- Preserve ALL numbers, percentages, and currency values EXACTLY
- Convert tables into clean markdown tables
- Describe charts and graphs with axis values, legends, and trends
- Do NOT infer, assume, or hallucinate missing information
- Output ONLY structured markdown (no explanations)
"""

    image_base64 = encode_image_base64(image_path)

    response = openai_client.chat.completions.create(
        model=config.AZURE_OPENAI_VISION_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You are a highly accurate document analysis engine."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------
# Generate PAGE SUMMARY (AGENTIC ACCELERATOR)
# -------------------------------------------------------------------
def generate_page_summary(detail_text: str) -> str:
    response = openai_client.chat.completions.create(
        model=config.AZURE_OPENAI_VISION_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": f"""
Summarize the following page in 3–5 concise bullet points.

RULES:
- Focus on key facts, numbers, and decisions
- Do NOT introduce new information
- Do NOT change numeric values
- Keep it short and high‑signal

PAGE CONTENT:
{detail_text}
"""
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------
# Create text embedding
# -------------------------------------------------------------------
def create_embedding(text: str):
    response = openai_client.embeddings.create(
        model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding


# -------------------------------------------------------------------
# Upload page detail + summary to Azure AI Search
# -------------------------------------------------------------------
def upload_page(
    doc_name: str,
    page_num: int,
    year: str,
    detail_text: str,
    summary_text: str,
    section: str = "Auto-detected"
):
    documents = []

    safe_doc_name = safe_id(doc_name)

    for content, content_type in [
        (detail_text, "page_detail"),
        (summary_text, "page_summary")
    ]:
        documents.append({
            "id": f"{safe_doc_name}_page_{page_num}_{content_type}",
            "content": content,
            "contentVector": create_embedding(content),
            "doc": doc_name,   # keep original filename here
            "page": page_num,
            "section": section,
            "year": year,
            "content_type": content_type
        })

    search_client.upload_documents(documents)
    print(f"✅ Stored page {page_num} ({doc_name})")

# -------------------------------------------------------------------
# MAIN INGESTION PIPELINE
# -------------------------------------------------------------------
def ingest_pdf(pdf_path: str, doc_name: str, year: str):
    print(f"📄 Starting ingestion for: {pdf_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        page_images = pdf_to_images(pdf_path, tmpdir)

        for page_num, image_path in page_images:
            print(f"🔍 Processing page {page_num}")

            try:
                image_path = convert_to_jpeg(image_path)

                size_kb = os.path.getsize(image_path) / 1024
                print(f"📐 Image size: {size_kb:.1f} KB")

                detail_text = extract_page_detail(image_path)
                summary_text = generate_page_summary(detail_text)

                upload_page(
                    doc_name=doc_name,
                    page_num=page_num,
                    year=year,
                    detail_text=detail_text,
                    summary_text=summary_text
                )

            except Exception as e:
                print(f"❌ Failed to process page {page_num}: {e}")

    print("✅ PDF ingestion completed successfully")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    files_dir = os.path.join(root_dir, "files")

    pdf_path = None
    doc_name = "Document"

    if os.path.isdir(files_dir):
        for f in os.listdir(files_dir):
            if f.lower().endswith(".pdf"):
                pdf_path = os.path.join(files_dir, f)
                doc_name = os.path.splitext(f)[0]
                break

    if not pdf_path:
        raise FileNotFoundError("No PDF found in files/ directory")

    ingest_pdf(
        pdf_path=pdf_path,
        doc_name=doc_name,
        year="FY24"
    )