# Constants for GPT-4o
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_API_VERSION = ""
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"

# Azure AI Search Constants
AZURE_SEARCH_ENDPOINT = ""
AZURE_SEARCH_INDEX = ""
AZURE_SEARCH_ADMIN_KEY = ""

# Backwards-compatible deployment names used by ingest_pdf.py
# These can be changed to match your actual Azure deployment names
AZURE_OPENAI_VISION_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"

# Consolidated config object expected by ingest_pdf.py
azure_open = {
	"endpoint": AZURE_OPENAI_ENDPOINT,
	"api_key": AZURE_OPENAI_API_KEY,
	"api_version": AZURE_OPENAI_API_VERSION,
	"vision_deployment": AZURE_OPENAI_VISION_DEPLOYMENT,
	"embedding_deployment": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
}
