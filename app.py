import streamlit as st
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import config

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Azure AI Search Chat",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Knowledge Base Chat")
st.caption("Ask questions grounded in your documents")

st.markdown("### 💡 Suggested Questions")

suggested_questions = [
    "Provide summary on Swiggy Annual Report 2023-24",
    "Compare financial data between 2023-24"
]

cols = st.columns(len(suggested_questions))
for i, question in enumerate(suggested_questions):
    if cols[i].button(question):
        st.session_state.suggested_input = question


with st.sidebar:
    st.markdown("## 👋 Hi There")
    st.markdown(
        """
        Welcome to your **Knowledge Base Assistant**.

        Ask questions and get answers grounded
        strictly in your documents.

        This demo uses:
        - Azure OpenAI for embeddings.
        - Azure AI Search for vector retrieval
        - Streamlit for the UI
        - Swiggy Financial Report 2023-24 as sample data
        """
    )

    st.divider()

    st.markdown("### ⚙️ Controls")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Powered by Azure OpenAI & Azure AI Search")

# ------------------------------------------------------------
# Initialize clients (cached)
# ------------------------------------------------------------
@st.cache_resource
def init_clients():
    openai_client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT
    )

    search_client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=config.AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(config.AZURE_SEARCH_ADMIN_KEY)
    )

    return openai_client, search_client


openai_client, search_client = init_clients()

# ------------------------------------------------------------
# Helper: create query embedding
# ------------------------------------------------------------
def embed_query(text: str):
    response = openai_client.embeddings.create(
        model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding


# ------------------------------------------------------------
# Retrieve relevant documents from Azure AI Search
# ------------------------------------------------------------
def retrieve_documents(query: str, top_k: int = 5):
    vector = embed_query(query)

    results = search_client.search(
        search_text=None,
        vector_queries=[
            {
                "kind": "vector",
                "vector": vector,
                "fields": "contentVector",
                "k": top_k
            }
        ],
        select=["content", "doc", "page", "content_type"]
    )

    docs = []
    for r in results:
        docs.append({
            "content": r["content"],
            "doc": r["doc"],
            "page": r["page"],
            "content_type": r["content_type"]
        })

    return docs

# ------------------------------------------------------------
# Generate grounded answer
# ------------------------------------------------------------
def generate_answer_stream(question: str, contexts: list):
    context_text = "\n\n".join(
        f"[Document: {c['doc']} | Page: {c['page']} | Type: {c['content_type']}]\n{c['content']}"
        for c in contexts
    )

    system_prompt = """
You are a professional assistant answering questions strictly
based on the provided context.

RULES:
- Use ONLY the given context
- Do NOT add external knowledge
- If the answer is not present, Politely say you don't know about this question instead of making something up"
- Cite document name and page numbers when relevant
- Try to display numbers in tabular format if it helps readability
- Keep the answer concise and to the point and user ask to give in depth provide detailed answer
"""

    stream = openai_client.chat.completions.create(
        model=config.AZURE_OPENAI_CHAT_DEPLOYMENT
        if hasattr(config, "AZURE_OPENAI_CHAT_DEPLOYMENT")
        else config.AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
CONTEXT:
{context_text}

QUESTION:
{question}
"""
            }
        ],
        temperature=0,
        stream=True,
        max_tokens=16384
    )

    return stream

# ------------------------------------------------------------
# Streamlit Chat UI
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question about your documents...")

if "suggested_input" in st.session_state:
    user_input = st.session_state.pop("suggested_input")

if user_input:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve context
    with st.spinner("🔍 Searching knowledge base..."):
        contexts = retrieve_documents(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream = generate_answer_stream(user_input, contexts)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_response += token
                    placeholder.markdown(full_response)

    # ✅ Store final assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )