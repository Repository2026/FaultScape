# You can use LLama-3-70b or your local model
API_KEY = "Your_LLM_API_KEY"  # Replace with your actual API key
API_BASE = "The_API_base_URL" # Replace with the base URL of your LLM API

MODEL_NAME = "meta-llama/Llama-3-70b-chat-hf"
TOP_K = 10  # Number of documents to retrieve for reranking

# === Prompt template configuration ===
PROMPT_TEMPLATE_PATH = "./LLM_FL_Result/FL/pairwise_rerank/prompt_template.txt"