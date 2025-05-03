"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:
you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.
Each llm definition has unique temperature value relevant to the specific class.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables from .env file
load_dotenv()

class AgentDecisionConfig:  # Fixed typo in class name
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=os.getenv("model_name", "gpt-4o-mini"),
            api_key=os.getenv("openai_api_key"),
            temperature=0.1  # Deterministic
        )

class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=os.getenv("model_name", "gpt-4o-mini"),
            api_key=os.getenv("openai_api_key"),
            temperature=0.7  # Creative but factual
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=os.getenv("model_name", "gpt-4o-mini"),
            api_key=os.getenv("openai_api_key"),
            temperature=0.3  # Slightly creative but factual
        )
        self.context_limit = 20  # Include last 20 messages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1536
        self.distance_metric = "Cosine"
        self.use_local = True
        self.local_path = "./data/qdrant_db"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.processed_docs_dir = "./data/processed"

        # Initialize OpenAI Embeddings
        self.embedding_model = OpenAIEmbeddings(
            model=os.getenv("embedding_model_name", "text-embedding-3-small"),
            api_key=os.getenv("embedding_openai_api_key", os.getenv("openai_api_key"))
        )

        self.llm = ChatOpenAI(
            model_name=os.getenv("model_name", "gpt-4o-mini"),
            api_key=os.getenv("openai_api_key"),
            temperature=0.3  # Slightly creative but factual
        )

        self.top_k = 5
        self.similarity_threshold = 0.75
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.chunking_strategy = "hybrid"  # Options: semantic, sliding_window, recursive, hybrid
        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 5
        self.max_context_length = 8192
        self.response_format_instructions = """Instructions:
        1. Answer the query based ONLY on the information provided in the context.
        2. If the context doesn't contain relevant information, state: "I don't have enough information."
        3. Do not use prior knowledge not contained in the context.
        4. Be concise and accurate.
        5. Provide a well-structured response based on retrieved knowledge."""
        self.include_sources = True
        self.metrics_save_path = "./logs/rag_metrics.json"
        self.min_retrieval_confidence = 0.8
        self.context_limit = 20  # Include last 20 messages (10 Q&A pairs) in history

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = ChatOpenAI(
            model_name=os.getenv("model_name", "gpt-4o-mini"),
            api_key=os.getenv("openai_api_key"),
            temperature=0.1  # Keep deterministic for classification tasks
        )

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # MB

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID (Rachel)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisionConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 40  # Storing 20 sets of QnA in history
