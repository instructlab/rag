# Imports
import asyncio
import os

from fastapi.responses import JSONResponse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
from docling_pdf_reader import DoclingPDFReader
from pymilvus import MilvusClient
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.llms.langchain import LangChainLLM
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Constants
# HF_EMBED_MODEL_ID = "BAAI/bge-m3"
HF_EMBED_MODEL_ID = "dunzhang/stella_en_400M_v5"

# Check if milvus_demo.db exists
MILVUS_DB_PATH = os.path.expanduser("~/milvus_demo.db")
RELOAD_DOCS = False
RELOAD_DOCS = not os.path.exists(MILVUS_DB_PATH) or RELOAD_DOCS

# PDFs to load
PDF_PATH = os.environ.get(
    "PDF_PATH", os.path.join(os.path.dirname(__file__), "2q24-cfsu-1.pdf")
)
FILE_PATHS = [PDF_PATH]

# Initialize components
client = MilvusClient(MILVUS_DB_PATH)
reader = DoclingPDFReader(parse_type=DoclingPDFReader.ParseType.MARKDOWN)
node_parser = MarkdownNodeParser()
embed_model = HuggingFaceEmbedding(
    model_name=HF_EMBED_MODEL_ID,
    trust_remote_code=True,
    query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery:",
    text_instruction="Instruct: Retrieve semantically similar text.\nQuery:",
)

# Vector store setup
sparse_embedding = BGEM3SparseEmbeddingFunction()
vector_store = MilvusVectorStore(
    uri=MILVUS_DB_PATH,
    collection_name="quackling_hybrid_pipeline",
    dim=len(embed_model.get_text_embedding("hi")),
    overwrite=RELOAD_DOCS,
    hybrid_ranker="RRFRanker",
    hybrid_ranker_params={"k": 60},
    enable_sparse=True,
    sparse_embedding_function=sparse_embedding,
)
vector_store_query_mode = VectorStoreQueryMode.HYBRID

# Reranker setup
reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=10)

# Load and index documents
if RELOAD_DOCS:
    docs = reader.load_data(file_path=FILE_PATHS)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=docs,
        embed_model=embed_model,
        storage_context=storage_context,
        transformations=[node_parser],
    )
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

app = FastAPI()


class Query(BaseModel):
    llm: str
    query_str: str


error_response = "Error: keys not configured for this model"


@app.post("/get_response")
async def get_response(query: Query):
    LLM = query.llm
    query_str = query.query_str
    if LLM == "openai":
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.0,
                timeout=600,
            )
            llm = LangChainLLM(llm)
        else:
            return JSONResponse(content={"response": error_response})
    elif LLM == "claude":
        if os.getenv("ANTHROPIC_API_KEY"):
            llm = Anthropic(
                model="claude-3-5-sonnet-20240620",
            )
        else:
            return JSONResponse(content={"response": error_response})
    elif LLM == "granite":
        llm = ChatOpenAI(
            openai_api_base=f"http://localhost:8000/v1",
            model=os.getenv("MODEL_PATH"),
            temperature=0.0,
            timeout=600,
        )
        llm = LangChainLLM(llm)
    else:
        raise ValueError(f"LLM {LLM} not supported")

    # Query engine setup
    text_qa_template = """Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information, answer the query.
    Query: {query_str}
    Answer:
    """
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        node_postprocessors=[reranker],
        text_qa_template=PromptTemplate(text_qa_template),
        vector_store_query_mode=vector_store_query_mode,
        response_mode=ResponseMode.REFINE,
    )

    # Example query
    query_res = await query_engine.aquery(query_str)
    print(f"\033[92mResponse from {LLM}:\033[0m\n{query_res}")
    return JSONResponse(content={"response": query_res.response})


async def main():
    tasks = [get_response("claude"), get_response("granite"), get_response("openai")]
    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
