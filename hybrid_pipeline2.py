# Imports
import asyncio
from dataclasses import dataclass, field
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
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import chain
from langchain.prompts.prompt import PromptTemplate as LangChainPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn



# Constants
# HF_EMBED_MODEL_ID = "BAAI/bge-m3"
HF_EMBED_MODEL_ID = "dunzhang/stella_en_400M_v5"


os.environ["ANTHROPIC_API_KEY"] = "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"
HF_API_KEY = "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"

# Check if milvus_demo.db exists
MILVUS_DB_PATH = "/home/lab/milvus_demo.db"
RELOAD_DOCS = False
RELOAD_DOCS = not os.path.exists(MILVUS_DB_PATH) or RELOAD_DOCS

# PDFs to load
FILE_PATHS = ["/new_data/aldo/rag/2q24-cfsu-1.pdf",]

# Initialize components
client = MilvusClient("/home/lab/milvus_demo.db")
reader = DoclingPDFReader(parse_type=DoclingPDFReader.ParseType.MARKDOWN)
node_parser = MarkdownNodeParser()
embed_model = HuggingFaceEmbedding(model_name=HF_EMBED_MODEL_ID, 
                                   trust_remote_code=True,
                                   query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery:",
                                   text_instruction="Instruct: Retrieve semantically similar text.\nQuery:",
                                #    parallel_process=True,
                                #    model_kwargs={"device_map": "auto"},
                                   )

# Vector store setup
sparse_embedding = BGEM3SparseEmbeddingFunction()
vector_store = MilvusVectorStore(
    uri="/home/lab/milvus_demo.db",
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
    # from IPython import embed; embed(header="check docs")
    # print(docs)
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

@dataclass
class IBMPrompt:
    prompt: LangChainPromptTemplate = field(default_factory=lambda: LangChainPromptTemplate.from_template(
        "<|system|>\n"
        "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless, and you follow ethical guidelines and promote positive behavior.\n"
        "<|user|>\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, answer the query.\n"
        "Query: {query_str}\n"
        "Answer:\n"
        "<|assistant|>\n"
    ))
    def __call__(self, input: StringPromptValue):
        return self.prompt.invoke(input=input.text)

@app.post("/get_response")
async def get_response(query: Query):
    LLM = query.llm
    query_str = query.query_str
    if LLM == "openai":
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            timeout=600,
        )
        llm = LangChainLLM(llm)
    elif LLM == "claude":
        llm = Anthropic(
            model="claude-3-5-sonnet-20240620",
        )
    elif LLM == "granite":
        llm = OpenAI(
                openai_api_base=f"http://localhost:8000/v1",
                # openai_api_base=f"https://cf47-52-117-121-50.ngrok-free.app/v1",
                model="/new_data/experiments/ss-bnp-p10/hf_format/samples_2795520",
                temperature=0.0,
                timeout=600,
            )
        # llm = chain(IBMPrompt()) | llm
        llm = LangChainLLM(llm)
    else:
        raise ValueError(f"LLM {LLM} not supported")

    # Query engine setup
    text_qa_template = (
        "<|system|>\n"
        "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless, and you follow ethical guidelines and promote positive behavior.\n"
        "<|user|>\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, answer the query.\n"
        "Query: {query_str}\n"
        "Answer:\n"
        "<|assistant|>\n"
    )
    refine_template = (
        "<|system|>\n"
        "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless, and you follow ethical guidelines and promote positive behavior.\n"
        "<|user|>\n"
        "The original query is as follows: {query_str}\n"
        "We have provided an existing answer: {existing_answer}\n"
        "We have the opportunity to refine the existing answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the query. "
        "If the context isn't useful, return the original answer.\n"
        "Refined Answer:\n"
        "<|assistant|>\n"
)
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        node_postprocessors=[reranker],
        text_qa_template=PromptTemplate(text_qa_template),
        refine_template=PromptTemplate(refine_template),
        vector_store_query_mode=vector_store_query_mode,
        response_mode=ResponseMode.REFINE,
    )

    # Example query
    query_res = await query_engine.aquery(query_str)
    print(f"\033[92mResponse from {LLM}:\033[0m\n{query_res}")
    return JSONResponse(content={"response": query_res.response})

async def main():
    tasks = [
        get_response("claude"),
        get_response("granite"),
        get_response("openai")
    ]
    return await asyncio.gather(*tasks)
    


if __name__ == "__main__":
    # asyncio.run(main())
    # print("\033[92mClaude Response Full Context:\033[0m")
    # print("According to the financial statements provided, BNP Paribas Group's net income attributable to equity holders for the first half of 2024 was 6,498 million euros, compared to 7,245 million euros for the first half of 2023.")
    # from IPython import embed; embed(header="select LLM between 'claude' and 'mixtral' and run get_response(LLM)")
    uvicorn.run(app, host="0.0.0.0", port=8001)

# Optional: Test LLM
# response = llm.complete("[INST] tell me a joke about bears [/INST] ")
# print(str(response))