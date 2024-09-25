import os
from pymilvus import MilvusClient
from docling_pdf_reader import DoclingPDFReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.core import StorageContext, VectorStoreIndex

# Constants
HF_EMBED_MODEL_ID = "dunzhang/stella_en_400M_v5"


def create_index(
    milvus_db_path="/home/lab/milvus_demo.db",
    collection_name="hybrid_pipeline",
    reload_docs=False,
    file_paths=("/new_data/aldo/rag/2q24-cfsu-1.pdf",),
):
    # Initialize components
    reload_docs = reload_docs or not os.path.exists(milvus_db_path)
    client = MilvusClient(milvus_db_path)
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
        uri=milvus_db_path,
        collection_name=collection_name,
        dim=len(embed_model.get_text_embedding("hi")),
        overwrite=reload_docs,
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
        enable_sparse=True,
        sparse_embedding_function=sparse_embedding,
    )

    if reload_docs:
        
        docs = reader.load_data(file_path=list(file_paths))
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

    return index
