import nest_asyncio

from llama_index.core import SummaryIndex
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings, SimpleDirectoryReader, PromptTemplate
from llama_index.core import StorageContext, ServiceContext

from llama_index.graph_stores.neo4j import Neo4jGraphStore

from llama_index.llms.ollama import Ollama

#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

## to use Qdrant, install it w/ poetry add llama-index-vector-stores-qdrant
## note: vectorstore not supporting python 3.13 yet, 
#from llama_index.vector_stores.qdrant import QdrantVectorStore
#import qdrant_client

from llama_index.vector_stores.redis import RedisVectorStore
from redis import Redis

from config import config

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nest_asyncio.apply()


# def create_qdrant_index(documents):
#     client = qdrant_client.QdrantClient(
#         host="localhost",
#         port=6333
#     )
#     vector_store = QdrantVectorStore(client=client, collection_name="qdrant_collection")
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     index = VectorStoreIndex.from_documents(
#         documents,
#         storage_context=storage_context,
#     )
#     return index
#
# def create_redis_index(documents):
#     redis_client = Redis(
#         host=config.REDIS_HOST,
#         port=config.REDIS_PORT,
#         password=config.REDIS_PASSWORD,
#         username=config.REDIS_USERNAME,
#         db=0
#     )
#     logger.info("Redis client created: %s", redis_client.ping())
#
#     # create the vector store wrapper
#     vector_store = RedisVectorStore(redis_client=redis_client, overwrite=True)
#
#     # load storage context
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#
#     # build and load index from documents and storage context
#     index = VectorStoreIndex.from_documents(
#         documents, storage_context=storage_context
#     )
#     # index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
#     return index


# setup llm & embedding model
llm=Ollama(model=config.OLLAMA_LLM_MODEL, base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}", 
           request_timeout=600.0)
Settings.llm = llm
# embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
embed_model = OllamaEmbedding(model_name=config.OLLAMA_EMBED_MODEL, 
                              base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}", 
                              trust_remote_code=True)
Settings.embed_model = embed_model

# load data
loader = SimpleDirectoryReader(
            input_dir = config.DOC_DIR,
            required_exts=[".pdf"],
            recursive=True
        )
docs = loader.load_data()

# # Creating a vector index over loaded data
# logger.info('Creating vector index')
# try:
#     index = create_redis_index(docs)
#     logger.info('Using Redis collection')
# except:
#     logger.warning("Failed to create Redis collection, using local collection")
#     index = VectorStoreIndex.from_documents(docs, show_progress=True)


# # ====== Customise prompt template ======
# qa_prompt_tmpl_str = (
# "Context information is below.\n"
# "---------------------\n"
# "{context_str}\n"
# "---------------------\n"
# "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
# "Query: {query_str}\n"
# "Answer: "
# )
# qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# query_engine.update_prompts(
#     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
# )
# response = query_engine.query(user_query,)


# Create the knowledge graph index
uri = config.NEO4J_URI
username = config.NEO4J_USERNAME
password = config.NEO4J_PASSWORD

from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex

vector_store = Neo4jVectorStore(
    username=username,
    password=password,
    url=uri,
    database="neo4j",
    timeout=600.0,
    embedding_dimension=1024
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# NOTE: only need once to build the KG, can take a while!
index = VectorStoreIndex.from_documents(
    documents=docs,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

vector_retriever = index.as_retriever(similarity_top_k=5)
#
# from llama_index.core.retrievers import RouterRetriever, RetrieverTool
# from llama_index.core.selectors import PydanticMultiSelector
#
# # Wrap each retriever in a tool with metadata
# kg_tool = RetrieverTool(
#     retriever=kg_retriever,
#     metadata={"name": "KGRetriever", "description": "Answers relationship-based queries about technologies."}
# )
#
# vector_tool = RetrieverTool(
#     retriever=vector_retriever,
#     metadata={"name": "VectorRetriever", "description": "Retrieves semantic matches from technical documents."}
# )
#
# # Define a selector (you can also use LLMBasedSelector)
# selector = PydanticMultiSelector.from_defaults()
#
# # Build RouterRetriever
# hybrid_retriever = RouterRetriever(
#     selector=selector,
#     retriever_tools=[kg_tool, vector_tool],
#     verbose=True
# )



from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=vector_retriever,
    embed_model=embed_model
)

# get user query
while (user_query := input("\n\nWhat do you want to know about these files?\n")):
    # Generate the response
    response = query_engine.query(user_query,)

    print(str(response))

    for node in response.source_nodes:
        print({
            "text": node.text,
            "score": node.score,
            "source": node.metadata.get("file_name", "unknown")
        })


