import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="Aurora Books Chatbot API")

print("Loading documents...")

loader = DirectoryLoader('./docs/', glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()

if len(documents) == 0:
    raise RuntimeError("No documents found in docs folder")

print(f"Loaded {len(documents)} document(s).")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunk(s).")

for chunk in chunks:
    if "source" not in chunk.metadata:
        chunk.metadata["source"] = "unknown"
    else:
        chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])

print("Creating vector database...")
embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-english-v3.0"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created successfully.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """You are the virtual assistant of Aurora Books, an online bookstore based in Italy.
Answer the user's question based ONLY on the following context.
If the answer is not in the context, say you don't know and invite the user to contact support at aurorabooks dot it.

IMPORTANT CITATION RULES.
After every piece of information you provide, you must cite the source document.
Use the format Source followed by the filename in square brackets.
If you cannot cite a source for a statement, do not include that statement.
Never invent or guess a source name.

Be friendly, professional and helpful. Respond in the same language the user writes in.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def format_docs_with_sources(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Risponde a una domanda usando il sistema RAG."""
    risposta = rag_chain.invoke(request.question)
    return {"answer": risposta}

@app.get("/health")
async def health_check():
    return {"status": "ok"}