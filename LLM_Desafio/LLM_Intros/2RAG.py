# rag_rag_modern.py
# Modern RAG: Ollama (llama3) + LangChain create_retrieval_chain + Chroma

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS

# new-style chain imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# 1) Load & split docs
# -------------------------
loader = DirectoryLoader("documents", glob="*.txt", loader_cls=TextLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs)
print(f"Split to {len(docs)} chunks")

# -------------------------
# 2) Create embeddings + vectorstore
# -------------------------
embeddings = OllamaEmbeddings(model="llama3")  # embeddings via Ollama

vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# 3) Build the combine-docs chain (prompt + llm)
# -------------------------
llm = OllamaLLM(model="llama3")  # uses Ollama for generation

# IMPORTANT: the prompt must expect a {context} variable (create_stuff_documents_chain will inject it)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the provided context to answer the question. "
                   "If the answer isn't contained in the context, say you don't know."),
        ("user", "{input}\n\nContext:\n{context}")
    ]
)

combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# -------------------------
# 4) Create retrieval chain 
# -------------------------
retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

# -------------------------
# 5) Interactive loop (invoke)
# -------------------------
print("\nRAG ready (modern chain). Type 'exit' to quit.\n")
while True:
    q = input("You: ").strip()
    if q.lower() in ("exit", "quit"):
        break

    # invoke the retrieval chain; it returns a dict (usually contains at least 'answer' and 'context')
    out = retrieval_chain.invoke({"input": q})
    # prefer common keys across LangChain versions
    answer = out.get("answer") or out.get("result") or out.get("output") or out.get("text")
    print("\nAnswer:\n", answer, "\n")

    # optional: show which docs were used
    context_docs = out.get("context") or out.get("documents") or []
    if context_docs:
        print("--- Sources (top retrieved) ---")
        for d in context_docs[:5]:
            src = d.metadata.get("source", "<no-source>")
            preview = (d.page_content[:200] + "...") if len(d.page_content) > 200 else d.page_content
            print(f"- {src}\n  {preview}\n")

