# rag_rag_modern.py
# Modern RAG: Ollama (llama3) + LangChain create_retrieval_chain + FAISS + LangSmith + Session Memory

#from dotenv import load_dotenv
#load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS

from langsmith import traceable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# -------------------------
# 1) Load & split docs
# -------------------------
@traceable
def load_and_split_documents(path="documents", glob="*.txt", chunk_size=500, chunk_overlap=50):
    loader = DirectoryLoader(path, glob=glob, loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(docs)
    print(f"Split to {len(docs)} chunks")

    return docs


# -------------------------
# 2) Create retriever
# -------------------------
@traceable
def create_retriever(docs, k=3):
    embeddings = OllamaEmbeddings(model="llama3")  # embeddings via Ollama
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


# -------------------------
# 3) Build retrieval chain with memory support
# -------------------------
@traceable
def build_chain(retriever):
    llm = OllamaLLM(model="llama3")

    # Now we include {history} in the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the conversation history and provided context "
                   "to answer the question. If the answer isn't in either, say you don't know."),
        ("user", "Conversation so far:\n{history}\n\n"
                 "Question: {input}\n\n"
                 "Context:\n{context}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)


# -------------------------
# 4) Ask a question (with memory buffer)
# -------------------------
@traceable(name="ask_question")
def ask_question(chain, query: str, history: list, history_window: int = 5):
    
    # prepare history string (last N turns)
    clipped_history = history[-history_window:]
    history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in clipped_history]) or "<empty>"

    # run chain with history included
    out = chain.invoke({"input": query, "history": history_text})
    answer = out.get("answer") or out.get("result") or out.get("output") or out.get("text")

    print("\nAnswer:\n", answer, "\n")

    # update memory buffer
    history.append((query, answer))



# -------------------------
# Main interactive loop
# -------------------------
if __name__ == "__main__":
    docs = load_and_split_documents()
    retriever = create_retriever(docs)
    chain = build_chain(retriever)

    conversation_history = []  # in-memory buffer for session only

    print("\nRAG with session memory ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ask_question(chain, q, conversation_history)
