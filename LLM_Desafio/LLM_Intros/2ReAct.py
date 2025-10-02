from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate

# 1. Load & split documents
loader = DirectoryLoader("documents", glob="*.txt", loader_cls=TextLoader)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs)

# 2. Build FAISS retriever
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Wrap retriever as tool
def document_retriever(query: str) -> str:
    results = retriever.get_relevant_documents(query)
    if not results:
        return "No docs found"
    return "\n\n".join([f"[{r.metadata.get('source','?')}] {r.page_content[:200]}" for r in results])


tools = {"document_retriever": document_retriever}

# 4. ReAct prompt
template = """You are a helpful assistant. You can use tools, but only the ones listed.

Available tools:
- document_retriever: search local documents.

Rules:
- When using a tool, always write its name exactly as shown above (no extra words).
- Do not invent tools.
- If the tool returns 'No docs found', admit you don't know.
- If after 2 tool calls you still don't find useful context, stop and say you don't know.

Format:
Question: the user question
Thought: your reasoning
Action: the tool to use
Action Input: input for the tool
Observation: tool output
... (repeat as needed)
Final Answer: the final answer

Question: {input}
Thought:"""


prompt = PromptTemplate.from_template(template)

# 5. Run loop
llm = OllamaLLM(model="llama3")

def react_agent(query: str):
    scratchpad = ""
    for step in range(5):
        response = llm.invoke(prompt.format(input=query) + scratchpad)

        print(f"\n--- Step {step+1} ---")
        print(response)

        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()

        if "Action:" in response and "Action Input:" in response:
            action = response.split("Action:")[-1].split("\n")[0].strip()
            action_input = response.split("Action Input:")[-1].split("\n")[0].strip()

            obs = tools.get(action, lambda x: f"Unknown tool: {x}")(action_input)
            print(f"[Tool: {action}] Input: {action_input}\n[Observation] {obs[:200]}...\n")

            scratchpad += f"\n{response}\nObservation: {obs}\nThought:"
        else:
            return "Agent got stuck."
    return "Step limit reached."

# 2. Interactive loop
print("\nReAct Agent ready! Type 'exit' to quit.\n")
while True:
    q = input("You: ").strip()
    if q.lower() in ("exit", "quit"):
        print("Bye!")
        break

    answer = react_agent(q)
    print("\nAssistant:\n", answer, "\n")
    