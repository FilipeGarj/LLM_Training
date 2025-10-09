from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate

import requests
from bs4 import BeautifulSoup
import os
import re

# 1. Load & split documents
loader = DirectoryLoader("documents", glob="*.txt", loader_cls=TextLoader)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs)

# 2. Build FAISS retriever
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Wrap retriever as tool
def document_retriever(query: str) -> str:
    results = retriever.get_relevant_documents(query)
    if not results:
        return "No docs found"
    return "\n\n".join([f"[{r.metadata.get('source','?')}] {r.page_content[:200]}" for r in results])

 

def search_cve(query: str) -> str:
    url = "https://duckduckgo.com/html/"
    site = "https://euvd.enisa.europa.eu/"
    params = {"q": f"site:{site} {query}"}
    r = requests.get(url, params=params)

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for a in soup.select(".result__a")[:5]:
        title = a.text.strip()
        link = a["href"]
        results.append(f"{title} - {link}")

    if not results:
        return f"No results found on {"https://euvd.enisa.europa.eu/"} for '{query}'."
    return "\n".join(results)



def save_to_db(answer_text: str, name_hint: str = "response") -> str:
    """Saves agent output to a uniquely named file and embeds it into the vectorstore."""
    
    # Sanitize filename
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name_hint.lower())
    filename = f"documents/{safe_name}.txt"
    
    # Ensure the directory exists
    os.makedirs("documents", exist_ok=True)
    
    # Save the file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(answer_text)
    
    # Load and embed only this new file
    new_doc = TextLoader(filename).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    new_doc = splitter.split_documents(new_doc)
    
    # Add to existing vectorstore
    vectorstore.add_documents(new_doc)
    
    return f"Saved and indexed new document: {filename}"

tools = {
    "save_to_db": save_to_db,
    "document_retriever": document_retriever,
    "search_cve": search_cve
}
# 4. ReAct prompt
template = """You are a helpful assistant. You can use tools, but only the ones listed.

Available tools:
- document_retriever: Search local documents in the knowledge base. 
   Use this first whenever the question might relate to something you've already seen or stored.
- search_cve: Search the web (within ENISA CVE database) using DuckDuckGo. 
   Only use this if local documents do NOT contain relevant info.
- save_to_db: Save the final answer into a local database (Chroma) for future retrieval.

Rules:
- ALWAYS use document_retriever first to check local memory before calling search_cve.
- If the local documents information about the subject is brief or not detailed enough you should use other tools
- If document_retriever returns 'No docs found', then try search_cve.
- Do not say you used a different tool to evade other rules
- Do not invent tools.
- If the tool returns 'No docs found', admit you don't know.
- If after 2 tool calls you still don't find useful context, stop and say you don't know.
- If you used search_cve tool, you must explicitly call the save_to_db tool using the following format before giving your Final Answer:
Action: save_to_db
Action Input: [the text or answer you want to save]
Observation: [result of saving]

Do not just say “Save_to_db:” — you must use the full Action format so the tool is executed.



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
