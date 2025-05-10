# lease_workflow.py

"""
A LangGraph workflow for Challenge 4: a Shariah-compliance assistant for Ijarah MBT accounting.
This graph-based flow:
 1. Retrieves relevant AAOIFI/Shariah snippets
 2. Calculates the present value (PV) of lease payments
 3. Drafts the journal entry
 4. Validates the drafted entry against the PDFs

Usage:
  pip install langchain-graph langchain-community langchain-openai
  python lease_workflow.py
"""

from langgraph import Graph
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import subprocess

# 1) Setup retriever & QA chain
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "faiss_index", embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10, "fetch_k": 20, "search_type": "mmr"}
)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a Shariah compliance assistant. Use **only** the information in the Context below.
If the answer is not contained there, reply EXACTLY: I don’t know based on the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt},
)

# 2) Define the LangGraph workflow
graph = Graph(name="IjarahMBT_Accounting")

# Node: Retrieve relevant standards
retrieve_node = graph.add_node(
    name="RetrieveStandards",
    description="Fetch relevant AAOIFI/Shariah snippets",
    func=lambda scenario: qa_chain({"query": scenario})["result"],
)

# Node: Calculate PV of lease payments
def calculate_pv(payments, rate):
    return sum(p / (1 + rate) ** (i + 1) for i, p in enumerate(payments))

pv_node = graph.add_node(
    name="CalculatePV",
    description="Compute present value of lease payments",
    func=lambda scenario: calculate_pv([300_000, 300_000], 0.05),
)

# Node: Draft journal entry
draft_node = graph.add_node(
    name="DraftEntry",
    description="Draft the journal entry using standards & PV",
    func=lambda inputs: llm.predict(
        f"Using the following standard guidance: {inputs['RetrieveStandards']}\n" \
        f"and PV of lease payments: {inputs['CalculatePV']}, draft the journal entry."
    ),
)

# Node: Validate the draft
validate_node = graph.add_node(
    name="ValidateEntry",
    description="Validate the drafted entry against the PDFs",
    func=lambda entry: qa_chain({"query": f"Validate this entry: {entry}"})["result"],
)

# 3) Wire up the graph edges
graph.add_edge("RetrieveStandards", "CalculatePV")
graph.add_edge("RetrieveStandards", "DraftEntry")
graph.add_edge("CalculatePV", "DraftEntry")
graph.add_edge("DraftEntry", "ValidateEntry")

# 4) Execute the workflow
def main():
    scenario = input("Enter your Ijarah MBT scenario: ")
    results = graph.run(scenario)
    final = results.get("ValidateEntry")
    print("\n✅ Final Validated Entry:\n", final)

if __name__ == "__main__":
    main()
