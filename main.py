# main.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Detailed system prompt for Shariah-compliant JSON responses
SYSTEM_PROMPT = """
You are an Islamic Finance accounting assistant. Your task is to analyze lease contract scenarios and produce Shariah compliant accounting entries using Islamic finance rules and AAOIFI standards. You are connected to a vector database containing all AAOIFI standards. Your job: When given a user input describing a lease, extract and compute all relevant data. Perform every calculation for the extracted variables. If the contract is Ijarah Muntahia Bittamleek (Ijarah MBT): Compute the right of use asset as total acquisition cost (purchase plus import tax plus freight) minus the promised purchase price using AAOIFI FAS 23 paragraph 31(c). Compute the retained benefit as expected residual value minus promised purchase price. Calculate the net amortizable amount as right of use asset minus retained benefit. Apply straight line amortization over the Ijarah term using AAOIFI FAS 23 paragraphs 29 and 30. Explain each step and cite the relevant paragraph numbers from FAS 23. For any other Islamic finance operation, draw on the relevant AAOIFI FAS as needed, while remembering that only FAS 23 governs Ijarah MBT amortization. Rules: Use precise Islamic finance terminology and never use conventional finance terms such as interest or discount rate. Provide the final response in valid, well formatted JSON with quoted keys and values and without trailing commas. Do not invent values or make assumptions; if data is missing, state exactly what is missing and stop. If the context lacks relevant AAOIFI guidance, reply “Hmm, I’m not sure.”



Using the provided context, answer the user's question to the best of your ability using the resources provided.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure" and stop after that. Refuse to answer any question not about the info. Never break character.
"""

def build_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    # fetch_k: how many candidates to fetch; k: how many to return
    return vectorstore.as_retriever(search_kwargs={"fetch_k":30, "k":10})

def build_qa_chain(retriever, llm):
    # Prompt for chunk-level QA
    question_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(
            "Context: {context}\n\nQuestion: {question}\n\nAnswer in JSON:"
        ),
    ])
    # Prompt for combining chunk-level JSON fragments
    combine_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a JSON combiner. Combine the following partial JSON answers into a single valid JSON object without extra commas."
        ),
        HumanMessagePromptTemplate.from_template("{summaries}"),
    ])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt,
        },
    )

def build_agent(qa_chain, llm):
    tools = [
        Tool(
            name="aaofi_qa",
            func=lambda q: qa_chain.invoke({"query": q})["result"],
            description="Answer strictly from AAOIFI/Shariah PDFs; if not found, reply 'Hmm, I’m not sure.'"
        ),
    ]
    return initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )

if __name__ == "__main__":
    retriever = build_retriever()
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.4,
        frequency_penalty=0.2,
        presence_penalty=0,
        max_tokens=1500,
    )
    qa_chain = build_qa_chain(retriever, llm)
    agent = build_agent(qa_chain, llm)

    print("\n✅ Agent ready! Type a scenario and press Enter (or 'exit' to quit).\n")
    while True:
        user_input = input("📝 Scenario> ")
        if user_input.strip().lower().startswith("exit"):
            break
        response = agent.invoke(user_input)
        print(f"\n{response}\n{'─'*80}\n")