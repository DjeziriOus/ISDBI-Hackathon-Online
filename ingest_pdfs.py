from pathlib import Path

# ← community loader
from langchain_community.document_loaders import PyPDFLoader
# ← community text splitter stays the same
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ← community vectorstore
from langchain_community.vectorstores import FAISS
# ← new standalone OpenAI embeddings package
from langchain_openai import OpenAIEmbeddings

# 1. Gather all your PDFs
pdf_folder = Path("shariah_pdfs")
pdf_files  = list(pdf_folder.glob("*.pdf"))

# 2. Load & split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_chunks = []
for pdf in pdf_files:
    loader = PyPDFLoader(str(pdf))
    docs   = loader.load()
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["source"] = pdf.name
    all_chunks.extend(chunks)

# 3. Embed & index
embeddings  = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_chunks, embeddings)

# 4. Save for later
vectorstore.save_local("faiss_index")

print(f"Indexed {len(all_chunks)} chunks from {len(pdf_files)} PDFs.")