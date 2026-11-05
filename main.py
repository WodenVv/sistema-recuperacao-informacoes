import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
# from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings

print("Ambiente configurado! Carregando PDF...")

# 1 carregamento do documento PDF
pdf_path = 'data/PPC_TSI_EaD.pdf'
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# 2 divisão do texto em chunks (uma maior divisão exige mais carga computacional da sua máquina, por isso o tamanho atual)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_documents(pages)
print(f"Quantidade total de chunks gerados: {len(texts)}")

# 3 criação do banco de dados vetorial com FAISS e ollama embeddings
print("Gerando embeddings (mxbai-embed-large)...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# testando
print("Testando geração de um embedding...")

db = FAISS.from_documents(texts, embeddings)

# 4 Configuração do LLM e do Pipeline de RetrievalQA
print("Configurando RetrievalQA com Llama 3.2...")
llm = OllamaLLM(model="llama3.2:latest")
retriever = db.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    chain_type="stuff"
)

# 5 Testando o Sistema
query = "Este documento possui sumário?"
print(f"Pergunta de teste: '{query}'\n")
response = qa_chain.invoke(query)

print("--- RESPOSTA DO SISTEMA ---")
print(response['result'])