# import torch
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from tqdm import tqdm

# Initialize GPT-4o model
llm = OpenAI(model_name="gpt-4o", temperature=0.5)

# Adjust the prompt template for handling personal information
prompt_template = """
    You are a helpful personal-information assistant.
    Provide gentle, informative, and respectful answers based on available data.
    Current year is 2025.

    Context: {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(template=prompt_template)

# Create LLMChain for generating questions
question_generator = LLMChain(
    llm=llm,
    prompt=CONDENSE_QUESTION_PROMPT,
    verbose=False
)

# Create LLMChain for generating answers
doc_chain = load_qa_chain(
    llm=llm,
    chain_type='stuff',
    prompt=PROMPT,
    verbose=False
)

# Create Conversation Memory
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

# Create Embedding Model
# model_name = 'hkunlp/instructor-base'
# model_name = 'Xenova/text-embedding-ada-002'
model_name = 'hkunlp/instructor-large'
embedding_model = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs={"device": 'cpu'}
)

vector_path = './vector-store'
db_file_name = 'nlp_stanford'
os.makedirs(vector_path, exist_ok=True)

# Load PDF documents
pdf_folder = './pdf/'
pdf_files = glob.glob(f"{pdf_folder}/*.pdf")

documents = []
for pdf_file in pdf_files:
    loader = PyMuPDFLoader(pdf_file)
    documents.extend(loader.load())

# Split the documents into smaller chunks for vectorization
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 100
)

doc = text_splitter.split_documents(documents)

# Create Vector DB
vectordb = FAISS.from_documents(
    documents = doc,
    embedding = embedding_model
)

# Save the Vector DB
vectordb.save_local(
    folder_path = os.path.join(vector_path, db_file_name),
    index_name = 'nlp'
)

# Load the Vector DB
vectordb = FAISS.load_local(
    folder_path=os.path.join(vector_path, db_file_name),
    embeddings=embedding_model,
    index_name='nlp'
)

# Create Retriever
retriever = vectordb.as_retriever()

# Create Conversational Retrieval Chain using GPT-4o as the Generator
chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=False,
    get_chat_history=lambda h: h
)

# Temporary alias for backward compatibility to suppress warnings
if not hasattr(chain, "invoke"):
    chain.invoke = chain.__call__

def make_source_docs_serializable(source_docs):
    """
    Process the source_documents list to extract only serializable metadata.
    Adjust the keys below as needed.
    """
    serializable_docs = []
    for doc in source_docs:
        if hasattr(doc, "metadata"): # If the doc has metadata, extract only simple types (like strings or numbers)
            metadata = doc.metadata
            serializable_docs.append({
                "source": metadata.get("source", "Unknown Document"),
                "page": metadata.get("page", "Unknown Page")
            })
        else:
            serializable_docs.append(str(doc))
    return serializable_docs

def ask(prompt_question):
    
    answer = chain.invoke({"question": prompt_question}) # Use chain.invoke to generate the answer to suppress warnings
    answer_text = answer['answer'].strip().rstrip('\n') # Clean up the answer text
    answer_text = answer_text.replace('\n', ' ') # Remove newline
    answer_text = ' '.join(answer_text.split()) # Remove double whitespaces
    serializable_sources = make_source_docs_serializable(answer.get("source_documents", [])) # Convert the source_documents to only include serializable metadata
    return {
        "answer": answer_text,
        "source_documents": serializable_sources
    }

if __name__ == '__main__':
    questions = [
        "How old are you?",
        "What is your highest level of education?",
        "What major or field of study did you pursue during your education?",
        "How many years of work experience do you have?",
        "What type of work or industry have you been involved in?",
        "Can you describe your current role or job responsibilities?",
        "What are your core beliefs regarding the role of technology in shaping society?",
        "How do you think cultural values should influence technological advancements?",
        "As a master's student, what is the most challenging aspect of your studies so far?",
        "What specific research interests or academic goals do you hope to achieve during your time as a master's student?"
    ]
    
    os.system('cls' if os.name == 'nt' else 'clear') # Clear screen
    qa_list = []
    for prompt_question in tqdm(questions, desc="Processing questions"):
        result = ask(prompt_question)
        qa_list.append({
            "question": prompt_question,
            "answer": result["answer"],
            "source_documents": result["source_documents"]
        })
    
    os.system('cls' if os.name == 'nt' else 'clear') # Clear screen
    print(json.dumps(qa_list, indent=2)) # Print Q&A in JSON format