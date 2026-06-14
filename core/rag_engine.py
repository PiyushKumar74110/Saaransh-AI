import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever



# LLM
def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )



# FORMAT DOCS
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



# CORE RAG (FIXED)
def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Use ONLY the context below:

{context}

If answer is not found, say:
"I could not find this information in the meeting transcript."

Be concise and precise."""
        ),
        ("human", "{question}"),
    ])

    # FIXED: no LCEL pipe confusion
    def get_context(inputs):
        question = inputs["question"]
        docs = retriever.invoke(question)
        return format_docs(docs)

    rag_chain = (
        {
            "context": RunnableLambda(get_context),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



# LOAD RAG (FIXED)

def load_rag_chain():

    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Use ONLY the context below:

{context}

If answer not found, say:
"I could not find this information in the meeting transcript."

Be concise."""
        ),
        ("human", "{question}"),
    ])

    def get_context(inputs):
        question = inputs["question"]
        docs = retriever.invoke(question)
        return format_docs(docs)

    rag_chain = (
        {
            "context": RunnableLambda(get_context),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



# ASK FUNCTION (FIXED)

def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")

    # ALWAYS pass dict
    answer = rag_chain.invoke({"question": question})

    print(f"Answer: {answer}")
    return answer