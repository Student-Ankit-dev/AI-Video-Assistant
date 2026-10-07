import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)

DEBUG = False


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    if DEBUG:
        print("\n===== Retrieved Chunks =====")

        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:300])

        print("============================\n")

    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(transcript: str):

    print("Creating Vector Store...")

    vector_store = build_vector_store(transcript)

    # Retrieve more context from transcript
    retriever = get_retriever(
        vector_store,
        k=8
    )

    llm = get_llm()


    prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant helping users understand a video transcript.

Answer the user's question using the transcript context.

Rules:
- Use the provided context first.
- If the answer is partially available, explain it using the context.
- If the transcript does not contain the answer, clearly say:
  "This topic was not discussed in the transcript, but based on general knowledge..."

Context:
{context}


Question:
{input}


Answer:
"""
)


    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


    return rag_chain



def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(
        vector_store,
        k=8
    )


    llm = get_llm()


    prompt = ChatPromptTemplate.from_template(
"""
You are an expert meeting assistant.

Answer the question using only the transcript context.

If the answer is not present, say:
"I could not find this information in the meeting transcript."

Context:
{context}


Question:
{input}


Answer:
"""
)


    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


    return rag_chain



def ask_question(rag_chain, question: str) -> str:

    # print(f"\nQuestion: {question}")

    answer = rag_chain.invoke(question)

    # print(f"\nAnswer: {answer}")

    return answer