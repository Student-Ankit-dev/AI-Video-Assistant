# #Actionableitems , decision , questions 

# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# import os 


# def get_llm():
#     return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)



# def build_chain(system_prompt : str):
#     llm = get_llm()
#     return (
#         RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human","{text}"),
#     ]) | llm |StrOutputParser()
#     )

# def extract_action_items(transcript:str)->str:
#     chain = build_chain(
#          "You are an expert meeting analyst. From the meeting transcript, "
#         "extract all action items. For each provide:\n"
#         "- Task description\n"
#         "- Owner (who is responsible)\n"
#         "- Deadline (if mentioned, else write 'Not specified')\n\n"
#         "Format as a numbered list. If none found say 'No action items found.'"
#     )

#     return chain.invoke(transcript)


# def extract_key_decisions(transcript: str) -> str:
#     chain = build_chain(
#         "You are an expert meeting analyst. From the meeting transcript, "
#         "extract all key decisions made. Format as a numbered list. "
#         "If none found say 'No key decisions found.'"
#     )
#     return chain.invoke(transcript)


# def extract_questions(transcript: str) -> str:
#     chain = build_chain(
#         "From the meeting transcript, extract all unresolved questions "
#         "or topics needing follow-up. Format as a numbered list. "
#         "If none found say 'No open questions found.'"
#     )
#     return chain.invoke(transcript)



from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import time
from groq import RateLimitError

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def safe_invoke(chain, prompt):
    while True:
        try:
            response = chain.invoke(prompt)
            time.sleep(1)
            return response

        except RateLimitError:
            print("⚠️ Groq rate limit reached. Waiting 5 seconds...")
            time.sleep(5)



def build_chain(system_prompt):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}")
        ]
    )

    return prompt | llm | StrOutputParser()



def prepare_transcript(transcript):

    # Limit size sent to Groq
    if len(transcript) > 12000:
        transcript = transcript[:12000]

    return transcript



def extract_action_items(transcript: str):

    chain = build_chain(
        """
        You are an expert meeting analyst.

        Extract action items from the transcript.

        Format:

        1. Task:
           Owner:
           Deadline:

        If no action items exist return:
        No action items found.
        """
    )

    return safe_invoke(
        chain,
        {
            "text": prepare_transcript(transcript)
        }
    )



def extract_key_decisions(transcript: str):

    chain = build_chain(
        """
        You are an expert meeting analyst.

        Extract important decisions from this transcript.

        Return numbered points.

        If none exist:
        No key decisions found.
        """
    )

    return safe_invoke(
        chain,
        {
            "text": prepare_transcript(transcript)
        }
    )



def extract_questions(transcript: str):

    chain = build_chain(
        """
        You are an expert meeting analyst.

        Extract unresolved questions and follow-up topics.

        Return numbered points.

        If none exist:
        No open questions found.
        """
    )


    return safe_invoke(
        chain,
        {
            "text": prepare_transcript(transcript)
        }
    )