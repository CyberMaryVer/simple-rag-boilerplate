import os
import pandas as pd
import streamlit as st
import openai
import re
# from langchain_community.llms import OpenAI
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# from langchain.docstore.document import Document
# from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
# from dotenv import load_dotenv
#
# load_dotenv('.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
INDEX_PATH = os.getenv("INDEX_PATH", st.secrets["INDEX_PATH"])
MODEL = os.getenv("MODEL", st.secrets["MODEL"])

openai.api_key = OPENAI_API_KEY


def get_data(csv):
    data = pd.read_csv(csv, index_col=0)
    return data


def get_faiss_index(index_path=None):
    index_path = index_path if index_path else INDEX_PATH
    return FAISS.load_local(folder_path=index_path,
                            embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
                            allow_dangerous_deserialization=True
                            )


def remove_weird_tags(text):
    cleaned_text = re.sub(r'[\xa0]', ' ', text)
    return cleaned_text


def extract_sources(context):
    metadata = [s.metadata['source'] for s in context]
    sources = []
    for s in metadata:
        if type(s) is list:
            for ss in s:
                if ss not in sources:
                    sources.append(ss)
        elif type(s) is str:
            if s not in sources:
                sources.append(s)
    return sources


def extract_metadata(context, metadata_key):
    metadata = [s.metadata[metadata_key] for s in context]
    extracted = []
    for m in metadata:
        if type(m) is list:
            for mm in m:
                if mm not in extracted:
                    extracted.append(mm)
        elif type(m) is str:
            if m not in extracted:
                extracted.append(m)
    return extracted


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    r = openai.embeddings.create(input=[text], model=model)
    return r.dict()['data'][0]['embedding']


def get_context(question, use_embedding=True, verbose=False, metadata_keys=None):
    search_index = get_faiss_index()
    if use_embedding:
        q_embedding = get_embedding(question)
        context = search_index.similarity_search_by_vector(q_embedding, k=4)
    else:
        context = search_index.similarity_search(question, k=4)

    metadata_keys = metadata_keys if metadata_keys else []
    metadata_dict = {k: extract_metadata(context, k) for k in metadata_keys}
    sources = extract_sources(context)
    context = [remove_weird_tags(d.page_content) for d in context if len(d.page_content) > 20]
    context = '\n\n###\n\n'.join(context)
    return context, sources, metadata_dict


def get_context_with_score(question, threshold=0.33, use_embedding=True, verbose=False, metadata_keys=None):
    search_index = get_faiss_index()
    if use_embedding:
        q_embedding = get_embedding(question)
        context = search_index.similarity_search_with_score_by_vector(q_embedding, k=4, score_threshold=threshold)
    else:
        context = search_index.similarity_search_with_score(question, k=4)

    ##############################################
    if verbose:
        for d, s in context:
            print("\033[093SCORE:", s, "\033[0m")
            print("CONTENT:", d.page_content)
            print("SOURCE:", d.metadata['source'])
            print("---")
    ##############################################

    docs = [d for d, s in context if s < threshold]
    metadata_keys = metadata_keys if metadata_keys else []
    metadata_dict = {k: extract_metadata(docs, k) for k in metadata_keys}
    sources = extract_sources(docs)
    content = [d.page_content for d in docs]
    context = '\n\n###\n\n'.join(content)
    return context, sources, metadata_dict


def answer_with_openai(question,
                       temperature=0.02,
                       language='ru',
                       context_threshold=0.33,
                       mkeys=None,
                       use_retry=True
                       ):
    context, sources, meta = get_context_with_score(question, threshold=context_threshold, metadata_keys=mkeys)
    # print(context)
    if context != "":
        answer = answer_with_context(question, context, temperature=temperature, language=language)
    else:
        answer = answer_without_context(question)
        sources = {"OpenAI": {"href": "https://chat.openai.com/", "name": "OpenAI GPT-3.5 Turbo w/o context"}}

    if use_retry and _check_if_answer_is_empty(answer):
        answer = answer_without_context(question)
        sources += ["OpenAI", ]

    print(f"\033[093mAnswer: {answer}\033[0m")
    print(f"\033[090mSources: {sources}\033[0m")

    return answer, sources, meta


def print_openai_answer(question, verbose=False, temperature=0.02, metadata_keys=None):
    answer, sources, meta = answer_with_openai(question,
                                               verbose=verbose,
                                               temperature=temperature,
                                               mkeys=metadata_keys)
    print("\033[095mОТВЕТ: \033[0m", answer)
    print("\n\033[095mИСТОЧНИКИ: \033[0m")
    for s in sources:
        print(s)
    print("\n\033[095mМЕТАДАННЫЕ: \033[0m")
    for k, v in meta.items():
        print(k, v)


def answer_with_context(
        question,
        context,
        model=MODEL,
        max_tokens=2400,
        temperature=0.02,
        language="ru",
):
    pretext = "You an expert in the field of Russian labour law. You are answering questions from a client."
    addition = "IN CASE THERE IS NO ANSWER: If context does not contain enough information for answer, " \
               "write 'Not enough information in the context'"
    task = "TASK: Answer the question using the context below. Give your best guess. Be specific and use bullet points"
    lang = ", return answer in Russian. " if language == "ru" else ". "
    info = f"{pretext}\n\n{addition}\n\n{task}{lang}\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
    messages = [{"role": "system", "content": info}]

    try:
        # Create a completions using the question and context
        completion = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return completion.dict()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("[ERROR-answer_with_context]", e)
        return ""


def answer_without_context(
        question,
        model=MODEL,
        max_tokens=2800,
):
    pretext = "You an expert in the field of business, finance, law, and HR. You are answering questions from a client."
    task = "Answer the question. Be specific and use bullet points, return answer in Russian\n\n"
    info = f"Context: {pretext}\n\n---\n\nQuestion: {question}\nAnswer:"
    messages = [{"role": "system", "content": f"{task}{info}"}]
    try:
        # Create a completions using the question and context
        completion = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return completion.dict()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("[ERROR-answer_without_context]", e)
        return ""


def format_answer_with_openai(
        answer,
        api_key,
        model="gpt-3.5-turbo",
        max_tokens=2800,
):
    # print("\033[091mFormatting answer with OpenAI\033[0m")
    openai.api_key = api_key
    pretext = "You are editing an answer from an expert."
    task = "Improve the text below and translate it to Russian. " \
           "Be specific and use bullet points '<br>🔸'. " \
           "Don't add any other text\n\n"
    info = f"Context: {pretext}\n\n---\n\nEnglish text: {answer}\nImproved Russian text:"
    messages = [{"role": "system", "content": f"{task}{info}"}]
    try:
        # Create a completions using the question and context
        completion = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return completion.dict()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("[ERROR-format_answer_with_openai]", e)
        return ""


def _check_if_answer_is_empty(answer):
    """Check if the answer is empty or contains key phrases, like 'not enough information' in EN or RU"""
    if len(answer) < 10:
        return True
    if "недостаточно информации" in answer.lower():
        return True
    if "not enough information" in answer.lower():
        return True
    return False


def get_ai_assistant_response(user_input, user_id=0, user_key="12345test", metadata_keys=None):
    metadata_keys = metadata_keys if metadata_keys else []
    answer, sources, meta = answer_with_openai(user_input, mkeys=metadata_keys)
    return {"answer": answer, "sources": sources, "meta": meta}


if __name__ == '__main__':
    import time

    print(OPENAI_API_KEY)
    get_context_with_score("Что надо и что не надо маркировать?", verbose=True)
    start = time.time()
    print_openai_answer("Что надо и что не надо маркировать?", verbose=False, metadata_keys=[])
    print("\033[093m", time.time() - start, "\033[0m")
