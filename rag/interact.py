import sys
import argparse
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Here is some context:

{context}

---

Answer the following question based on only the above context: {question}
"""

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    return query_text

def query_db_for_content(query_text, db):
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Error: Question is out of scope of context.")
        sys.exit()
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text, results

def model_response(context_text, query_text, results):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

def main():
    query_text = cli() 
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    context_text,results = query_db_for_content(query_text,db)
    print(model_response(context_text,query_text,results))


if __name__ == "__main__":
    main()