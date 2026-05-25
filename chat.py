from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI()

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333/",
    collection_name="learning_rag",
    embedding=embeddings_model,
)

# user query 

user_query = input("Ask something: ")

# returns the relavant chunks from the vector db 
search_result = vector_db.similarity_search(query=user_query)


# print(search_result)

context = "\n\n\n".join(
    [
        f"Page Content: {result.page_content}\n"
        f"Page Number: {result.metadata['page_label']}\n"
        f"File Location: {result.metadata['source']}"
        for result in search_result
    ]
)


SYSTEM_PROMPT = f"""
    You're an helpful AI Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the user to open the right page number to know more.

    CONTEXT: 
        {context}
"""

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":user_query},
    ]
)

print(f"🤖: {response.choices[0].message.content}")