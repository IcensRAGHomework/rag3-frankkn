import datetime
import chromadb
import traceback
import pandas as pd
import time

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_file = "COA_OpenData.csv"

def generate_hw01():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )

    chroma_client = chromadb.PersistentClient(path = dbpath)

    collection = chroma_client.get_or_create_collection(
        name = "TRAVEL",
        metadata = {"hnsw:space": "cosine"},
        embedding_function = openai_ef
    )

    if collection.count() == 0:
        data = pandas.read_csv(csv_file)
        for index, row in data.iterrows():
            id = str(row["ID"])
            metadata = {
                "file_name": csv_file,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row["CreateDate"], '%Y-%m-%d').timestamp())
            }
            document = row["HostWords"]

            collection.add(ids = id, metadatas = metadata, documents = document)

    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    # 取得 collection
    collection = generate_hw01()

    # 進行相似度查詢
    query_results = collection.query(
        query_texts=[question],
        n_results=10, 
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}},
                {"date": {"$lte": int(end_date.timestamp())}},
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )
    # print(query_results)

    metadatas = query_results['metadatas'][0]
    distances = query_results['distances'][0]

    sorted_results = sorted(
        zip(metadatas, distances),
        key=lambda x: 1 - x[1],  # similarity = 1 - distance
        reverse=True
    )

    sorted_names = [metadata['name'] for metadata, _ in sorted_results]
    # print(sorted_names)

    filtered_results = [metadata['name'] for metadata, distance in sorted_results if (1 - distance) >= 0.8]

    return filtered_results
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    return collection

if __name__ == "__main__":
    # question = "What are the best travel destinations?"
    # collection = demo(question)
    # print("Collection successfully created/retrieved:", collection.name)

    generate_hw01()

    # question = "我想要找有關茶餐點的店家"
    # city = ["宜蘭縣", "新北市"]
    # store_type = ["美食"]
    # start_date = datetime.datetime(2024, 4, 1)
    # end_date = datetime.datetime(2024, 5, 1)
    
    # ans_list = generate_hw02(question, city, store_type, start_date, end_date)
    # print(ans_list)