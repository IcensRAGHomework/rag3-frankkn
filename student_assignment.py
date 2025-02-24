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

def generate_hw01():
    csv_file = "COA_OpenData.csv"
    try:
        # 讀取 CSV 檔案
        df = pd.read_csv(csv_file)
        
        # 初始化 ChromaDB 客戶端
        chroma_client = chromadb.PersistentClient(path=dbpath)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=gpt_emb_config['api_key'],
            api_base=gpt_emb_config['api_base'],
            api_type=gpt_emb_config['openai_type'],
            api_version=gpt_emb_config['api_version'],
            deployment_id=gpt_emb_config['deployment_name']
        )
        
        # 建立或獲取 Collection
        collection = chroma_client.get_or_create_collection(
            name="TRAVEL",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef
        )
        
        # 插入資料到 Collection
        for _, row in df.iterrows():
            metadata = {
                "file_name": csv_file,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(time.mktime(datetime.datetime.strptime(row["CreateDate"], "%Y-%m-%d").timetuple()))
            }
            document = row.get("HostWords", "")  
            document_id = str(row["ID"])  
            collection.add(ids=[document_id], documents=[document], metadatas=[metadata])
        
        return collection
    
    except Exception as e:
        print("Error in generate_hw01:", e)
        print(traceback.format_exc())
        return None
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
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

    collection_hw01 = generate_hw01()
    if collection_hw01:
        print("generate_hw01() executed successfully. Collection:", collection_hw01.name)
    else:
        print("generate_hw01() failed.")