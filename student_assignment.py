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

    # 讀取 CSV 檔案
    df = pd.read_csv(csv_file)
    
    # 初始化 ChromaDB
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
        name="TRAVEL", # 這是此 Collection 的名稱，用於標識此數據集的用途。
        metadata={"hnsw:space": "cosine"}, # 這是設定查詢相似度計算的參數，cosine 表示使用餘弦相似度來進行距離計算。
        embedding_function=openai_ef
    )
    
    # 在初始化資料庫時，需從 CSV 檔案中提取每條記錄的相關欄位，並將其作為 Metadata 存入 ChromaDB
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
        
        # 文件數據（documents） 將 CSV 檔案中的 HostWords 欄位內容提取作為文本數據存入 ChromaDB
        # HostWords是查詢時進行相似度計算的核心。
        document = row.get("HostWords", "") # 如果HostWords是null，則設為"" (空字串)
        document_id = str(row["ID"])  
        collection.add(ids=[document_id], documents=[document], metadatas=[metadata])
    
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

    # collection_hw01 = generate_hw01()
    # if collection_hw01:
    #     print("generate_hw01() executed successfully. Collection:", collection_hw01.name)
    # else:
    #     print("generate_hw01() failed.")

    question = "我想要找有關茶餐點的店家"
    city = ["宜蘭縣", "新北市"]
    store_type = ["美食"]
    start_date = datetime.datetime(2024, 4, 1)
    end_date = datetime.datetime(2024, 5, 1)
    
    ans_list = generate_hw02(question, city, store_type, start_date, end_date)
    # print(ans_list)