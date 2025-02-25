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
            # 這些數據是查詢時進行相似度計算的核心。
            document = row.get("HostWords", "")
            document_id = str(row["ID"])  
            collection.add(ids=[document_id], documents=[document], metadatas=[metadata])
        
        return collection
    
    except Exception as e:
        print("Error in generate_hw01:", e)
        print(traceback.format_exc())
        return None
    
def generate_hw02(question, city, store_type, start_date, end_date):
    try:
        # 初始化 ChromaDB 客戶端
        chroma_client = chromadb.PersistentClient(path=dbpath)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=gpt_emb_config['api_key'],
            api_base=gpt_emb_config['api_base'],
            api_type=gpt_emb_config['openai_type'],
            api_version=gpt_emb_config['api_version'],
            deployment_id=gpt_emb_config['deployment_name']
        )

        # 取得 collection
        collection = chroma_client.get_or_create_collection(
            name="TRAVEL",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef
        )

        # 進行相似度查詢
        query_results = collection.query(
            query_texts=[question],
            n_results=10
        )

        results = []
        for i in range(len(query_results["ids"][0])):
            metadata = query_results["metadatas"][0][i]
            score = query_results["distances"][0][i]  # ChromaDB 回傳的是距離，需轉換為相似度
            similarity = 1 - score

            # 依條件過濾
            if similarity >= 0.80:
                if city and metadata["city"] not in city:
                    continue
                if store_type and metadata["type"] not in store_type:
                    continue
                entry_date = datetime.datetime.fromtimestamp(metadata["date"])
                if not (start_date <= entry_date <= end_date):
                    continue
                
                results.append((metadata["name"], similarity))
        
        # 根據相似度排序並回傳店家名稱列表
        results.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in results]
    
    except Exception as e:
        print("Error in generate_hw02:", e)
        return []
    
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
    print(ans_list)