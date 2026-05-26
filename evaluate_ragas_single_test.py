import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rag_update import RAGPipelineAdvanced

def calculate_ir_metrics(results, k=5):
    hits = 0
    mrr_sum = 0
    total = len(results)

    for res in results:
        ground_truth_id = str(res["reference_context_id"]).strip()
        retrieved_ids = res["retrieved_doc_ids"][:k]

        rank = 0
        for idx, doc_id in enumerate(retrieved_ids):
            if ground_truth_id in str(doc_id) or str(doc_id) in ground_truth_id:
                rank = idx + 1
                break

        if rank > 0:
            hits += 1
            mrr_sum += 1.0 / rank

    hit_rate = hits / total if total > 0 else 0
    mrr = mrr_sum / total if total > 0 else 0
    return hit_rate, mrr

def main():
    print("="*60)
    print(" BẮT ĐẦU TEST LUỒNG ĐÁNH GIÁ (FIX LỖI KIỂU DỮ LIỆU) ")
    print("="*60)

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("LỖI: Chưa có GOOGLE_API_KEY")
        return

    data_folder = "tailieu"
    try:
        df_dataset = pd.read_csv("test_dataset.csv")
    except FileNotFoundError:
        print("Không tìm thấy file 'test_dataset.csv'.")
        return
        
    df_single = df_dataset.head(1)
    if df_single.empty: return

    test_row = df_single.iloc[0]
    file_name = str(test_row['file_source']).strip()
    
    # [QUAN TRỌNG NHẤT]: ÉP KIỂU VÀ XÓA KÝ TỰ ẨN TỪ PANDAS
    question = str(test_row['question']).strip()
    ground_truth = str(test_row['ground_truth']).strip()
    ref_id = str(test_row['reference_context_id']).strip()
    
    full_path = os.path.join(data_folder, file_name)

    print(f"\n[1] Đang khởi tạo RAG cho file: {file_name}...")
    app = RAGPipelineAdvanced(input_file=full_path)
    
    print(f"\n[2] Đang truy vấn câu hỏi:")
    print(f"    Q: '{question}'")
    print(f"    (Type: {type(question)})") # Xác nhận kiểu dữ liệu
    
    ans, retrieved_context_str, ctx_ids = app.chat(
        user_query=question, 
        return_context=True,
        use_metadata_filter=True, 
        use_reranker=True         
    )
    
    print(f"\n[3] Hệ thống trả lời:")
    print(f"    A: {ans}")
    
    app.close_connection()

    ctx_texts = retrieved_context_str.split("\n---\n")
    all_ragas_data = {
        "question": [question],
        "answer": [ans],
        "contexts": [ctx_texts],
        "ground_truth": [ground_truth]
    }
    all_ir_results = [{"reference_context_id": ref_id, "retrieved_doc_ids": ctx_ids}]

    print("\n" + "-"*60)
    print(" CHẤM ĐIỂM MÔ-ĐUN TRUY XUẤT (QDRANT + RERANKER) ")
    print("-"*60)
    hit_rate, mrr = calculate_ir_metrics(all_ir_results, k=5)
    
    print(f" - Đáp án đúng nằm ở ID: {ref_id}")
    print(f" - ID hệ thống tìm được: {ctx_ids}")
    print(f" => Hit Rate@5 : {hit_rate} (1.0 = Trúng, 0.0 = Trượt)")
    print(f" => MRR@5      : {mrr:.4f} (1.0 = Top 1, 0.5 = Top 2...)")

    print("\n" + "-"*60)
    print(" CHẤM ĐIỂM MÔ-ĐUN TẠO SINH (LLM-AS-A-JUDGE BẰNG GEMINI) ")
    print("-"*60)
    
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    dataset = Dataset.from_dict(all_ragas_data)
    
    try:
        result = evaluate(
            dataset = dataset,
            metrics=[context_recall, context_precision, faithfulness],
            llm=gemini_llm, 
            embeddings=gemini_embeddings 
        )
        
        df_result = result.to_pandas()
        
        print("\n=== KẾT QUẢ ĐIỂM SỐ CHI TIẾT CỦA CÂU HỎI NÀY ===")
        print(f" 1. Context Recall    : {df_result['context_recall'][0]:.4f}")
        print(f" 2. Context Precision : {df_result['context_precision'][0]:.4f}")
        print(f" 3. Faithfulness      : {df_result['faithfulness'][0]:.4f}")
        
    except Exception as e:
        print(f"\n[!] LỖI CHẤM ĐIỂM RAGAS: {e}")

if __name__ == "__main__":
    main()