import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness
from ragas.run_config import RunConfig
import os
import warnings
import time
from dotenv import load_dotenv

# Tắt các cảnh báo nhỏ
warnings.filterwarnings("ignore")

# CẤU HÌNH LLM CHO RAGAS SỬ DỤNG GEMINI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# KẾT NỐI VỚI HỆ THỐNG RAG CỦA BẠN
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
            # Cắt ID trả về cẩn thận và so sánh
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
    print(" BẮT ĐẦU ĐÁNH GIÁ TOÀN BỘ DATASET (91 CÂU HỎI) ")
    print("="*60)

    # 1. Load API Key
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("LỖI: Chưa có GOOGLE_API_KEY trong file .env")
        return

    # 2. Đọc Dataset
    data_folder = "tailieu"
    try:
        df_dataset = pd.read_csv("test_dataset.csv")
        print(f"✅ Đã tải thành công file Dataset với {len(df_dataset)} câu hỏi.")
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file 'test_dataset.csv'.")
        return
        
    all_ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    all_ir_results = []
    
    # 3. Lấy danh sách các file duy nhất từ cột 'file_source' trong file CSV
    json_files = df_dataset['file_source'].unique()
    
    # 4. Vòng lặp duyệt qua từng file cuộc họp
    for file_name in json_files:
        # Ép kiểu tên file về str cho chắc ăn
        file_name = str(file_name).strip()
        full_path = os.path.join(data_folder, file_name)
        
        if not os.path.exists(full_path):
            print(f"\n[!] Bỏ qua: Không tìm thấy file gốc {full_path}")
            continue
            
        print(f"\n" + "="*50)
        print(f" ĐANG ĐÁNH GIÁ FILE: {file_name} ")
        print("="*50)
        
        # Lọc câu hỏi của file này
        df_file = df_dataset[df_dataset['file_source'] == file_name]
        
        # KHỞI TẠO RAG CHO FILE NÀY
        print(f"[*] Khởi tạo Vector Database cho {file_name}...")
        app = RAGPipelineAdvanced(input_file=full_path)
            
        # Vòng lặp chạy từng câu hỏi
        for index, row in df_file.iterrows():
            # [QUAN TRỌNG] Ép kiểu toàn bộ về chuỗi chuẩn
            question = str(row['question']).strip()
            ground_truth = str(row['ground_truth']).strip()
            ref_id = str(row['reference_context_id']).strip()
            
            print(f"  -> Q: {question}")
            
            try:
                ans, retrieved_context_str, ctx_ids = app.chat(
                    user_query=question, 
                    return_context=True,
                    use_metadata_filter=True, 
                    use_reranker=True         
                )
                
                # Tách text context thành mảng
                ctx_texts = retrieved_context_str.split("\n---\n")
                
                # Lưu trữ kết quả
                all_ragas_data["question"].append(question)
                all_ragas_data["answer"].append(str(ans))
                all_ragas_data["contexts"].append(ctx_texts)
                all_ragas_data["ground_truth"].append(ground_truth)
                
                all_ir_results.append({
                    "reference_context_id": ref_id, 
                    "retrieved_doc_ids": ctx_ids
                })
                
            except Exception as e:
                print(f"    [!] Lỗi khi truy vấn câu này: {e}")
            
            # NGỦ 4 GIÂY ĐỂ TRÁNH LỖI QUÁ TẢI (RATE LIMIT) CỦA GEMINI
            time.sleep(12)
            
        app.close_connection()
    
    if not all_ir_results:
        print("Không có kết quả nào được ghi nhận. Dừng chương trình.")
        return

    # ==========================================
    # 5. CHẤM ĐIỂM TRUY XUẤT (Hit Rate & MRR)
    # ==========================================
    print("\n" + "="*50)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ-ĐUN TRUY XUẤT (RETRIEVAL) ")
    print("="*50)
    hit_rate, mrr = calculate_ir_metrics(all_ir_results, k=5)
    print(f" => Hit Rate@5 : {hit_rate:.4f} (Trên 0.8 là xuất sắc)")
    print(f" => MRR@5      : {mrr:.4f} (Trên 0.7 là xuất sắc)")
    
    # LƯU CHECKPOINT (Tránh mất sạch công sức nếu Ragas lỗi)
    print("\n--- Đang lưu Checkpoint dữ liệu câu trả lời ---")
    df_checkpoint = pd.DataFrame(all_ragas_data)
    df_checkpoint.to_csv("checkpoint_rag_answers.csv", index=False, encoding='utf-8-sig')
    print("✅ Đã lưu file 'checkpoint_rag_answers.csv'")
    
    # ==========================================
    # 6. CHẤM ĐIỂM TẠO SINH BẰNG RAGAS (GEMINI)
    # ==========================================
    print("\n" + "="*50)
    print(" ĐANG CHẠY LLM-AS-A-JUDGE (RAGAS BẰNG GEMINI) ")
    print("="*50)
    print("Lưu ý: Quá trình này sẽ mất thêm khoảng 5-10 phút để AI tự chấm điểm...")
    
   
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0
    )
    
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    dataset = Dataset.from_dict(all_ragas_data)
    
    my_run_config = RunConfig(max_workers=1, max_wait=60, max_retries=10)

    try:
        result = evaluate(
            dataset = dataset,
            metrics=[context_recall, context_precision, faithfulness],
            llm=gemini_llm, 
            embeddings=gemini_embeddings,
            run_config=my_run_config  # <-- THÊM DÒNG NÀY VÀO ĐÂY
        )
        
        df_result = result.to_pandas()
        df_result.to_csv("ragas_evaluation_results.csv", index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(" TỔNG KẾT ĐIỂM RAGAS (GENERATION METRICS) ")
        print("="*50)
        print(f" - Context Recall    : {result['context_recall']:.4f}")
        print(f" - Context Precision : {result['context_precision']:.4f}")
        print(f" - Faithfulness      : {result['faithfulness']:.4f}")
        
        print("\n✅ Đã xuất điểm chi tiết từng câu ra file: 'ragas_evaluation_results.csv'")
        print("🎉 XIN CHÚC MỪNG! HỆ THỐNG ĐÃ HOÀN THÀNH QUÁ TRÌNH ĐÁNH GIÁ!")
        
    except Exception as e:
        print(f"\n[!] LỖI RAGAS: {e}")
        print("Tuy nhiên dữ liệu vẫn an toàn trong file 'checkpoint_rag_answers.csv'.")

if __name__ == "__main__":
    main()
