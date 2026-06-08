import pandas as pd
import csv
import os
import warnings
import time
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rag_update import RAGPipelineAdvanced

# Tắt cảnh báo
warnings.filterwarnings("ignore")

def calculate_ir_metrics(results, k=5):
    hits = 0
    mrr_sum = 0
    total = len(results)
    if total == 0:
        return 0, 0

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

    hit_rate = hits / total
    mrr = mrr_sum / total
    return hit_rate, mrr

def main():
    print("="*60)
    print(" ĐANG CHẠY ĐÁNH GIÁ (AUTO-RETRY 429 & LOG CHI TIẾT ID) ")
    print("="*60)

    # 1. Khởi tạo file log real-time
    log_filename = "evaluation_realtime_log.csv"
    with open(log_filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "ground_truth", "retrieved_doc_ids", "answer", "status"])

    load_dotenv()
    df_dataset = pd.read_csv("test_dataset.csv")
    
    all_ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    all_ir_results = []
    
    json_files = df_dataset['file_source'].unique()
    
    for file_name in json_files:
        full_path = os.path.join("tailieu", str(file_name).strip())
        if not os.path.exists(full_path): 
            print(f"[!] Không tìm thấy file {full_path}")
            continue
            
        print(f"\n[*] Đang đánh giá file: {file_name}")
        app = RAGPipelineAdvanced(input_file=full_path)
            
        df_file = df_dataset[df_dataset['file_source'] == file_name]
        
        for index, row in df_file.iterrows():
            question = str(row['question']).strip()
            ground_truth = str(row['ground_truth']).strip()
            ref_id = str(row['reference_context_id']).strip() # KHÔI PHỤC DÒNG NÀY ĐỂ CHẤM IR
            
            print(f"  -> Q: {question}")
            
            # CƠ CHẾ AUTO-RETRY ĐỂ CHỐNG LỖI 429 QUÁ TẢI API
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ans, retrieved_context_str, ctx_ids = app.chat(
                        user_query=question, 
                        return_context=True,
                        use_metadata_filter=False
                    )
                    
                    # Nếu câu trả lời bị dính lỗi từ Gemini, ép nó văng lỗi để Retry
                    if "Lỗi khi tạo câu trả lời" in str(ans):
                        raise Exception(ans)
                        
                    # GHI LOG THÀNH CÔNG
                    with open(log_filename, "a", encoding="utf-8-sig", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([question, ground_truth, str(ctx_ids), str(ans), "OK"])
                    
                    print(f"     ✅ Done! IDs: {ctx_ids}")
                    
                    # NẠP DỮ LIỆU ĐỂ CHẤM RAGAS & IR
                    ctx_texts = retrieved_context_str.split("\n---\n")
                    all_ragas_data["question"].append(question)
                    all_ragas_data["answer"].append(str(ans))
                    all_ragas_data["contexts"].append(ctx_texts)
                    all_ragas_data["ground_truth"].append(ground_truth)
                    all_ir_results.append({"reference_context_id": ref_id, "retrieved_doc_ids": ctx_ids})
                    
                    break # Thành công thì thoát khỏi vòng lặp Retry
                    
                except Exception as e:
                    print(f"    [!] Lỗi API (Thử lại {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print("    ⏳ Đang chờ 30 giây để Google nhả Rate Limit...")
                        time.sleep(30)
                    else:
                        print("    ❌ Bỏ qua do API lỗi liên tục.")
                        with open(log_filename, "a", encoding="utf-8-sig", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([question, ground_truth, "[]", "ERROR", str(e)])
            
            # Ngủ 15 giây giữa các câu để né Limit 15 RPM
            time.sleep(15)
            
        app.close_connection()
    
    # -----------------------------------------------------
    # CHẤM ĐIỂM IR METRICS (ĐÃ ĐƯỢC KHÔI PHỤC)
    # -----------------------------------------------------
    print("\n" + "="*50)
    print(" TỔNG KẾT ĐIỂM TRUY XUẤT (IR METRICS) ")
    print("="*50)
    hit_rate, mrr = calculate_ir_metrics(all_ir_results, k=5)
    print(f" - Hit Rate@5 : {hit_rate:.4f}")
    print(f" - MRR@5      : {mrr:.4f}")

    # -----------------------------------------------------
    # CHẤM ĐIỂM RAGAS
    # -----------------------------------------------------
    print("\n" + "="*50)
    print(" BẮT ĐẦU CHẤM ĐIỂM RAGAS (SẼ MẤT 5-10 PHÚT...) ")
    print("="*50)
    
    # Bật tính năng Retry ngầm của Langchain
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        max_retries=5
    )
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    dataset = Dataset.from_dict(all_ragas_data)
    
    try:
        # Ép Ragas chạy từng luồng một (max_workers=1) để tránh bị chặn IP
        my_run_config = RunConfig(max_workers=1, max_wait=60, max_retries=10)
        
        result = evaluate(
            dataset=dataset,
            metrics=[context_recall, context_precision, faithfulness],
            llm=gemini_llm,
            embeddings=gemini_embeddings,
            run_config=my_run_config # Truyền cấu hình chặn lỗi vào đây
        )
        
        df_result = result.to_pandas()
        df_result.to_csv("ragas_evaluation_results.csv", index=False, encoding="utf-8-sig")
        
        print("\n" + "="*50)
        print(" TỔNG KẾT ĐIỂM RAGAS (GENERATION METRICS) ")
        print("="*50)
        print(f" - Context Recall    : {result['context_recall']:.4f}")
        print(f" - Context Precision : {result['context_precision']:.4f}")
        print(f" - Faithfulness      : {result['faithfulness']:.4f}")
        
        print("\n✅ Đã xuất điểm chi tiết từng câu ra file: 'ragas_evaluation_results.csv'")
    except Exception as e:
        print(f"\n❌ Lỗi khi chạy thư viện RAGAS: {e}")

if __name__ == "__main__":
    main()