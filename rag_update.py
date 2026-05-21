import json
import os
import re
import time 
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import InvalidArgument

# ----- Cài đặt Thư viện OSS (Mã nguồn mở) -----
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    HnswConfigDiff,
    Filter, FieldCondition, MatchText, MatchValue
)
from sentence_transformers import SentenceTransformer, CrossEncoder

# ----- (Bước 3) Cấu hình Google Gemini -----
# Load environment variables from .env (if present)
load_dotenv()

# Read API key from environment variable `GOOGLE_API_KEY`
# Fallback remains a placeholder to help detect missing config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

llm_model = None
try:
    # Guard: missing or still the placeholder
    if not GOOGLE_API_KEY or "YOUR_GOOGLE_API_KEY" in GOOGLE_API_KEY:
        raise InvalidArgument("Vui lòng dán API Key của bạn vào biến môi trường GOOGLE_API_KEY.")

    genai.configure(api_key=GOOGLE_API_KEY) # type: ignore
    # Sử dụng Gemini variant phù hợp
    llm_model = genai.GenerativeModel('gemini-2.5-flash') # type: ignore
    print("Đã kết nối thành công với Google Gemini.")

except Exception as e:
    print(f"LỖI KẾT NỐI GEMINI: {e}")

# ----- Cấu hình Model OSS -----
embed_model = None
rerank_model = None
EMBEDDING_DIM = None

try:
    print("Đang tải model AI (có thể mất chút thời gian lần đầu)...")
    # Tải model embedding
    embed_model = SentenceTransformer('BAAI/bge-m3', device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu')
    EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
    
    # Tải model reranker
    rerank_model = CrossEncoder('BAAI/bge-reranker-large', device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu')
    print("Tải model hoàn tất.")
except Exception as e:
    print(f"LỖI TẢI MODEL: {e}")

# -----------------------------------------------

class RAGPipelineAdvanced:
    def __init__(self, input_file="input.txt"):
        if not llm_model or not embed_model or not rerank_model or EMBEDDING_DIM is None:
            print("LỖI: Chưa khởi tạo được các Model AI. Dừng hệ thống.")
            self.client = None
            return

        self.input_file = input_file
        
        # Tạo tên Collection dựa trên tên file
        # Ví dụ: file "trans_f1.json" -> collection "db_trans_f1_json"
        
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', os.path.basename(input_file))
        self.collection_name = f"db_{safe_name}" 
        # --------------------

        self.db_path = "./qdrant_db_storage"
        self.chat_history = [] 
        self.current_context = ""
        
        # 1. Khởi tạo Qdrant Client (On-disk)
        try:
            self.client = QdrantClient(path=self.db_path)
        except Exception as e:
            print(f"Lỗi khởi tạo Qdrant: {e}")
            self.client = None
            return
        
        # 2. Tải dữ liệu thô vào bộ nhớ (để lấy ngữ cảnh & tóm tắt)
        self.kb_memory = self._load_memory_kb()
        if not self.kb_memory: 
            self.client = None
            return

        # 3. Kiểm tra và xây dựng lại CSDL Vector nếu cần
        self._check_and_build_db()

    def _load_input_data(self):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            json_start = content.find('[')
            if json_start == -1: return []
            return json.loads(content[json_start:])
        except Exception: return []

    def _load_memory_kb(self):
        data = self._load_input_data()
        if not data: return None
        kb = {"utterances": {}, "utterance_order": [], "full_text": ""}
        full_text_list = []
        
        # SỬA LỖI: Xử lý trường hợp thiếu key 'speaker'
        for i, utt in enumerate(data):
            utt_id = f"utt_{i}"
            
            # Gán giá trị mặc định nếu thiếu
            speaker = utt.get('speaker', 'Unknown Speaker')
            text = utt.get('text', '')
            start = utt.get('start', 0)
            
            # Cập nhật lại utt với các key đảm bảo tồn tại
            utt_safe = utt.copy()
            utt_safe['speaker'] = speaker
            utt_safe['text'] = text
            utt_safe['start'] = start
            
            kb["utterances"][utt_id] = utt_safe
            kb["utterance_order"].append(utt_id)
            
            # Tạo văn bản đầy đủ cho tính năng tóm tắt
            full_text_list.append(f"{speaker}: {text}")
        
        kb["full_text"] = "\n".join(full_text_list)
        return kb
    
    def _chunk_text(self, text, chunk_size=40, overlap=10):
        """Chia văn bản dài thành các đoạn ngắn hơn dựa trên số từ."""
        words = text.split()
        # Nếu câu ngắn hơn chunk_size, giữ nguyên
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_str = " ".join(words[i:i + chunk_size])
            chunks.append(chunk_str)
        return chunks
    
    def _check_and_build_db(self):
        if self.client is None or self.kb_memory is None or embed_model is None or EMBEDDING_DIM is None: return

        # Logic kiểm tra cache đơn giản hóa 
        if self.client.collection_exists(self.collection_name): # type: ignore
            # self.client.delete_collection(self.collection_name)
            return # Đã có DB
        
        print("\n--- Đang xây dựng CSDL Vector lần đầu (Áp dụng Chunking)... ---")
        self.client.create_collection( # type: ignore
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE), # type: ignore
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100) # type: ignore
        )
        
        # Vector hóa và nạp dữ liệu
        docs = []
        ids = []
        payloads = []
        utterance_ids = self.kb_memory["utterance_order"]
        
        point_id = 0 # Biến đếm ID độc lập cho Qdrant vì 1 câu có thể đẻ ra nhiều chunk
        
        for uid in utterance_ids:
            utt = self.kb_memory["utterances"][uid]
            original_text = utt['text']
            speaker = utt['speaker']
            
            # 1. Băm nhỏ văn bản nếu câu quá dài (Giúp Vector không bị loãng)
            sub_chunks = self._chunk_text(original_text, chunk_size=40, overlap=10)
            
            for chunk in sub_chunks:
                # 2. Gắn tên người nói vào chunk để Model Embedding hiểu rõ bối cảnh nhân vật
                chunk_for_embedding = f"{speaker}: {chunk}"
                
                docs.append(chunk_for_embedding) # Dùng chunk nhỏ, được làm giàu ngữ cảnh để vector hóa
                ids.append(point_id)
                payloads.append({
                    "text_original": original_text, #  Vẫn lưu lại câu thoại gốc
                    "doc_id": uid,                  #  Vẫn trỏ về ID gốc để lấy ngữ cảnh [idx-1, idx+1]
                    "speaker": speaker 
                })
                point_id += 1 # Tăng ID cho chunk tiếp theo
            
        embeddings = embed_model.encode(docs, show_progress_bar=True)
        
        points = [
            PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=payloads[i]) 
            for i in range(len(ids))
        ]
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection_name, points[i:i+batch_size]) # type: ignore
            
        print(f"Xây dựng DB hoàn tất! Đã lưu {point_id} vector chunks.")
    # ==========================================
    # HÀM ĐÓNG KẾT NỐI
    # ==========================================
    def close_connection(self):
        """Đóng kết nối Qdrant để giải phóng thư mục DB"""
        if self.client:
            self.client.close()
            print("Đã đóng kết nối CSDL.")
    # ==========================================
    # TÍNH NĂNG 1: TÓM TẮT CUỘC HỌP
    # ==========================================
    def summarize_meeting(self):
        if not llm_model or not self.kb_memory: return "Lỗi: Hệ thống chưa sẵn sàng."

        print("\n--- Đang tổng hợp và tóm tắt cuộc họp... ---")
        full_transcript = self.kb_memory["full_text"]
        
        
        
        prompt = f"""
        Bạn là một trợ lý thư ký chuyên nghiệp. Dưới đây là biên bản ghi lại của một cuộc họp.
        Hãy viết một bản tóm tắt chi tiết bao gồm:
        1. Mục đích chính của cuộc họp.
        2. Các nội dung chính đã thảo luận.
        3. Các quyết định đã được đưa ra hoặc các bước hành động (Action Items) tiếp theo.
        4. Các mốc thời gian (Deadline) nếu có.

        --- TRANSCRIPT ---
        {full_transcript}
        --- END TRANSCRIPT ---
        
        Bản tóm tắt (bằng tiếng Việt):
        
        QUAN TRỌNG: 
        - Chỉ sử dụng thông tin CÓ TRONG TRANSCRIPT ở trên. 
        - TUYỆT ĐỐI KHÔNG thêm thắt thông tin bên ngoài. 
        - Nếu thông tin không có trong transcript (ví dụ: không có deadline), hãy ghi là "Không được đề cập".
        """
        try:
            response = llm_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Lỗi khi tóm tắt: {e}"
    # ==========================================
    # TÍNH NĂNG 2: CHATBOT HỎI ĐÁP (RAG THÔNG MINH)
    # ==========================================
    def rewrite_query(self, user_query):
        """
        Viết lại câu hỏi dựa trên lịch sử chat để đầy đủ ý nghĩa.
        """
        if not self.chat_history or not llm_model:
            return user_query 
            
        # Lấy 3 lượt hội thoại gần nhất
        recent_history = self.chat_history[-3:] 
        history_str = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in recent_history])
        
        prompt = f"""
        You are a conversation context analyzer. 
        Read the chat history and the new user query below.
        
        Your task is to rewrite the "New Query" to make it a standalone, fully complete sentence by replacing pronouns (it, they, he, she...) with the actual subjects from the history.
        
        CRITICAL RULES:
        1. Output ONLY the rewritten query, nothing else.
        2. DO NOT answer the query.
        3. You MUST keep the EXACT SAME LANGUAGE as the "New Query". If the New Query is in English, the output MUST be in English. If it is in Vietnamese, output in Vietnamese.

        --- History ---
        {history_str}
        
        New Query: {user_query}
        
        Rewritten Query:
        """
        
        try:
            response = llm_model.generate_content(prompt)
            rewritten = response.text.strip()
            # In ra để debug (có thể comment lại nếu muốn giao diện sạch hơn)
            # print(f"   [Debug] Query gốc: '{user_query}' -> Query sửa: '{rewritten}'")
            return rewritten
        except:
            return user_query

    def check_context_sufficiency(self, query, context):
        """
        Kiểm tra xem context hiện tại có đủ để trả lời câu hỏi không.
        """
        if not context or not llm_model: return False
        
        prompt = f"""
        Ngữ cảnh hiện tại:
        ---
        {context}
        ---
        
        Câu hỏi: "{query}"
        
        Dựa VÀO CHÍNH XÁC ngữ cảnh trên, liệu có đủ thông tin để trả lời câu hỏi này không?
        Chỉ trả lời duy nhất một từ: "YES" hoặc "NO".
        """
        try:
            response = llm_model.generate_content(prompt)
            return "YES" in response.text.strip().upper()
        except:
            return False

    def retrieve_context(self, query):
        if self.client is None or embed_model is None or rerank_model is None or self.kb_memory is None:
            return []

        # 1. Tìm kiếm Vector
        query_vec = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        resp = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,                 
                # query_filter=query_filter,       # áp dụng filter nếu có
                limit=30,
                with_payload=True
            )

        hits = resp.points or []
        
        
        if not hits: return []
        
        # 2. Lọc trùng lặp & Rerank
        unique_hits = []
        seen_doc_ids = set() # Cuốn sổ ghi nhớ ID đã gặp
        
        for hit in hits:
            if not hit.payload: continue
            doc_id = hit.payload.get('doc_id')
            
            # KIỂM TRA: Nếu doc_id chưa từng xuất hiện thì mới lấy
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_hits.append(hit)
        
        if not unique_hits: return []

        # Tạo pairs CHỈ TỪ những hit không trùng lặp
        pairs = [[query, hit.payload.get('text_original', '')] for hit in unique_hits]
        scores = rerank_model.predict(pairs) # type: ignore
        
        # Lấy Top 5 (Lưu ý: Dùng unique_hits thay vì hits)
        ranked_hits = sorted(zip(scores, unique_hits), key=lambda x: x[0], reverse=True)[:5]
        
        # 3. Context Augmentation (Lấy 1 trước, 1 sau)
        final_chunks = []
        processed_ids = set()
        
        # Map ID sang index trong list tuần tự
        id_to_index = {uid: i for i, uid in enumerate(self.kb_memory["utterance_order"])}
        
        for score, hit in ranked_hits:
            if not hit.payload: continue
            doc_id = hit.payload.get('doc_id')
            if not doc_id or doc_id in processed_ids: continue
            
            idx = id_to_index.get(doc_id)
            if idx is None: continue
            
            # Cửa sổ [idx-1, idx+1]
            start = max(0, idx - 1)
            end = min(len(self.kb_memory["utterance_order"]), idx + 2)
            
            chunk_text = []
            for i in range(start, end):
                uid = self.kb_memory["utterance_order"][i]
                processed_ids.add(uid)
                u = self.kb_memory["utterances"][uid]
                chunk_text.append(f"[{u['start']}] {u['speaker']}: {u['text']}")
            
            final_chunks.append("\n".join(chunk_text))
            
        return final_chunks



    def chat(self, user_query, return_context: bool = False):
        """
        Trả về câu trả lời. Nếu `return_context=True` thì trả về tuple (answer, final_context_str).
        """
        if not llm_model:
            if return_context:
                return "Lỗi: Model chưa sẵn sàng.", ""
            return "Lỗi: Model chưa sẵn sàng."

        # --- TỐI ƯU 1: CHỈ REWRITE NẾU CÓ LỊCH SỬ ---
        # Nếu chưa chat câu nào (list rỗng), thì câu hỏi của user là ngữ cảnh đầy đủ rồi.
        if not self.chat_history:
            rewritten_query = user_query
            print(f"   [Smart RAG] Câu hỏi đầu tiên, bỏ qua bước Rewrite.")
        else:
            # Chỉ tốn request này khi đã chat > 1 câu
            rewritten_query = self.rewrite_query(user_query)
            print(f"   [Smart RAG] Đã viết lại câu hỏi: {rewritten_query}")
        
        # --- TỐI ƯU 2: LUÔN LUÔN TÌM KIẾM (BỎ CHECK SUFFICIENCY) ---
        
        print(f"   [Smart RAG] Đang tìm kiếm thông tin...")
        context_chunks = self.retrieve_context(rewritten_query)
        
        if not context_chunks:
            final_context_str = "Không tìm thấy thông tin cụ thể trong tài liệu."
        else:
            final_context_str = "\n---\n".join(context_chunks)
            # Cập nhật context hiện tại (nếu sau này cần dùng lại)
            self.current_context = final_context_str 
        
        # Tạo prompt lịch sử
        history_str = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in self.chat_history[-3:]])

        prompt = f"""
        Bạn là trợ lý ảo chuyên nghiệp.
        Dựa vào [THÔNG TIN ĐƯỢC CUNG CẤP] dưới đây để trả lời câu hỏi.
        
        QUY TẮC:
        1. Chỉ trả lời dựa trên thông tin được cung cấp.
        2. Nếu thông tin không đủ, hãy nói "Tài liệu cuộc họp không đề cập đến vấn đề này".
        3. Trả lời ngắn gọn, đi thẳng vào vấn đề.

        --- LỊCH SỬ TRÒ CHUYỆN ---
        {history_str}
        
        --- THÔNG TIN ĐƯỢC CUNG CẤP ---
        {final_context_str}
        
        --- CÂU HỎI ---
        User: {user_query}
        (Ngữ cảnh ẩn: {rewritten_query})
        
        Câu trả lời:
        """
        
        try:
            # --- TỐI ƯU 3: CẤU HÌNH GENERATION ---
            # Giảm max_output_tokens để trả lời nhanh hơn nếu cần
            response = llm_model.generate_content(prompt)
            answer = response.text.strip()
            
            # Lưu lịch sử
            self.chat_history.append((user_query, answer))
            if return_context:
                return answer, final_context_str
            return answer
        except Exception as e:
            return f"Lỗi khi tạo câu trả lời: {e}"

# ==========================================
# CHƯƠNG TRÌNH CHÍNH (MAIN MENU)
# ==========================================
def main():
    print("Đang khởi động hệ thống...")
    
    # --- CẤU HÌNH THƯ MỤC ---
    data_folder = 'tailieu'  # Tên thư mục chứa file json
    
    # Kiểm tra xem thư mục có tồn tại không để tránh lỗi crash
    if not os.path.exists(data_folder):
        print(f"LỖI: Không tìm thấy thư mục '{data_folder}' tại đường dẫn hiện tại!")
        print(f"Đường dẫn đang tìm: {os.path.abspath(data_folder)}")
        return

    # VÒNG LẶP NGOÀI: CHỌN FILE
    while True:
        # Thay đổi: Quét file trong folder 'tailieu'
        files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
        
        if not files:
            print(f"LỖI: Không tìm thấy file .json nào trong thư mục '{data_folder}'!")
            return

        print("\n" + "="*40)
        print(f" DANH SÁCH CÁC CUỘC HỌP TRONG '{data_folder}' ")
        print("="*40)
        for i, f in enumerate(files):
            print(f"{i + 1}. {f}")
        print(f"{len(files) + 1}. ❌ Thoát chương trình")

        selected_filename = None # Lưu tên file
        
        while True:
            try:
                choice = input(f"\n>> Chọn file (1-{len(files)+1}): ").strip()
                idx = int(choice) - 1
                if idx == len(files): # Chọn thoát
                    print("Tạm biệt!")
                    return
                if 0 <= idx < len(files):
                    selected_filename = files[idx]
                    break
                print("Số không hợp lệ.")
            except ValueError:
                print("Vui lòng nhập số.")

        # Thay đổi: Tạo đường dẫn đầy đủ (ví dụ: tailieu/abc.json)
        full_path_to_file = os.path.join(data_folder, selected_filename)

        print(f"\n✅ Đang tải dữ liệu từ: {full_path_to_file}")
        
        # Khởi tạo App với đường dẫn đầy đủ
        app = RAGPipelineAdvanced(input_file=full_path_to_file)
        
        if not app.client:
            print("\n!!! KHỞI TẠO THẤT BẠI - Quay lại chọn file...")
            continue 

        # VÒNG LẶP TRONG: TƯƠNG TÁC VỚI FILE ĐÃ CHỌN
        back_to_menu = False
        while not back_to_menu:
            print(f"\n--- Đang làm việc với: {selected_filename} ---") # Hiển thị tên file cho gọn
            print("1. 📝 Tóm tắt nội dung")
            print("2. 💬 Chatbot hỏi đáp")
            print("3. 🔄 Đổi sang file khác (Đóng DB hiện tại)")
            print("4. ❌ Thoát hẳn")
            
            choice = input("\n>> Nhập lựa chọn: ").strip()
            
            if choice == '1':
                print(app.summarize_meeting())
                input("\n[Enter để tiếp tục...]")
                
            elif choice == '2':
                print("\n--- CHAT (Gõ 'exit' để quay lại) ---")
                print("Gõ '/ctx <câu hỏi>' nếu muốn hiển thị ngữ cảnh retrieval kèm câu trả lời.")
                app.chat_history = [] 
                while True:
                    q = input("Bạn: ")
                    if q.lower() in ['exit', 'quit', 'thoát']: break
                    if not q.strip(): continue
                    show_ctx = False
                    if q.startswith('/ctx ') or q.startswith('/context '):
                        show_ctx = True
                        q = q.split(' ', 1)[1].strip()
                        if not q: continue
                    print("Bot đang suy nghĩ...", end="\r")
                    if show_ctx:
                        ans, ctx = app.chat(q, return_context=True)
                        print(" "*20, end="\r")
                        print(f"Bot: {ans}\n\n--- Ngữ cảnh tìm được ---\n{ctx}")
                    else:
                        ans = app.chat(q)
                        print(" "*20, end="\r") 
                        print(f"Bot: {ans}")

            elif choice == '3':
                app.close_connection()
                back_to_menu = True 

            elif choice == '4':
                app.close_connection()
                print("Tạm biệt!")
                return 
            else:
                print("Sai cú pháp.")

if __name__ == "__main__":
    main()