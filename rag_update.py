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
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    Filter,
    FieldCondition,
    MatchText,
    MatchValue,
    Range,  # THÊM TÍNH NĂNG LỌC THEO KHOẢNG THỜI GIAN
)
from sentence_transformers import SentenceTransformer, CrossEncoder

# ----- (Bước 3) Cấu hình Google Gemini -----
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

llm_model = None
try:
    if not GOOGLE_API_KEY or "YOUR_GOOGLE_API_KEY" in GOOGLE_API_KEY:
        raise InvalidArgument(
            "Vui lòng dán API Key của bạn vào biến môi trường GOOGLE_API_KEY."
        )

    genai.configure(api_key=GOOGLE_API_KEY)  # type: ignore
    llm_model = genai.GenerativeModel("gemini-2.5-flash")  # type: ignore
    print("Đã kết nối thành công với Google Gemini.")

except Exception as e:
    print(f"LỖI KẾT NỐI GEMINI: {e}")

# ----- Cấu hình Model OSS -----
embed_model = None
rerank_model = None
EMBEDDING_DIM = None

try:
    print("Đang tải model AI (có thể mất chút thời gian lần đầu)...")
    embed_model = SentenceTransformer(
        "BAAI/bge-m3",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    )
    EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()

    rerank_model = CrossEncoder(
        "BAAI/bge-reranker-large",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    )
    print("Tải model hoàn tất.")
except Exception as e:
    print(f"LỖI TẢI MODEL: {e}")

# -----------------------------------------------


class RAGPipelineAdvanced:
    def __init__(self, input_file="input.txt"):
        if (
            not llm_model
            or not embed_model
            or not rerank_model
            or EMBEDDING_DIM is None
        ):
            print("LỖI: Chưa khởi tạo được các Model AI. Dừng hệ thống.")
            self.client = None
            return

        self.input_file = input_file
        safe_name = re.sub(r"[^a-zA-Z0-9]", "_", os.path.basename(input_file))
        self.collection_name = f"db_{safe_name}"
        self.db_path = "./qdrant_db_storage"
        self.chat_history = []
        self.current_context = ""

        try:
            self.client = QdrantClient(path=self.db_path)
        except Exception as e:
            print(f"Lỗi khởi tạo Qdrant: {e}")
            self.client = None
            return

        self.kb_memory = self._load_memory_kb()
        if not self.kb_memory:
            self.client = None
            return

        self._check_and_build_db()

    def _load_input_data(self):
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                content = f.read()
            json_start = content.find("[")
            if json_start == -1:
                return []
            return json.loads(content[json_start:])
        except Exception:
            return []

    def _load_memory_kb(self):
        data = self._load_input_data()
        if not data:
            return None
        kb = {"utterances": {}, "utterance_order": [], "full_text": ""}
        full_text_list = []

        for i, utt in enumerate(data):
            utt_id = f"utt_{i}"
            speaker = utt.get("speaker", "Unknown Speaker")
            text = utt.get("text", "")
            start = utt.get("start", 0)

            # Chuyển đổi start time sang giây để dùng cho Filter Range
            start_sec = 0
            if isinstance(start, str):
                try:
                    h, m, s = map(int, start.split(":"))
                    start_sec = h * 3600 + m * 60 + s
                except:
                    pass

            utt_safe = utt.copy()
            utt_safe["speaker"] = speaker
            utt_safe["text"] = text
            utt_safe["start"] = start
            utt_safe["start_sec"] = start_sec  # Thêm start_sec

            kb["utterances"][utt_id] = utt_safe
            kb["utterance_order"].append(utt_id)
            full_text_list.append(f"{speaker}: {text}")

        kb["full_text"] = "\n".join(full_text_list)
        return kb

    def _chunk_text(self, text, chunk_size=40, overlap=10):
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_str = " ".join(words[i : i + chunk_size])
            chunks.append(chunk_str)
        return chunks

    def _check_and_build_db(self):
        if (
            self.client is None
            or self.kb_memory is None
            or embed_model is None
            or EMBEDDING_DIM is None
        ):
            return

        if self.client.collection_exists(self.collection_name):
            return

        print("\n--- Đang xây dựng CSDL Vector lần đầu (Áp dụng Chunking)... ---")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )

        docs = []
        ids = []
        payloads = []
        utterance_ids = self.kb_memory["utterance_order"]
        point_id = 0

        for uid in utterance_ids:
            utt = self.kb_memory["utterances"][uid]
            original_text = utt["text"]
            speaker = utt["speaker"]

            sub_chunks = self._chunk_text(original_text, chunk_size=40, overlap=10)

            for chunk in sub_chunks:
                chunk_for_embedding = f"{speaker}: {chunk}"
                docs.append(chunk_for_embedding)
                ids.append(point_id)
                payloads.append(
                    {
                        "text_original": original_text,
                        "doc_id": uid,
                        "speaker": speaker,
                        "start_sec": utt.get("start_sec", 0),  # Gắn metadata thời gian
                    }
                )
                point_id += 1

        embeddings = embed_model.encode(docs, show_progress_bar=True)
        points = [
            PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=payloads[i])
            for i in range(len(ids))
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection_name, points[i : i + batch_size])

        print(f"Xây dựng DB hoàn tất! Đã lưu {point_id} vector chunks.")

    def close_connection(self):
        if self.client:
            self.client.close()
            print("Đã đóng kết nối CSDL.")

    def summarize_meeting(self):
        if not llm_model or not self.kb_memory:
            return "Lỗi: Hệ thống chưa sẵn sàng."

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

    def rewrite_query(self, user_query):
        if not self.chat_history or not llm_model:
            return user_query

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
            return response.text.strip()
        except:
            return user_query

    def check_context_sufficiency(self, query, context):
        if not context or not llm_model:
            return False
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

    def _extract_metadata_from_query(self, query):
        metadata = {}
        speaker_match = re.search(r"(Speaker\s*\d+)", query, re.IGNORECASE)
        if speaker_match:
            metadata["speaker"] = speaker_match.group(1).title()

        time_match = re.search(r"phút\s*(?:thứ\s*)?(\d+)", query, re.IGNORECASE)
        if time_match:
            minute = int(time_match.group(1))
            center_seconds = minute * 60
            metadata["time_range"] = {
                "gte": max(0, center_seconds - 120),
                "lte": center_seconds + 120,
            }
        return metadata

    # --- BỔ SUNG: Trả về thêm final_ids cho File Đánh Giá ---
    def retrieve_context(self, query, use_metadata_filter=True, use_reranker=True):
        if (
            self.client is None
            or embed_model is None
            or rerank_model is None
            or self.kb_memory is None
        ):
            return [], []

        query_filter = None
        if use_metadata_filter:
            extracted_meta = self._extract_metadata_from_query(query)
            must_conditions = []

            if "speaker" in extracted_meta:
                # print(f"   [Smart Filter] Khoanh vùng người nói: {extracted_meta['speaker']}")
                must_conditions.append(
                    FieldCondition(
                        key="speaker", match=MatchValue(value=extracted_meta["speaker"])
                    )
                )

            if "time_range" in extracted_meta:
                time_window = extracted_meta["time_range"]
                # print(f"   [Smart Filter] Khoanh vùng thời gian: Từ {time_window['gte']}s đến {time_window['lte']}s")
                must_conditions.append(
                    FieldCondition(
                        key="start_sec",
                        range=Range(gte=time_window["gte"], lte=time_window["lte"]),
                    )
                )

            if must_conditions:
                query_filter = Filter(must=must_conditions)

        query_vec = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=30 if use_reranker else 5,
            with_payload=True,
        )

        hits = resp.points or []
        if not hits:
            return [], []

        unique_hits = []
        seen_doc_ids = set()

        for hit in hits:
            if not hit.payload:
                continue
            doc_id = hit.payload.get("doc_id")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_hits.append(hit)

        if not unique_hits:
            return [], []

        ranked_hits = []
        if use_reranker:
            pairs = [
                [query, hit.payload.get("text_original", "")] for hit in unique_hits
            ]
            scores = rerank_model.predict(pairs)  # type: ignore
            ranked_hits = sorted(
                zip(scores, unique_hits), key=lambda x: x[0], reverse=True
            )[:5]
        else:
            ranked_hits = [(hit.score, hit) for hit in unique_hits[:5]]

        final_chunks = []
        final_ids = []  # Thêm mảng này để trả về cho file đánh giá
        processed_ids = set()
        id_to_index = {
            uid: i for i, uid in enumerate(self.kb_memory["utterance_order"])
        }

        for score, hit in ranked_hits:
            if not hit.payload:
                continue
            doc_id = hit.payload.get("doc_id")

            if not doc_id or doc_id in processed_ids:
                continue
            idx = id_to_index.get(doc_id)
            if idx is None:
                continue

            start = max(0, idx - 1)
            end = min(len(self.kb_memory["utterance_order"]), idx + 2)

            chunk_text = []
            for i in range(start, end):
                uid = self.kb_memory["utterance_order"][i]
                processed_ids.add(uid)
                u = self.kb_memory["utterances"][uid]
                chunk_text.append(f"[{u['start']}] {u['speaker']}: {u['text']}")

            final_chunks.append("\n".join(chunk_text))
            final_ids.append(doc_id)

        return final_chunks, final_ids

    # --- BỔ SUNG: Nhận tham số cấu hình và trả về 3 biến ---
    def chat(
        self,
        user_query,
        return_context: bool = False,
        use_metadata_filter=True,
        use_reranker=True,
    ):
        if not llm_model:
            if return_context:
                return "Lỗi: Model chưa sẵn sàng.", "", []
            return "Lỗi: Model chưa sẵn sàng."

        if not self.chat_history:
            rewritten_query = user_query
        else:
            rewritten_query = self.rewrite_query(user_query)

        # Lấy context và ID
        context_chunks, context_ids = self.retrieve_context(
            rewritten_query,
            use_metadata_filter=use_metadata_filter,
            use_reranker=use_reranker,
        )

        if not context_chunks:
            final_context_str = "Không tìm thấy thông tin cụ thể trong tài liệu."
        else:
            final_context_str = "\n---\n".join(context_chunks)
            self.current_context = final_context_str

        history_str = "\n".join(
            [f"User: {h[0]}\nBot: {h[1]}" for h in self.chat_history[-3:]]
        )

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
            response = llm_model.generate_content(prompt)
            answer = response.text.strip()
            self.chat_history.append((user_query, answer))

            if return_context:
                return answer, final_context_str, context_ids
            return answer
        except Exception as e:
            if return_context:
                return f"Lỗi khi tạo câu trả lời: {e}", "", []
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
                        ans, ctx, _ = app.chat(q, return_context=True)
                        print(" "*20, end="\r")
                        print(f"Bot: {ans}\n\n--- Ngữ cảnh tìm được ---\n{ctx}")
                        print("ID các đoạn được dùng:", _)
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