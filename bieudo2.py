import matplotlib.pyplot as plt
import numpy as np

# 1. Khai báo dữ liệu (Hãy thay đổi các con số này bằng kết quả thực tế của bạn)
metrics = ['Context Recall\n(Độ bao phủ)', 'Context Precision\n(Độ tinh chuẩn)', 'Faithfulness\n(Tính trung thực)']
naive_scores = [0.70, 0.69, 0.85]      # Điểm số của Naive RAG (Dữ liệu mẫu)
advanced_scores = [0.88, 0.86, 0.93]   # Điểm số của Advanced RAG (Dữ liệu mẫu)

# Thiết lập vị trí và độ rộng của các cột
x = np.arange(len(metrics))  # Vị trí các nhãn trên trục x
width = 0.35  # Độ rộng của mỗi cột

# Tạo figure và trục
fig, ax = plt.subplots(figsize=(9, 6)) # Chiều ngang rộng hơn chút để chứa chữ

# 2. Vẽ các cột (Màu xanh và cam đồng bộ với Biểu đồ 1)
rects1 = ax.bar(x - width/2, naive_scores, width, label='Naive RAG', color='#3498db', edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x + width/2, advanced_scores, width, label='Advanced RAG', color='#e67e22', edgecolor='black', linewidth=0.5)

# 3. Trang trí biểu đồ (Nhãn, Tiêu đề, Trục)
ax.set_ylabel('RAGAS Score (0.00 - 1.00)', fontsize=12)
ax.set_title('Chart 2: RAGAS Metrics Evaluation\n(Naive RAG vs Advanced RAG)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.1) # Đặt giới hạn trục y cao hơn chút để có không gian cho số liệu
ax.legend(fontsize=11, loc='upper left')

# Thêm lưới ngang để dễ so sánh điểm
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 4. Hàm thêm nhãn dữ liệu (con số) lên đầu mỗi cột
def autolabel(rects):
    """Gắn text label hiển thị chiều cao phía trên mỗi cột."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Dịch lên 3 points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

# Gọi hàm để gắn số
autolabel(rects1)
autolabel(rects2)

# Căn chỉnh layout
fig.tight_layout()

# 5. Lưu biểu đồ thành file ảnh (Xóa dấu # ở dòng dưới để lưu ảnh)
plt.savefig('chart2_ragas_metrics.png', dpi=300) 

# Hiển thị biểu đồ
plt.show()