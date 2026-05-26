import matplotlib.pyplot as plt
import numpy as np

# 1. Khai báo dữ liệu (Hãy thay bằng số liệu thực tế trong đồ án của bạn)
metrics = ['Hit Rate', 'MRR']
naive_scores = [0.70, 0.61]      # Điểm của Naive RAG
advanced_scores = [0.86, 0.75]   # Điểm của Advanced RAG

# Thiết lập vị trí và độ rộng của các cột
x = np.arange(len(metrics))  # Vị trí các nhãn trên trục x
width = 0.35  # Độ rộng của mỗi cột

# Tạo figure và trục
fig, ax = plt.subplots(figsize=(8, 6))

# 2. Vẽ các cột
rects1 = ax.bar(x - width/2, naive_scores, width, label='Naive RAG', color='#3498db', edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x + width/2, advanced_scores, width, label='Advanced RAG', color='#e67e22', edgecolor='black', linewidth=0.5)

# 3. Trang trí biểu đồ (Nhãn, Tiêu đề, Trục)
ax.set_ylabel('Performance Score', fontsize=12)
ax.set_title('Chart 1: Retrieval Performance\n(Naive RAG vs Advanced RAG)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.05) # Đặt giới hạn trục y từ 0 đến 1.05 để có chỗ hiển thị số
ax.legend(fontsize=11, loc='upper left')

# Thêm lưới ngang để dễ nhìn điểm số hơn
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 4. Hàm thêm nhãn dữ liệu (con số) lên đầu mỗi cột
def autolabel(rects):
    """Gắn text label hiển thị chiều cao phía trên mỗi cột."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Dịch lên 3 points so với đỉnh cột
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

# Gọi hàm để gắn số cho cả 2 nhóm cột
autolabel(rects1)
autolabel(rects2)

# Căn chỉnh layout cho đẹp
fig.tight_layout()

# 5. Hiển thị hoặc Lưu biểu đồ
plt.savefig('chart1_retrieval_performance.png', dpi=300) # Mở comment dòng này nếu muốn lưu thành file ảnh nét cao
plt.show()