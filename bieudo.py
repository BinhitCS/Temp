import matplotlib.pyplot as plt
import numpy as np

# Cài đặt style cho đẹp và chuyên nghiệp
plt.style.use('seaborn-v0_8-whitegrid')

# Khởi tạo biểu đồ (Kích thước 8x6)
fig, ax = plt.subplots(figsize=(8, 6))

# --- DỮ LIỆU CỦA CẬU ĐIỀN VÀO ĐÂY ---
labels = ['Hit Rate @1', 'Hit Rate @3', 'Hit Rate @5']
baseline_hit = [45, 62, 70] # Số liệu của Cách cũ (1 câu = 1 chunk)
proposed_hit = [68, 86, 94] # Số liệu của Cách mới (Parent Document)

x = np.arange(len(labels))  # Vị trí các nhóm cột
width = 0.35  # Độ rộng của cột

# Vẽ các cột
rects1 = ax.bar(x - width/2, baseline_hit, width, label='Base line (Naive RAG)', color='#B0BEC5')
rects2 = ax.bar(x + width/2, proposed_hit, width, label='Advanced RAG (Hệ thống đề xuất)', color='#4CAF50')

# Căn chỉnh nhãn và tiêu đề
ax.set_ylabel('Tỷ lệ phần trăm (%)', fontsize=12, fontweight='bold')
ax.set_title('So sánh hiệu năng tìm kiếm: Hit Rate @K', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 110) # Giới hạn trục Y đến 110 để có chỗ hiển thị số trên đầu cột

# Hiển thị chú thích (Legend)
ax.legend(fontsize=11, loc='upper left')

# Hàm thêm số liệu phần trăm tự động trên đầu mỗi cột
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Dịch lên 3 points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Chỉnh bố cục và lưu ảnh
plt.tight_layout()
plt.savefig('Hit_Rate_Evaluation.png', dpi=300) # dpi=300 giúp ảnh cực kỳ sắc nét khi chiếu lên màn hình lớn
plt.show()