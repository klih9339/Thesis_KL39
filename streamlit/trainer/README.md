# Knowledge Map Trainer

Hệ thống trực quan hóa bản đồ kiến thức và đề xuất kế hoạch ôn tập dựa trên lý thuyết Active Recall.

## Tổng quan

Knowledge Map Trainer là một hệ thống AI được xây dựng để:
1. Hiển thị bản đồ kiến thức của học sinh với phân loại theo mức độ thành thạo (thấp, trung bình, cao)
2. Đề xuất kế hoạch ôn tập theo lý thuyết Active Recall
3. Dự đoán kết quả học tập của học sinh
4. Mô phỏng quá trình học trong tương lai

Hệ thống sử dụng mô hình Deep Concept-based Knowledge Tracing (DCRKT) để theo dõi trạng thái kiến thức của học sinh theo thời gian.

## Tính năng chính

### 1. Trực quan hóa bản đồ kiến thức
- Hiển thị bản đồ kiến thức dưới dạng đồ thị với các node là các khái niệm
- Màu sắc của node thể hiện mức độ thành thạo: đỏ (thấp), cam (trung bình), xanh (cao)
- Kích thước của node tỉ lệ với mức độ thành thạo

### 2. Kế hoạch ôn tập Active Recall
- Tạo kế hoạch ôn tập tập trung vào các khái niệm yếu
- Lập lịch ôn tập theo khoảng thời gian tối ưu (spaced repetition)
- Đề xuất câu hỏi ôn tập phù hợp với từng khái niệm

### 3. Dự đoán và mô phỏng
- Dự đoán khả năng học sinh trả lời đúng một câu hỏi cụ thể
- Mô phỏng quá trình học trong tương lai với một chuỗi câu hỏi
- Theo dõi sự thay đổi mức độ thành thạo theo thời gian

## Cấu trúc hệ thống

Hệ thống bao gồm các module chính:

- `knowledge_map.py`: Module trực quan hóa bản đồ kiến thức
- `active_recall.py`: Module lập kế hoạch ôn tập theo Active Recall
- `dashboard.py`: Module giao diện tương tác
- `main.py`: Module khởi tạo và điều khiển hệ thống

## Cách sử dụng

### Sử dụng từ dòng lệnh

```python
# Import thư viện
from trainer.main import init_trainer, prepare_for_student

# Khởi tạo hệ thống
model, dashboard, dfs, mappings = init_trainer("đường_dẫn_đến_mô_hình.pt")

# Chọn một học sinh
student_id = list(mappings["user2idx"].keys())[0]

# Tải dữ liệu học sinh
model = prepare_for_student(model, student_id, dfs, mappings)

# Hiển thị bản đồ kiến thức
dashboard.display_knowledge_map(student_id, save=True)

# Tạo kế hoạch ôn tập
dashboard.display_study_plan(student_id, plan_type="active_recall", days=7)

# Dự đoán kết quả
question_id = list(mappings["question2idx"].keys())[0]
dashboard.display_prediction(student_id, question_id)
```

### Sử dụng dashboard tương tác (yêu cầu Jupyter Notebook)

```python
from trainer.main import create_interactive_session

# Tạo phiên tương tác
model, dashboard, dfs, mappings = create_interactive_session()
```

## Yêu cầu hệ thống

- Python 3.7+
- PyTorch 1.7+
- NetworkX
- Matplotlib
- Pandas
- NumPy
- Seaborn
- ipywidgets (cho dashboard tương tác)

## Cài đặt

```bash
# Clone repo
git clone <repository_url>

# Cài đặt dependencies
pip install -r requirements.txt
```

## Tài liệu tham khảo

- Deep Concept-based Knowledge Tracing (DCRKT): [arXiv:2108.08105](https://arxiv.org/abs/2108.08105)
- Active Recall: Phương pháp học tập hiệu quả yêu cầu người học nhớ lại thông tin một cách chủ động