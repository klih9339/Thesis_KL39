import os
import torch
import matplotlib.pyplot as plt
from trainer.main import init_trainer, prepare_for_student

def main():
    """
    Demonstrate how to use the Knowledge Map Trainer system
    """
    # Sử dụng CUDA nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Tải mô hình (sử dụng mô hình đã huấn luyện nếu có)
    model_path = "checkpoints/dcrkt_model_fold_0.pt"
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file mô hình: {model_path}")
        print("Tìm kiếm mô hình trong thư mục checkpoints...")
        model_path = "checkpoints/dcrkt_model_fold_0.pt"
        if not os.path.exists(model_path):
            print(f"Không tìm thấy file mô hình: {model_path}")
            model_path = None
    
    # Khởi tạo hệ thống
    print("Đang khởi tạo hệ thống Knowledge Map Trainer...")
    model, dashboard, dfs, mappings = init_trainer(model_path, device=device)
    print("Đã khởi tạo hoàn tất!")
    
    # Hiển thị các ID học sinh có trong dữ liệu
    student_ids = list(mappings["user2idx"].keys())
    print(f"\nCó {len(student_ids)} học sinh trong dữ liệu.")
    print(f"Ví dụ ID học sinh: {student_ids[:5]}")
    
    # Chọn một học sinh để hiển thị
    # student_id = student_ids[0]
    student_id = 39
    
    # Chuẩn bị mô hình cho học sinh
    print(f"\nĐang tải dữ liệu lịch sử cho học sinh {student_id}...")
    model = prepare_for_student(model, student_id, dfs, mappings)
    
    # Hiển thị bản đồ kiến thức
    print("\n1. Hiển thị bản đồ kiến thức với mức độ thành thạo:")
    knowledge_map = dashboard.display_knowledge_map(student_id, save=True)
    plt.show()
    
    # Hiển thị heatmap mức độ thành thạo
    print("\n2. Hiển thị heatmap mức độ thành thạo các khái niệm:")
    heatmap = dashboard.display_mastery_heatmap(student_id, save=True)
    plt.show()
    
    # Lấy kế hoạch ôn tập
    print("\n3. Tạo kế hoạch ôn tập Active Recall:")
    dashboard.display_study_plan(student_id, plan_type="active_recall", days=7)
    
    # Dự đoán kết quả
    print("\n4. Dự đoán kết quả học sinh cho một câu hỏi:")
    question_ids = list(mappings["question2idx"].keys())
    # question_id = question_ids[0]
    question_id = 9
    prediction = dashboard.display_prediction(student_id, question_id, save=True)
    plt.show()
    
    # Mô phỏng quá trình học trong tương lai
    print("\n5. Mô phỏng quá trình học trong tương lai:")
    future_questions = question_ids[1:6]  # Lấy 5 câu hỏi kế tiếp
    simulation = dashboard.simulate_learning_path(student_id, future_questions)
    plt.show()
    
    print("\nĐã hoàn tất demo!")

if __name__ == "__main__":
    main() 