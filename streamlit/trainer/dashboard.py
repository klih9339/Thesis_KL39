import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import datetime
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from .knowledge_map import KnowledgeMapVisualizer
from .active_recall import ActiveRecallPlanner

class KnowledgeDashboard:
    def __init__(self, model, dataframes, mappings):
        """
        Dashboard để hiển thị và tương tác với bản đồ kiến thức
        
        Args:
            model: Model DCRKT đã huấn luyện
            dataframes: Dictionary chứa các dataframe (question_df, option_df, concept_df, etc.)
            mappings: Dictionary chứa các ánh xạ (user2idx, question2idx, kc2idx, etc.)
        """
        self.model = model
        self.dfs = dataframes
        self.mappings = mappings
        
        # Khởi tạo bộ trực quan hóa và lập kế hoạch
        self.visualizer = KnowledgeMapVisualizer(
            model=model,
            kc_list=mappings["kc_list"],
            kc_df=dataframes["concept_df"],
            relation_df=dataframes["relation_df"],
            kc2idx=mappings["kc2idx"]
        )
        
        self.planner = ActiveRecallPlanner(
            model=model,
            question_df=dataframes["question_df"],
            question_concept_df=dataframes["question_concept_df"],
            kc_df=dataframes["concept_df"],
            kc2idx=mappings["kc2idx"],
            question2idx=mappings["question2idx"]
        )
        
        # Ánh xạ ngược
        self.idx2user = {v: k for k, v in mappings["user2idx"].items()}
        self.idx2question = {v: k for k, v in mappings["question2idx"].items()}
        
        # Tạo thư mục lưu hình ảnh nếu chưa tồn tại
        os.makedirs("results/knowledge_maps", exist_ok=True)
        
    def display_knowledge_map(self, student_id, save=False, filename=None):
        """Hiển thị bản đồ kiến thức của học sinh"""
        # Chuyển đổi sang student_idx nếu cần
        student_idx = student_id
        if student_id in self.mappings["user2idx"]:
            student_idx = self.mappings["user2idx"][student_id]
            
        # Hiển thị bản đồ kiến thức
        fig = self.visualizer.visualize_knowledge_map(student_idx)
        
        if fig is None:
            print(f"Không có dữ liệu cho học sinh {student_id}")
            return
        
        if save:
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/knowledge_maps/student_{student_id}_knowledge_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Đã lưu bản đồ kiến thức tại: {filename}")
            
        return fig
        
    def display_mastery_heatmap(self, student_id, save=False, filename=None):
        """Hiển thị heatmap mức độ thành thạo các khái niệm của học sinh"""
        student_idx = student_id
        if student_id in self.mappings["user2idx"]:
            student_idx = self.mappings["user2idx"][student_id]
            
        fig = self.visualizer.create_concept_mastery_heatmap(student_idx)
        
        if fig is None:
            print(f"Không có dữ liệu cho học sinh {student_id}")
            return
            
        if save:
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/knowledge_maps/student_{student_id}_mastery_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Đã lưu heatmap tại: {filename}")
            
        return fig
        
    def get_study_plan(self, student_id, plan_type="active_recall", days=7):
        """Tạo kế hoạch ôn tập cho học sinh"""
        student_idx = student_id
        if student_id in self.mappings["user2idx"]:
            student_idx = self.mappings["user2idx"][student_id]
            
        if plan_type == "active_recall":
            return self.planner.create_active_recall_plan(student_idx, days=days)
        elif plan_type == "spaced_repetition":
            return self.planner.generate_spaced_repetition_schedule(student_idx, duration_days=days)
        else:
            print(f"Loại kế hoạch không hợp lệ: {plan_type}")
            return None
            
    def display_study_plan(self, student_id, plan_type="active_recall", days=7):
        """Hiển thị kế hoạch ôn tập dưới dạng bảng"""
        plan = self.get_study_plan(student_id, plan_type, days)
        
        if plan is None:
            print(f"Không thể tạo kế hoạch ôn tập cho học sinh {student_id}")
            return
            
        for day in plan:
            date = day["date"]
            print(f"\n=== NGÀY {date} ===")
            print("-" * 80)
            
            for concept in day["concepts"]:
                name = concept["concept_name"]
                mastery = concept.get("mastery", 0)
                print(f"Khái niệm: {name} (Mức độ thành thạo: {mastery:.2f})")
                
                for i, q in enumerate(concept["questions"]):
                    print(f"  Câu {i+1}: {q['text'][:100]}...")
                    
                print("-" * 80)
                
    def predict_question_performance(self, student_id, question_id):
        """Dự đoán khả năng học sinh trả lời đúng câu hỏi"""
        student_idx = student_id
        if student_id in self.mappings["user2idx"]:
            student_idx = self.mappings["user2idx"][student_id]
            
        q_id = question_id
        if question_id in self.mappings["question2idx"]:
            q_id = question_id
        elif isinstance(question_id, int) and question_id in self.idx2question:
            q_id = self.idx2question[question_id]
            
        prob, concepts = self.planner.predict_student_performance(student_idx, q_id)
        
        if prob is None:
            print(f"Không thể dự đoán cho học sinh {student_id} và câu hỏi {question_id}")
            return None
            
        # Format kết quả
        result = {
            "student_id": student_id,
            "question_id": q_id,
            "probability": prob,
            "concepts": concepts
        }
        
        # Hiển thị thông tin câu hỏi
        try:
            q_text = self.dfs["question_df"][self.dfs["question_df"]["id"] == q_id]["question_text"].values[0]
            result["question_text"] = q_text
        except:
            result["question_text"] = f"Câu hỏi ID: {q_id}"
            
        return result
        
    def display_prediction(self, student_id, question_id, save=False, filename=None):
        """Hiển thị dự đoán và thông tin liên quan"""
        result = self.predict_question_performance(student_id, question_id)
        
        if result is None:
            return
            
        # Hiển thị dự đoán
        prob = result["probability"]
        print(f"Câu hỏi: {result['question_text'][:150]}...")
        print(f"Xác suất trả lời đúng: {prob:.2%}")
        
        # Hiển thị các khái niệm liên quan
        print("\nCác khái niệm liên quan:")
        for concept in result["concepts"]:
            print(f"- {concept['concept_name']}")
            
        # Tạo biểu đồ dự đoán
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Màu sắc dựa trên xác suất
        if prob < 0.33:
            color = '#FF9999'  # Đỏ nhạt
            level = "Thấp"
        elif prob < 0.66:
            color = '#FFCC99'  # Cam nhạt
            level = "Trung bình"
        else:
            color = '#99FF99'  # Xanh lá nhạt
            level = "Cao"
            
        # Vẽ biểu đồ cột
        ax.bar(['Dự đoán'], [prob], color=color, width=0.4)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Thêm nhãn
        ax.text(0, prob + 0.05, f"{prob:.2%}", ha='center', va='bottom', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Xác suất trả lời đúng')
        ax.set_title(f'Dự đoán kết quả học sinh {student_id} cho câu hỏi {question_id}')
        
        # Thêm nhãn mức độ
        ax.text(0.85, 0.9, f"Mức độ: {level}", transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/knowledge_maps/student_{student_id}_question_{question_id}_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Đã lưu dự đoán tại: {filename}")
            
        return fig
        
    def simulate_learning_path(self, student_id, question_ids, visualize=True):
        """Mô phỏng quá trình học của học sinh và hiển thị sự thay đổi mức độ thành thạo"""
        student_idx = student_id
        if student_id in self.mappings["user2idx"]:
            student_idx = self.mappings["user2idx"][student_id]
            
        # Chuẩn bị danh sách câu hỏi
        q_ids = []
        for qid in question_ids:
            if qid in self.mappings["question2idx"]:
                q_ids.append(qid)
            elif isinstance(qid, int) and qid in self.idx2question:
                q_ids.append(self.idx2question[qid])
                
        if not q_ids:
            print("Không có câu hỏi hợp lệ để mô phỏng")
            return None
            
        # Thực hiện mô phỏng
        predictions, snapshots = self.planner.simulate_future_learning(student_idx, q_ids)
        
        if not predictions or not snapshots:
            print("Mô phỏng không thành công")
            return None
            
        # Nếu không cần biểu đồ, trả về kết quả mô phỏng
        if not visualize:
            return {
                "predictions": predictions,
                "snapshots": snapshots
            }
            
        # Tạo biểu đồ mô phỏng
        mastery_over_time = []
        
        # Lấy danh sách concept
        concept_df = self.visualizer.get_concept_mastery_data(student_idx)
        if concept_df is None or concept_df.empty:
            print("Không có dữ liệu concept cho học sinh")
            return None
            
        # Lấy top 5 concept có mastery thấp nhất
        top_concepts = concept_df.sort_values("mastery").head(5)
        concept_indices = top_concepts["concept_idx"].tolist()
        concept_names = top_concepts["concept_name"].tolist()
        
        # Tính mastery theo thời gian cho mỗi concept
        for concept_idx in concept_indices:
            concept_mastery = []
            for snapshot in snapshots:
                mastery = snapshot[concept_idx].norm().item()
                concept_mastery.append(mastery)
            mastery_over_time.append(concept_mastery)
            
        # Vẽ biểu đồ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Biểu đồ dự đoán
        x = range(len(predictions))
        ax1.plot(x, predictions, 'o-', color='blue')
        ax1.set_xlabel('Số thứ tự câu hỏi')
        ax1.set_ylabel('Xác suất trả lời đúng')
        ax1.set_title('Dự đoán xác suất trả lời đúng theo thời gian')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 1)
        
        # Biểu đồ mastery
        for i, mastery in enumerate(mastery_over_time):
            ax2.plot(range(len(mastery)), mastery, 'o-', label=concept_names[i][:20])
        ax2.set_xlabel('Số thứ tự câu hỏi')
        ax2.set_ylabel('Mức độ thành thạo')
        ax2.set_title('Sự thay đổi mức độ thành thạo theo thời gian')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        return fig
        
    def interactive_dashboard(self):
        """Tạo dashboard tương tác với ipywidgets"""
        # Danh sách học sinh và câu hỏi
        student_list = sorted(self.idx2user.values())
        question_list = sorted(self.idx2question.values())
        
        # Tạo dropdown cho học sinh và câu hỏi
        student_dropdown = widgets.Dropdown(
            options=student_list,
            description='Học sinh:',
            value=student_list[0] if student_list else None,
            style={'description_width': 'initial'}
        )
        
        question_dropdown = widgets.Dropdown(
            options=question_list,
            description='Câu hỏi:',
            value=question_list[0] if question_list else None,
            style={'description_width': 'initial'}
        )
        
        # Tạo các button chức năng
        knowledge_map_btn = widgets.Button(description='Hiển thị bản đồ kiến thức')
        mastery_heatmap_btn = widgets.Button(description='Hiển thị mức độ thành thạo')
        prediction_btn = widgets.Button(description='Dự đoán kết quả')
        study_plan_btn = widgets.Button(description='Tạo kế hoạch ôn tập')
        
        # Tạo dropdown cho loại kế hoạch
        plan_type_dropdown = widgets.Dropdown(
            options=[('Active Recall', 'active_recall'), ('Spaced Repetition', 'spaced_repetition')],
            description='Loại kế hoạch:',
            value='active_recall',
            style={'description_width': 'initial'}
        )
        
        # Tạo slider cho số ngày
        days_slider = widgets.IntSlider(
            value=7,
            min=1,
            max=30,
            step=1,
            description='Số ngày:',
            style={'description_width': 'initial'}
        )
        
        # Output widget để hiển thị kết quả
        output = widgets.Output()
        
        # Xử lý sự kiện button
        def on_knowledge_map_clicked(b):
            with output:
                clear_output(wait=True)
                student_id = student_dropdown.value
                if student_id:
                    self.display_knowledge_map(student_id)
                else:
                    print("Vui lòng chọn học sinh")
                    
        def on_mastery_heatmap_clicked(b):
            with output:
                clear_output(wait=True)
                student_id = student_dropdown.value
                if student_id:
                    self.display_mastery_heatmap(student_id)
                else:
                    print("Vui lòng chọn học sinh")
                    
        def on_prediction_clicked(b):
            with output:
                clear_output(wait=True)
                student_id = student_dropdown.value
                question_id = question_dropdown.value
                if student_id and question_id:
                    self.display_prediction(student_id, question_id)
                else:
                    print("Vui lòng chọn học sinh và câu hỏi")
                    
        def on_study_plan_clicked(b):
            with output:
                clear_output(wait=True)
                student_id = student_dropdown.value
                plan_type = plan_type_dropdown.value
                days = days_slider.value
                if student_id:
                    self.display_study_plan(student_id, plan_type, days)
                else:
                    print("Vui lòng chọn học sinh")
                    
        # Gán sự kiện
        knowledge_map_btn.on_click(on_knowledge_map_clicked)
        mastery_heatmap_btn.on_click(on_mastery_heatmap_clicked)
        prediction_btn.on_click(on_prediction_clicked)
        study_plan_btn.on_click(on_study_plan_clicked)
        
        # Tạo layout
        student_section = widgets.VBox([student_dropdown])
        question_section = widgets.VBox([question_dropdown])
        plan_section = widgets.VBox([plan_type_dropdown, days_slider])
        
        button_row1 = widgets.HBox([knowledge_map_btn, mastery_heatmap_btn])
        button_row2 = widgets.HBox([prediction_btn, study_plan_btn])
        
        dashboard = widgets.VBox([
            widgets.HBox([student_section, question_section, plan_section]),
            button_row1,
            button_row2,
            output
        ])
        
        return dashboard 