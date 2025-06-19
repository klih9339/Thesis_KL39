import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime, timedelta

class ActiveRecallPlanner:
    def __init__(self, model, question_df, question_concept_df, kc_df, kc2idx, question2idx):
        """
        Bộ lập kế hoạch ôn tập theo Active Recall
        
        Args:
            model: mô hình DCRKT đã được huấn luyện
            question_df: dataframe chứa thông tin các câu hỏi
            question_concept_df: dataframe chứa mapping giữa câu hỏi và concept
            kc_df: dataframe chứa thông tin về knowledge component
            kc2idx: dictionary ánh xạ từ concept id thực tế tới chỉ số nội bộ
            question2idx: dictionary ánh xạ từ question id thực tế tới chỉ số nội bộ
        """
        self.model = model
        self.question_df = question_df
        self.question_concept_df = question_concept_df
        self.kc_df = kc_df
        self.kc2idx = kc2idx
        self.question2idx = question2idx
        self.idx2question = {v: k for k, v in question2idx.items()}
        
        # Tạo mapping giữa concept và câu hỏi
        self.concept_to_questions = self._create_concept_question_mapping()
        
    def _create_concept_question_mapping(self):
        """Tạo mapping từ concept tới danh sách câu hỏi"""
        concept_to_questions = defaultdict(list)
        
        for _, row in self.question_concept_df.iterrows():
            qid = row["question_id"]
            cid = row["knowledgecomponent_id"]
            
            if qid in self.question2idx and cid in self.kc2idx:
                concept_to_questions[self.kc2idx[cid]].append(self.question2idx[qid])
                
        return concept_to_questions
        
    def get_weak_concepts(self, student_id, threshold=0.33, top_n=5):
        """Lấy các concept yếu nhất của học sinh dựa trên ngưỡng và số lượng"""
        snapshot = self.model.get_snapshot(student_id)
        
        if snapshot is None:
            return []
            
        # Tính mức độ thành thạo
        concept_mastery = []
        for idx in range(len(snapshot)):
            mastery = snapshot[idx].norm().item()
            concept_mastery.append((idx, mastery))
            
        # Chuẩn hóa
        if concept_mastery:
            max_mastery = max(m for _, m in concept_mastery)
            normalized_mastery = [(idx, m / max_mastery) for idx, m in concept_mastery]
            
            # Lọc ra các concept yếu
            weak_concepts = [(idx, m) for idx, m in normalized_mastery if m < threshold]
            
            # Sắp xếp theo mức độ thành thạo tăng dần
            weak_concepts.sort(key=lambda x: x[1])
            
            # Trả về top_n concept yếu nhất
            return weak_concepts[:top_n]
        
        return []
        
    def get_concept_questions(self, concept_idx, n=3):
        """Lấy n câu hỏi cho một concept"""
        question_idxs = self.concept_to_questions.get(concept_idx, [])
        
        if not question_idxs:
            return []
            
        # Lấy ngẫu nhiên n câu hỏi
        n = min(n, len(question_idxs))
        selected_idxs = np.random.choice(question_idxs, size=n, replace=False)
        
        questions = []
        for idx in selected_idxs:
            q_id = self.idx2question[idx]
            q_text = self.question_df[self.question_df["id"] == q_id]["question_text"].values[0]
            questions.append({
                "question_idx": idx,
                "question_id": q_id,
                "text": q_text
            })
            
        return questions
        
    def create_active_recall_plan(self, student_id, days=7, questions_per_concept=3):
        """Tạo kế hoạch ôn tập Active Recall cho học sinh"""
        weak_concepts = self.get_weak_concepts(student_id)
        
        if not weak_concepts:
            return None
            
        # Tạo lịch ôn tập
        today = datetime.now()
        schedule = []
        
        for day in range(days):
            day_date = today + timedelta(days=day)
            day_concepts = weak_concepts[day % len(weak_concepts):] + weak_concepts[:day % len(weak_concepts)]
            day_concepts = day_concepts[:min(3, len(day_concepts))]
            
            day_schedule = {
                "date": day_date.strftime("%Y-%m-%d"),
                "concepts": []
            }
            
            for concept_idx, mastery in day_concepts:
                concept_id = self.kc_df.iloc[concept_idx]["id"] if concept_idx < len(self.kc_df) else None
                concept_name = self.kc_df[self.kc_df["id"] == concept_id]["name"].values[0] if concept_id else f"Concept {concept_idx}"
                
                questions = self.get_concept_questions(concept_idx, questions_per_concept)
                
                day_schedule["concepts"].append({
                    "concept_idx": concept_idx,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "mastery": mastery,
                    "questions": questions
                })
                
            schedule.append(day_schedule)
            
        return schedule
        
    def generate_spaced_repetition_schedule(self, student_id, initial_interval=1, 
                                           multiplier=2, max_interval=30, duration_days=90):
        """
        Tạo lịch ôn tập theo nguyên lý spaced repetition
        
        Args:
            student_id: ID của học sinh
            initial_interval: Khoảng cách ban đầu (ngày)
            multiplier: Hệ số nhân khoảng cách
            max_interval: Khoảng cách tối đa (ngày)
            duration_days: Tổng số ngày của kế hoạch
        """
        weak_concepts = self.get_weak_concepts(student_id, threshold=0.5, top_n=10)
        
        if not weak_concepts:
            return None
            
        today = datetime.now()
        schedule = defaultdict(list)
        
        for concept_idx, mastery in weak_concepts:
            concept_id = self.kc_df.iloc[concept_idx]["id"] if concept_idx < len(self.kc_df) else None
            concept_name = self.kc_df[self.kc_df["id"] == concept_id]["name"].values[0] if concept_id else f"Concept {concept_idx}"
            
            # Điều chỉnh interval ban đầu dựa trên mức độ thành thạo
            adjusted_initial = max(1, int(initial_interval * (1 - mastery)))
            interval = adjusted_initial
            current_day = 0
            
            while current_day < duration_days:
                current_date = today + timedelta(days=current_day)
                date_str = current_date.strftime("%Y-%m-%d")
                
                questions = self.get_concept_questions(concept_idx, 2)
                
                schedule[date_str].append({
                    "concept_idx": concept_idx,
                    "concept_name": concept_name,
                    "mastery": mastery,
                    "questions": questions
                })
                
                # Tăng interval theo nguyên lý spaced repetition
                interval = min(interval * multiplier, max_interval)
                current_day += interval
                
        # Chuyển dictionary thành list theo ngày
        formatted_schedule = []
        for day in sorted(schedule.keys()):
            formatted_schedule.append({
                "date": day,
                "concepts": schedule[day]
            })
            
        return formatted_schedule
        
    def predict_student_performance(self, student_id, question_id):
        """
        Dự đoán khả năng học sinh trả lời đúng câu hỏi
        
        Args:
            student_id: ID của học sinh
            question_id: ID của câu hỏi
        
        Returns:
            prob: Xác suất học sinh trả lời đúng
            relevant_concepts: Các khái niệm liên quan đến câu hỏi
        """
        # Chuyển từ ID thực tế sang chỉ số nội bộ
        if question_id not in self.question2idx:
            return None, []
            
        q_idx = self.question2idx[question_id]
        
        # Lấy các concept liên quan đến câu hỏi
        related_concepts = []
        for _, row in self.question_concept_df[self.question_concept_df["question_id"] == question_id].iterrows():
            kc_id = row["knowledgecomponent_id"]
            if kc_id in self.kc2idx:
                kc_idx = self.kc2idx[kc_id]
                kc_name = self.kc_df[self.kc_df["id"] == kc_id]["name"].values[0]
                related_concepts.append({
                    "concept_idx": kc_idx,
                    "concept_id": kc_id,
                    "concept_name": kc_name
                })
        
        # Lấy snapshot hiện tại của học sinh
        snapshot = self.model.get_snapshot(student_id)
        if snapshot is None:
            return 0.5, related_concepts  # Mặc định 50% nếu không có dữ liệu
        
        # Tính toán dự đoán
        concept_ids = [c["concept_idx"] for c in related_concepts]
        if not concept_ids:
            return 0.5, related_concepts
            
        # Thực hiện dự đoán
        try:
            with torch.no_grad():
                pred = self.model.predictor(
                    self.model.question_emb(torch.tensor(q_idx)),
                    self.model.memory_key,
                    snapshot
                )
            return pred.item(), related_concepts
        except:
            return 0.5, related_concepts  # Fallback
            
    def simulate_future_learning(self, student_id, question_ids, timestamps=None):
        """
        Mô phỏng quá trình học trong tương lai của học sinh
        
        Args:
            student_id: ID của học sinh
            question_ids: Danh sách ID câu hỏi học sinh sẽ làm
            timestamps: Danh sách thời gian tương ứng (Unix timestamp)
        
        Returns:
            predictions: Danh sách dự đoán kết quả
            snapshots: Danh sách snapshot kiến thức sau mỗi câu hỏi
        """
        # Clone model memory hiện tại để không ảnh hưởng tới model gốc
        original_memory = self.model.get_snapshot(student_id)
        
        if original_memory is None:
            return [], []
            
        # Tạo memory tạm thời
        temp_memory = original_memory.clone()
        self.model.student_memory[student_id] = temp_memory
        
        # Nếu không có timestamps, tạo timestamps mặc định
        if timestamps is None:
            current_time = datetime.now().timestamp()
            timestamps = [current_time + i * 3600 for i in range(len(question_ids))]  # Mỗi câu cách nhau 1 giờ
            
        # Tạo lịch sử snapshot
        predictions = []
        snapshots = []
        
        # Lưu snapshots ban đầu
        snapshots.append(temp_memory.clone())
        
        # Mô phỏng học từng câu
        for i, q_id in enumerate(question_ids):
            if q_id not in self.question2idx:
                continue
                
            q_idx = self.question2idx[q_id]
            timestamp = timestamps[i]
            
            # Lấy concepts liên quan
            concept_ids = []
            for _, row in self.question_concept_df[self.question_concept_df["question_id"] == q_id].iterrows():
                kc_id = row["knowledgecomponent_id"]
                if kc_id in self.kc2idx:
                    concept_ids.append(self.kc2idx[kc_id])
                    
            # Dự đoán kết quả
            try:
                pred, _ = self.predict_student_performance(student_id, q_id)
                predictions.append(pred)
                
                # Mô phỏng student trả lời câu hỏi và cập nhật memory
                # Giả định student trả lời đúng với xác suất = pred
                is_correct = float(np.random.random() < pred)
                
                # Forward để cập nhật memory
                self.model.forward_single_step(
                    student_id=student_id,
                    q_idx=torch.tensor(q_idx),
                    o_idx=torch.tensor(0),  # Dummy value
                    u_idx=torch.tensor(0),  # Dummy value
                    score=torch.tensor(is_correct),
                    timestamp=torch.tensor(timestamp),
                    concept_ids=concept_ids
                )
                
                # Lưu snapshot mới
                snapshots.append(self.model.get_snapshot(student_id).clone())
            except Exception as e:
                print(f"Error in simulation: {e}")
                continue
                
        # Khôi phục memory ban đầu
        self.model.student_memory[student_id] = original_memory
        
        return predictions, snapshots 