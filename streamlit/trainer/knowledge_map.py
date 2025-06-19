import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
from collections import defaultdict

class KnowledgeMapVisualizer:
    def __init__(self, model, kc_list, kc_df, relation_df, kc2idx):
        """
        Khởi tạo bộ trực quan hóa bản đồ kiến thức
        
        Args:
            model: mô hình DCRKT đã được huấn luyện
            kc_list: danh sách các knowledge component id
            kc_df: dataframe chứa thông tin về knowledge component
            relation_df: dataframe chứa thông tin về mối quan hệ giữa các knowledge component
            kc2idx: dictionary ánh xạ từ id thực tế tới chỉ số nội bộ
        """
        self.model = model
        self.kc_list = kc_list
        self.kc_df = kc_df
        self.relation_df = relation_df
        self.kc2idx = kc2idx
        self.idx2kc = {v: k for k, v in kc2idx.items()}
        
        # Tạo bản đồ kiến thức cơ bản
        self.create_base_knowledge_graph()
        
    def create_base_knowledge_graph(self):
        """Tạo đồ thị cơ sở dựa trên mối quan hệ giữa các knowledge component"""
        self.G = nx.DiGraph()
        
        # Thêm tất cả các knowledge component vào đồ thị
        for idx, kc_id in enumerate(self.kc_list):
            try:
                kc_name = self.kc_df[self.kc_df["id"] == kc_id]["name"].values[0]
                # Thêm node với thông tin tên và id
                self.G.add_node(idx, name=kc_name, id=kc_id)
            except:
                self.G.add_node(idx, name=f"KC_{kc_id}", id=kc_id)
        
        # Thêm edges từ relation_df
        for _, row in self.relation_df.iterrows():
            src_id = row["from_knowledgecomponent_id"]
            tar_id = row["to_knowledgecomponent_id"]
            
            if src_id in self.kc2idx and tar_id in self.kc2idx:
                src_idx = self.kc2idx[src_id]
                tar_idx = self.kc2idx[tar_id]
                self.G.add_edge(src_idx, tar_idx)
    
    def get_mastery_level_colors(self, mastery_values):
        """Phân nhóm và tạo màu cho mức độ thành thạo"""
        # Tạo scaler để chuẩn hóa giá trị
        scaler = MinMaxScaler()
        if isinstance(mastery_values, torch.Tensor):
            normalized = scaler.fit_transform(mastery_values.cpu().numpy().reshape(-1, 1)).flatten()
        else:
            normalized = scaler.fit_transform(np.array(mastery_values).reshape(-1, 1)).flatten()
        
        # Phân loại thành 3 nhóm: thấp, trung bình, cao
        colors = []
        levels = []
        
        for value in normalized:
            if value < 0.33:
                colors.append('#FF9999')  # Đỏ nhạt - mức thấp
                levels.append("Thấp")
            elif value < 0.66:
                colors.append('#FFCC99')  # Cam nhạt - mức trung bình
                levels.append("Trung bình")
            else:
                colors.append('#99FF99')  # Xanh lá nhạt - mức cao
                levels.append("Cao")
                
        return colors, levels, normalized
    
    def visualize_knowledge_map(self, student_id, title=None, figsize=(12, 10)):
        """Hiển thị bản đồ kiến thức của học sinh với mức độ thành thạo"""
        # Lấy snapshot mới nhất từ model cho học sinh
        snapshot = self.model.get_snapshot(student_id)
        
        if snapshot is None:
            print(f"Không có dữ liệu cho học sinh {student_id}")
            return None
            
        # Tính mức độ thành thạo dựa trên norm của memory value
        mastery_values = [snapshot[idx].norm().item() for idx in range(len(self.kc_list))]
        node_colors, mastery_levels, normalized_values = self.get_mastery_level_colors(mastery_values)
        
        # Chuẩn bị dữ liệu cho visualization
        node_sizes = [300 + 1000 * val for val in normalized_values]
        
        # Tạo layout cho đồ thị
        pos = nx.spring_layout(self.G, seed=42)
        
        # Vẽ đồ thị
        plt.figure(figsize=figsize)
        
        # Vẽ edges
        nx.draw_networkx_edges(self.G, pos, alpha=0.3, arrows=True, 
                               arrowstyle='->', arrowsize=10)
        
        # Vẽ nodes
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, 
                               node_size=node_sizes, alpha=0.8)
        
        # Thêm labels
        labels = {}
        for node in self.G.nodes():
            labels[node] = self.G.nodes[node]['name']
            
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, 
                                font_family='sans-serif')
        
        # Thêm title và legend
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(f"Bản đồ kiến thức của học sinh {student_id}", fontsize=16)
            
        # Tạo legend cho các mức độ thành thạo
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9999', markersize=10, label='Thấp'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFCC99', markersize=10, label='Trung bình'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99FF99', markersize=10, label='Cao')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_concept_mastery_data(self, student_id):
        """Lấy dữ liệu về mức độ thành thạo của học sinh theo concept"""
        snapshot = self.model.get_snapshot(student_id)
        
        if snapshot is None:
            return None
            
        # Tạo dataframe chứa thông tin mức độ thành thạo
        data = []
        for idx in range(len(self.kc_list)):
            try:
                kc_id = self.kc_list[idx]
                name = self.kc_df[self.kc_df["id"] == kc_id]["name"].values[0]
                mastery = snapshot[idx].norm().item()
                
                data.append({
                    "concept_idx": idx,
                    "concept_id": kc_id,
                    "concept_name": name,
                    "mastery": mastery
                })
            except:
                pass
                
        df = pd.DataFrame(data)
        if not df.empty:
            df["mastery_normalized"] = MinMaxScaler().fit_transform(df[["mastery"]])
            
            # Phân loại mức độ thành thạo
            conditions = [
                (df["mastery_normalized"] < 0.33),
                (df["mastery_normalized"] < 0.66),
                (df["mastery_normalized"] >= 0.66)
            ]
            choices = ["Thấp", "Trung bình", "Cao"]
            df["mastery_level"] = np.select(conditions, choices, default="Không xác định")
            
        return df
        
    def create_concept_mastery_heatmap(self, student_id, figsize=(12, 8)):
        """Tạo heatmap về mức độ thành thạo các khái niệm của học sinh"""
        df = self.get_concept_mastery_data(student_id)
        
        if df is None or df.empty:
            print(f"Không có dữ liệu cho học sinh {student_id}")
            return None
            
        # Sắp xếp theo mức độ thành thạo
        df = df.sort_values("mastery", ascending=False)
        
        # Tạo bảng màu cho heatmap
        cmap = LinearSegmentedColormap.from_list(
            "mastery_cmap", ["#FF9999", "#FFCC99", "#99FF99"]
        )
        
        # Giới hạn số lượng concept hiển thị
        display_limit = min(20, len(df))
        df_display = df.head(display_limit)
        
        # Tạo heatmap
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            df_display[["mastery_normalized"]].T, 
            cmap=cmap,
            cbar_kws={"label": "Mức độ thành thạo"},
            xticklabels=df_display["concept_name"],
            yticklabels=["Thành thạo"],
            vmin=0, vmax=1
        )
        
        # Xoay labels
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        plt.title(f"Mức độ thành thạo các khái niệm của học sinh {student_id}")
        plt.tight_layout()
        
        return plt.gcf() 