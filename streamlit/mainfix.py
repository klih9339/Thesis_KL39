import pandas as pd
import numpy as np
import networkx as nx
import os
from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import (Embedding, Sequential, Linear, Sigmoid, ReLU, Dropout, LayerNorm)
from torch_geometric.nn import GAT
from torch_geometric.utils import dense_to_sparse, to_edge_index, add_self_loops
from collections import defaultdict
from argparse import Namespace
from itertools import chain
from time import time
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import pickle
import openai
from datetime import datetime, timedelta
import math
import torch.serialization
import argparse

folder_path = 'input/raw'

question_df = pd.read_csv(f"{folder_path}/Questions.csv")
option_df = pd.read_csv(f"{folder_path}/Question_Choices.csv")
concept_df = pd.read_csv(f"{folder_path}/KCs.csv")
relation_df = pd.read_csv(f"{folder_path}/KC_Relationships.csv")
question_concept_df = pd.read_csv(f"{folder_path}/Question_KC_Relationships.csv")
interaction_df = pd.read_csv(f"{folder_path}/Transaction.csv")
specialization_df = pd.read_csv(f"{folder_path}/Specialization.csv")
student_specialization_df = pd.read_csv(f"{folder_path}/Student_Specialization.csv")

# ----------------------- TIỀN XỬ LÝ -----------------------
def clean_data():
    global question_df, option_df, concept_df, relation_df, question_concept_df, interaction_df

    question_df.drop_duplicates(subset=["id"], inplace=True)
    question_concept_df.drop_duplicates(subset=["question_id", "knowledgecomponent_id"], inplace=True)
    concept_df.drop_duplicates(subset=["id"], inplace=True)
    option_df.drop_duplicates(subset=["id", "question_id"], inplace=True)
    interaction_df.drop_duplicates(subset=["student_id", "question_id", "start_time"], inplace=True)
    relation_df.drop_duplicates(subset=["from_knowledgecomponent_id", "to_knowledgecomponent_id"], inplace=True)

    concept_df["name"] = concept_df["name"].fillna("Unknown Concept")
    question_df["question_text"] = question_df["question_text"].fillna("")
    option_df["choice_text"] = option_df["choice_text"].fillna("")

    # Xử lý thời gian
    if not pd.api.types.is_datetime64_any_dtype(interaction_df["start_time"]):
        interaction_df["start_time"] = pd.to_datetime(interaction_df["start_time"], errors="coerce")

    interaction_df.dropna(subset=["start_time"], inplace=True)
    if pd.api.types.is_datetime64tz_dtype(interaction_df["start_time"]):
        interaction_df["start_time"] = interaction_df["start_time"].dt.tz_localize(None)

    before = len(interaction_df)
    interaction_df.dropna(subset=["student_id", "question_id", "answer_state"], inplace=True)
    after = len(interaction_df)

    print(f"[clean_data] Số dòng interaction còn lại: {after} (loại {before - after})")

# ----------------------- ÁNH XẠ ID -----------------------
def map_ids():
    global user2idx, question2idx, kc2idx, user_list, question_list, kc_list

    user_list = sorted(interaction_df["student_id"].unique())
    question_list = sorted(question_df["id"].unique())
    kc_list = sorted(concept_df["id"].unique())

    user2idx = {u: i for i, u in enumerate(user_list)}
    question2idx = {q: i for i, q in enumerate(question_list)}
    kc2idx = {k: i for i, k in enumerate(kc_list)}

    print(f"[map_ids] Số user: {len(user2idx)}, question: {len(question2idx)}, KC: {len(kc2idx)}")

# ----------------------- TẠO EVENTS -----------------------
def process_events():
    global interaction_df

    interaction_df["user_idx"] = interaction_df["student_id"].map(user2idx)
    interaction_df["question_idx"] = interaction_df["question_id"].map(question2idx)
    interaction_df["timestamp"] = interaction_df["start_time"].apply(lambda x: int(x.timestamp()))
    interaction_df["is_correct"] = interaction_df["answer_state"].astype(int)

# ----------------------- MAP CÂU TRẢ LỜI -----------------------
def map_answers():
    global interaction_df

    option_df_sorted = option_df.sort_values(["question_id", "id"])
    option_map = option_df_sorted.groupby("question_id")["id"].apply(list).to_dict()

    def convert_answer(qid, aid):
        if qid not in option_map or pd.isna(aid):
            return None
        try:
            return option_map[qid].index(aid)
        except ValueError:
            return None

    interaction_df["answer_idx"] = interaction_df.apply(
        lambda x: convert_answer(x["question_id"], x["answer_choice_id"]), axis=1
    )

# ----------------------- MAP CONCEPT -----------------------
def map_concepts():
    global interaction_df

    qkc_map = question_concept_df.groupby("question_id")["knowledgecomponent_id"].apply(list).to_dict()

    interaction_df["concept_idxs"] = interaction_df["question_id"].apply(
        lambda qid: [kc2idx[k] for k in qkc_map.get(qid, []) if k in kc2idx]
    )

    interaction_df = interaction_df[interaction_df["concept_idxs"].map(len) > 0]

# ----------------------- SORT EVENTS -----------------------
def sort_events():
    global interaction_df
    interaction_df.sort_values(["user_idx", "timestamp"], inplace=True)
    interaction_df.reset_index(drop=True, inplace=True)

# ----------------------- TẠO GRAPH KIẾN THỨC -----------------------
def create_concept_graph(relation_df, kc2idx, num_c):
    relation_df["src_idx"] = relation_df["from_knowledgecomponent_id"].map(kc2idx)
    relation_df["tar_idx"] = relation_df["to_knowledgecomponent_id"].map(kc2idx)
    rel_df = relation_df.dropna()

    edges = list(zip(rel_df["src_idx"], rel_df["tar_idx"]))
    filtered_edges = set()

    for s, t in edges:
        if s == t:
            continue
        if (t, s) in filtered_edges:
            continue
        filtered_edges.add((s, t))
    directed = list(filtered_edges)

    knowledge_mask = torch.zeros((num_c, num_c))
    for s, t in directed:
        if 0 <= s < num_c and 0 <= t < num_c:
            knowledge_mask[s, t] = 1
    concept_graph = {"directed": directed}
    print(f"[create_concept_graph] Directed: {len(directed)}")
    return concept_graph, knowledge_mask

# ----------------------- TẠO SNAPSHOT -----------------------
def create_snapshots():
    snapshots = {}
    for uid, group in interaction_df.groupby("user_idx"):
        times = group["timestamp"].tolist()
        concepts = group["concept_idxs"].tolist()

        cum_concepts = set()
        snap = []
        for t, cset in zip(times, concepts):
            cum_concepts.update(cset)
            snap.append((t, list(cum_concepts)))
        snapshots[uid] = snap

    print(f"[create_snapshots] Số người dùng có snapshot: {len(snapshots)}")
    return snapshots

def create_knowledge_graph_dcrkt(snapshot_mv, concept_df, kc_list, snapshot_graph=None, threshold_sim=0.2, knowledge_mask=None, kc_mapping=None ):
    """
    Tạo bản đồ kiến thức từ snapshot Mv của học sinh tại 1 thời điểm cụ thể.

    Params:
    - snapshot_mv: Tensor [num_concepts, dim] – snapshot Mv từ model.get_snapshot(student_id, step)
    - concept_df: DataFrame chứa thông tin khái niệm (id, name)
    - kc_list: List[int] – danh sách ID khái niệm gốc tương ứng với chỉ số trong Mv
    - snapshot_graph: (edge_index, edge_weight) tuple (tùy chọn) từ DCRKT.snapshots_graph
    - threshold_sim: ngưỡng similarity để vẽ cạnh nếu không dùng snapshot_graph

    Returns:
    - G: networkx.DiGraph với node = concept, edge = lan truyền hiểu biết
    """
    G = nx.DiGraph()

    # Chuẩn hóa tên khái niệm
    idx_to_name = {}
    for idx, cid in enumerate(kc_list):
        row = concept_df[concept_df["id"] == cid]
        name = row["name"].values[0] if not row.empty else f"Concept {cid}"
        idx_to_name[idx] = name

    # Tính norm từng vector Mv → biểu thị độ hiểu biết
    mastery_norms = [vec.norm().item() for vec in snapshot_mv]

    # Thêm node
    for idx, norm in enumerate(mastery_norms):
        cid = kc_list[idx]
        cname = idx_to_name.get(idx, f"Concept {cid}")
        color = "red" if norm < 1.0 else "orange" if norm < 2.0 else "green"
        G.add_node(idx, concept_id=cid, name=cname, mastery=norm, color=color)
    # Thêm cạnh dựa trên snapshot_graph nếu có
    if snapshot_graph:
        edge_index, edge_weight = snapshot_graph
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            tgt = edge_index[1, i].item()
            if src == tgt:
                continue
            global_src = kc_mapping[src] if kc_mapping else src
            global_tgt = kc_mapping[tgt] if kc_mapping else tgt
            if knowledge_mask is not None and knowledge_mask[src, tgt].item() == 0:
                continue
            weight = edge_weight[i].item()
            G.add_edge(src, tgt, weight=weight, relation_type="propagated")
    else:
        # Tự tính cạnh dựa trên độ tương đồng
        normed = snapshot_mv / snapshot_mv.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sim_matrix = torch.matmul(normed, normed.T).cpu().numpy()
        for i in range(len(kc_list)):
            for j in range(i + 1, len(kc_list)):
                if sim_matrix[i][j] >= threshold_sim:
                      global_i = kc_mapping[i] if kc_mapping else i
                      global_j = kc_mapping[j] if kc_mapping else j
                      if knowledge_mask is None or knowledge_mask[i, j] == 1:
                          G.add_edge(i, j, weight=sim_matrix[i][j], relation_type="similarity")
                      if knowledge_mask is None or knowledge_mask[j, i] == 1:
                          G.add_edge(j, i, weight=sim_matrix[i][j], relation_type="similarity")
                    # G.add_edge(i, j, weight=sim_matrix[i][j], relation_type="similarity")
                    # G.add_edge(j, i, weight=sim_matrix[i][j], relation_type="similarity")

    return G

def visualize_student_knowledge_graph(student_id, step, model, show_prerequisite=False):
    mv = model.get_snapshot(student_id, step)
    snapshot_graph = model.snapshots_graph[student_id][step]
    G = create_knowledge_graph_dcrkt(mv, concept_df, kc_list, snapshot_graph=snapshot_graph, knowledge_mask=knowledge_mask)

    if show_prerequisite:
        _, prereq_mask = create_concept_graph(relation_df, kc2idx, num_c)
        for i in range(num_c):
            for j in range(num_c):
                if prereq_mask[i, j] == 1 and i in G.nodes and j in G.nodes:
                    G.add_edge(i, j, weight=1.0, relation_type="prerequisite")
    return G


# ----------------------- TẠO CONCEPT MEMORY -----------------------
def create_concept_memory():
    memory = {cid: {"Mk": None, "Mv": None} for cid in range(len(kc_list))}
    print("[create_concept_memory] Khởi tạo concept memory hoàn tất.")
    return memory

# ----------------------- HÀM CHÍNH -----------------------
def run_all_steps():
    clean_data()
    map_ids()
    num_c = len(kc2idx)
    relation_dict = defaultdict(set)
    for _, row in relation_df.iterrows():
        src = kc2idx.get(row["from_knowledgecomponent_id"])
        tgt = kc2idx.get(row["to_knowledgecomponent_id"])
        if src is not None and tgt is not None:
            relation_dict[src].add(tgt)
            relation_dict[tgt].add(src)

    process_events()
    map_answers()
    map_concepts()
    sort_events()
    concept_graph, knowledge_mask = create_concept_graph(relation_df, kc2idx, num_c)
    snapshots = create_snapshots()
    concept_memory = create_concept_memory()

    # Đường dẫn lưu file processed
    output_path = "data/raw/processed_data.csv"

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Lưu file CSV đã xử lý
    interaction_df.to_csv(output_path, index=False)
    print(f"[run_all_steps] Dữ liệu đã được lưu tại: {output_path}")

    # return snapshots, concept_graph, concept_memory, interaction_df
    return snapshots, concept_graph, concept_memory, interaction_df, len(question2idx), len(kc2idx), interaction_df["answer_idx"].nunique(), relation_dict, knowledge_mask

# Gọi hàm chính nếu chạy trực tiếp
if __name__ == "__main__":
    snapshots, concept_graph, concept_memory, interaction_df, num_q, num_c, num_o, relation_dict, knowledge_mask  = run_all_steps()

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = LayerNorm(dim)
        self.drop = Dropout(dropout)

    def forward(self, query, key, value):
        out, _ = self.attn(query, key, value)
        return self.norm(query + self.drop(out))

# MODULE 1: Disentangled Response Encoder
class DisentangledResponseEncoder(nn.Module):
    def __init__(self, dim_q, dropout):
        super().__init__()
        dim_h = dim_q // 2
        self.enc_correct = Sequential(Linear(dim_q, dim_h), ReLU(), Dropout(dropout), Linear(dim_h, dim_q))
        self.enc_wrong = Sequential(Linear(dim_q, dim_h), ReLU(), Dropout(dropout), Linear(dim_h, dim_q))
        self.enc_unchosen = Sequential(Linear(dim_q, dim_h), ReLU(), Dropout(dropout), Linear(dim_h, dim_q))
        self.attn_response = MultiHeadAttention(dim_q, 2, dropout)

    def forward(self, ot, ut, score):
        correct_mask = score == 1
        wrong_mask = score == 0
        ot_prime = torch.zeros_like(ot)
        ot_prime[correct_mask] = self.enc_correct(ot[correct_mask])
        ot_prime[wrong_mask] = self.enc_wrong(ot[wrong_mask])
        ut_prime = self.enc_unchosen(ut)
        d_t = ot_prime - ut_prime
        d_t_hat = self.attn_response(d_t, d_t, d_t)
        return d_t_hat

# MODULE 2: Knowledge Retriever
class KnowledgeRetriever(nn.Module):
    def __init__(self, dim_q, num_heads, dropout):
        super().__init__()
        self.attn_question = MultiHeadAttention(dim_q, num_heads, dropout)
        self.attn_state = MultiHeadAttention(dim_q, num_heads, dropout)

    def forward(self, qt, d_t, concept_ids):# thêm concept_ids
        qt_hat = self.attn_question(qt, qt, qt)
        h_t_list = []
        for _ in concept_ids:
            h_t_cid = self.attn_state(qt_hat, qt_hat, d_t)
            h_t_list.append(h_t_cid.squeeze(0))
        h_t_final = torch.stack(h_t_list, dim=0)  # [len(concept_ids), D]
        return qt_hat, h_t_final
# MODULE 3: Per-Concept Memory with Forget Gate + Time Decay
class MemoryUpdater(nn.Module):
    def __init__(self, dim_g, decay_scale=0.5):
        super().__init__()
        self.decay_scale = decay_scale
        self.forget_gate = Linear(dim_g + 1, 1)
    def forward(self, memory_value, delta_t, response_update):
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)  # [num_c, 1]
        time_feat = torch.log1p(delta_t.float()) * self.decay_scale
        gate_input = torch.cat([memory_value, time_feat], dim=-1)
        gamma = torch.sigmoid(self.forget_gate(gate_input))
        updated_memory = gamma * memory_value + (1 - gamma) * response_update
        return updated_memory

# MODULE 4: Dynamic Latent Concept Graph Builder
class DynamicConceptGraphBuilder(nn.Module):
    def __init__(self, num_c, top_k, knowledge_mask):
        super().__init__()
        self.num_c = num_c
        self.top_k = top_k
        self.knowledge_mask = knowledge_mask

    def forward(self, memory_value, attn_weights=None, knowledge_mask=None):
        num_nodes = memory_value.size(0)
        normed = memory_value / memory_value.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sim_matrix = torch.matmul(normed, normed.T)
        if attn_weights is not None:
            attn_outer = torch.ger(attn_weights, attn_weights)  # outer product
            sim_matrix = sim_matrix * (0.5 + 0.5 * attn_outer)

        sim_matrix = sim_matrix * (sim_matrix > 0.05).float()
        # sim_matrix.fill_diagonal_(1.0)
        edge_index, edge_weight = dense_to_sparse(sim_matrix)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        return edge_index, edge_weight

# MODULE 5: Prediction
class AttentionBasedPredictor(nn.Module):
    def __init__(self, dim_q, dim_g):
        super().__init__()
        self.query_proj = Linear(dim_q, dim_g)
        self.sigmoid = Sigmoid()

    def forward(self, qt_hat, memory_key, memory_value):
        pred_query = self.query_proj(qt_hat)
        sim = torch.matmul(pred_query, memory_key.T)
        top_k = min(10, sim.size(-1))
        top_k_val, top_k_idx = torch.topk(sim, top_k)
        attn_mask = torch.full_like(sim, float('-inf'))
        attn_mask[0, top_k_idx[0]] = top_k_val[0]
        attn = torch.softmax(attn_mask, dim=-1)
        mastery = torch.matmul(attn, memory_value)
        logits = torch.sum(pred_query * mastery, dim=-1)
        return self.sigmoid(logits)

class DCRKT(nn.Module):
    def __init__(self, num_c, num_q, num_o, dim_q, dim_g, num_heads, top_k, dropout, knowledge_mask=None):
        super().__init__()
        self.num_c = num_c
        self.num_q = num_q
        self.num_o = num_o
        self.dim_g = dim_g

        self.question_emb = Embedding(num_q + 1, dim_q)
        self.response_emb = Embedding(num_q * num_o + 2, dim_q, padding_idx=-1)

        self.encoder = DisentangledResponseEncoder(dim_q, dropout)
        self.retriever = KnowledgeRetriever(dim_q, num_heads, dropout)
        self.memory_updater = MemoryUpdater(dim_g, decay_scale=0.5)
        self.graph_builder = DynamicConceptGraphBuilder(num_c, top_k, knowledge_mask)
        self.predictor = AttentionBasedPredictor(dim_q, dim_g)
        self.knowledge_mask = knowledge_mask
        self.memory_key = nn.Parameter(torch.randn(num_c, dim_g))

        # === NEW: Per-student memory ===
        self.student_memory = {}
        self.last_update_time = {}
        self.snapshots = {}  # dict[student_id] → list of Mv snapshot over time
        self.snapshots_graph = {}
        self.max_snapshot_edges = 7
        self.gat = GAT(in_channels=dim_g, hidden_channels=dim_g // 2,
                       out_channels=dim_g, num_layers=2, dropout=dropout)

    def _get_student_memory(self, student_id):
        if student_id not in self.student_memory:
            self.student_memory[student_id] = torch.zeros(self.num_c, self.dim_g)
        return self.student_memory[student_id]

    def _get_last_update_time(self, student_id):
        if student_id not in self.last_update_time:
            self.last_update_time[student_id] = torch.zeros(self.num_c)
        return self.last_update_time[student_id]

    def reset_memory(self, student_id):
        self.student_memory[student_id] = torch.zeros(self.num_c, self.dim_g)
        self.last_update_time[student_id] = torch.zeros(self.num_c)
        self.snapshots[student_id] = []
        self.snapshots_graph[student_id] = []

    def forward_single_step(self, student_id, q_idx, o_idx, u_idx, score, timestamp, concept_ids):
        concept_ids = [cid for cid in concept_ids if 0 <= cid < self.num_c]
        if len(concept_ids) == 0:
            return torch.tensor(0.5)
        device = self.memory_key.device
        q_idx = torch.clamp(q_idx, 0, self.num_q - 1)
        o_idx = torch.clamp(o_idx, 0, self.num_o - 1)
        u_idx = torch.clamp(u_idx, 0, self.num_o - 1)
        response_idx = q_idx * self.num_o + o_idx
        unchosen_idx = q_idx * self.num_o + u_idx
        max_idx = self.response_emb.num_embeddings - 1
        if response_idx > max_idx or unchosen_idx > max_idx:
            raise ValueError(f"[Embedding Overflow] response_idx={response_idx}, unchosen_idx={unchosen_idx}, max={max_idx}")
        qt = self.question_emb(q_idx)
        ot = self.response_emb(q_idx * self.num_o + o_idx)
        ut = self.response_emb(q_idx * self.num_o + u_idx)

        d_t = self.encoder(ot.unsqueeze(0), ut.unsqueeze(0), score.unsqueeze(0))
        qt_hat, h_t_all = self.retriever(qt.unsqueeze(0), d_t, concept_ids)

        mv = self._get_student_memory(student_id).to(device)
        mk = self.memory_key.to(device)
        last_time = self._get_last_update_time(student_id).to(device)

        # === UPDATE MEMORY CHỈ TRÊN CONCEPT LIÊN QUAN ===
        current_time = timestamp.item()  # thời gian hiện tại
        delta_t = torch.zeros_like(last_time)
        for cid in concept_ids:
            delta_t[cid] = current_time - last_time[cid]
            last_time[cid] = current_time

        all_seen = set([i for i in range(self.num_c) if mv[i].norm() > 0])
        all_seen.update(concept_ids)
        if self.knowledge_mask is not None:
            for cid in concept_ids:
                related = (self.knowledge_mask[cid] > 0).nonzero(as_tuple=True)[0]
                all_seen.update(related.tolist())
        RECENT_SNAPSHOT = 3
        EDGE_WEIGHT_THRESHOLD = 0.05  # tùy chỉnh theo kinh nghiệm

        if student_id in self.snapshots_graph and self.snapshots_graph[student_id]:
            recent_snapshots = self.snapshots_graph[student_id][-RECENT_SNAPSHOT:]  # lấy snapshot gần nhất
            for prev_edges, prev_weights in recent_snapshots:
                for cid in concept_ids:
                    cid_tensor = torch.tensor(cid, device=prev_edges.device)

                    # Lấy các cạnh có trọng số đủ lớn
                    mask_from = (prev_edges[0] == cid_tensor) & (prev_weights > EDGE_WEIGHT_THRESHOLD)
                    connected_from = prev_edges[1][mask_from].tolist()

                    mask_to = (prev_edges[1] == cid_tensor) & (prev_weights > EDGE_WEIGHT_THRESHOLD)
                    connected_to = prev_edges[0][mask_to].tolist()
                    all_seen.update(connected_from)
                    all_seen.update(connected_to)

        seen_list = [i for i in all_seen if 0 <= i < self.num_c]
        if len(seen_list) == 0:
            return torch.tensor(0.5, device=device)

        seen_tensor = torch.tensor(seen_list, dtype=torch.long, device=device)
        mv_filtered = torch.index_select(mv, 0, seen_tensor)
        if mv_filtered.dim() == 1:
            mv_filtered = mv_filtered.unsqueeze(0)

        masked_kmask = self.knowledge_mask[seen_tensor][:, seen_tensor] if self.knowledge_mask is not None else None

        pred_query = self.predictor.query_proj(qt_hat)
        pred_query = F.normalize(pred_query, dim=-1)
        memory_key = F.normalize(mk, dim=-1)
        sim = torch.matmul(pred_query, memory_key.T)
        num_concepts = sim.size(-1)
        top_k = min(10, num_concepts)
        if top_k == 0:
            return torch.tensor(0.5, device=self.memory_key.device)
        top_k_val, top_k_idx = torch.topk(sim, k=top_k, dim=-1)

        for cid in concept_ids:
            if cid not in top_k_idx[0]:
                top_k_idx[0][-1] = cid
                top_k_val[0][-1] = sim[0, cid]
        attn_masked = torch.full_like(sim, float('-inf'))
        attn_masked[0, top_k_idx[0]] = top_k_val[0]
        attn_weights = F.softmax(attn_masked, dim=-1).squeeze(0)
        attn_weights_subset = attn_weights[seen_tensor]
        # Boost: nếu α_k > 0.1 thì scale lên để tăng ảnh hưởng
        attn_weights_boosted = attn_weights_subset.clone()
        attn_weights_boosted[attn_weights_subset > 0.1] *= 1.5
        attn_weights_boosted = attn_weights_boosted / attn_weights_boosted.sum()
        if masked_kmask is not None:
          boost = masked_kmask.sum(dim=1).float()
          boost = boost / (boost.sum() + 1e-6)
          attn_weights_boosted += 0.3 * boost
          attn_weights_boosted = attn_weights_boosted / attn_weights_boosted.sum()
        edge_index, edge_weight = self.graph_builder(mv_filtered, attn_weights=attn_weights_boosted ,knowledge_mask=masked_kmask )
        # print(f"[GraphBuilder] seen={len(seen_list)}, kmask_sum={masked_kmask.sum().item() if masked_kmask is not None else 'None'}, edge_count={edge_index.shape[1]}")
        # print(f"[α_k] min={attn_weights_subset.min().item():.4f}, max={attn_weights_subset.max().item():.4f}")
        # print("=" * 40)
        # print(f"   [Step Debug] student_id={student_id}, q_idx={q_idx.item()}, score={score.item()}")
        # print(f"   Concept IDs mapped: {concept_ids}")
        # print(f"   Attention Weights (α_k): {[round(attn_weights[cid].item(), 4) for cid in concept_ids]}")
        # print(f"   Max α_k: {attn_weights.max().item():.4f} | Sim range: {sim.min().item():.4f} → {sim.max().item():.4f}")
        # print(f"response_idx = {response_idx}, emb norm = {ot.norm().item():.4f}")
        idx_map = {i: cid for i, cid in enumerate(seen_list)}
        edge_index = edge_index.cpu().numpy()
        edge_index_mapped = torch.tensor([[idx_map[int(i)] for i in edge_index[0]], [idx_map[int(i)] for i in edge_index[1]]], device=mv.device, dtype=torch.long)
        if self.knowledge_mask is not None:
            extra_edges = []
            extra_weights = []
            for i in seen_list:
                for j in seen_list:
                    if self.knowledge_mask[i, j] == 1:
                        exists = ((edge_index_mapped[0] == i) & (edge_index_mapped[1] == j)).any()
                        if not exists:
                            extra_edges.append([i, j])
                            extra_weights.append(0.05)  # trọng số nhỏ để giữ ảnh hưởng thấp

            if extra_edges:
                extra_edges = torch.tensor(extra_edges, device=mv.device, dtype=torch.long).T  # [2, num_extra]
                edge_index_mapped = torch.cat([edge_index_mapped, extra_edges], dim=1)
                edge_weight = torch.cat([edge_weight, torch.tensor(extra_weights, device=mv.device)])
        mv_propagated = self.gat(mv, edge_index_mapped, edge_weight.to(device))
        if student_id not in self.snapshots_graph:
            self.snapshots_graph[student_id] = []
        self.snapshots_graph[student_id].append((edge_index_mapped.detach().cpu(), edge_weight.detach().cpu()))

        response_update = torch.zeros_like(mv)
        mv_updated = mv.detach().clone()
        for i, cid in enumerate(concept_ids):
            h_update = h_t_all[i]
            alpha = attn_weights[cid].item()
            response_update[cid] = alpha * h_update if alpha > 1e-4 else h_update
            updated = self.memory_updater(
                mv_propagated[cid].unsqueeze(0),
                delta_t[cid].unsqueeze(0),
                response_update[cid].unsqueeze(0)
            ).detach()
            mv_updated = mv_updated.index_copy(0, torch.tensor([cid], device=mv.device), updated)
            # print(f"⮕ [Trực tiếp] Mv[{cid}] ‖new‖ = {mv_updated[cid].norm().item():.6f}")

        # Cập nhật
        self.student_memory[student_id] = mv_updated.detach()
        self.last_update_time[student_id] = last_time
        if student_id not in self.snapshots:
            self.snapshots[student_id] = []
        self.snapshots[student_id].append(mv_updated.detach().clone())

        # === Dự đoán ===
        pred = self.predictor(qt_hat, mk, mv_updated)
        return pred.squeeze()

    def get_snapshot(self, student_id, step=-1):
        if student_id not in self.snapshots or not self.snapshots[student_id]:
            return None
        return self.snapshots[student_id][step]

# snapshots, concept_graph, concept_memory, interaction_df, num_q, num_c, num_o, relation_dict, knowledge_mask  = run_all_steps()
snapshots, concept_graph, concept_memory, interaction_df, num_q, num_c, num_o, relation_dict, knowledge_mask = run_all_steps()
code = """
import os
import torch
import pickle
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt

def save_model_and_snapshot(model, fold, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold{fold}.pth"))
    torch.save(model, os.path.join(output_dir, f"full_model_fold{fold}.pt"))
    with open(os.path.join(output_dir, f"snapshots_fold{fold}.pkl"), "wb") as f:
        pickle.dump(model.snapshots, f)
    with open(os.path.join(output_dir, f"memory_fold{fold}.pkl"), "wb") as f:
        pickle.dump(model.student_memory, f)
    print(f" Đã lưu model, snapshot và memory vào thư mục `{output_dir}/`")

def save_val_metrics(val_metrics, fold, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"val_metrics_fold{fold}.json")
    with open(path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f" Đã lưu kết quả đánh giá vào `{path}`")

def export_mastery_csv(model, kc_list, concept_df, fold, output_dir="outputs/"):
    rows = []
    for student_id, memory_tensor in model.student_memory.items():
        for idx, vec in enumerate(memory_tensor):
            mastery_score = vec.norm().item()
            concept_id = kc_list[idx]
            concept_name = concept_df[concept_df["id"] == concept_id]["name"].values[0]
            level = (
                "Yếu" if mastery_score < 1.0 else
                "Trung bình" if mastery_score < 1.5 else
                "Tốt"
            )
            rows.append({
                "student_id": student_id,
                "concept_id": concept_id,
                "concept_name": concept_name,
                "mastery_score": round(mastery_score, 4),
                "mastery_level": level
            })
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"student_mastery_detail_fold{fold}.csv")
    df.to_csv(path, index=False)
    print(f" Đã lưu file: {path}")

"""
with open("utils_dcrkt.py", "w", encoding="utf-8") as f:
    f.write(code)
print(" Đã tạo file utils_dcrkt.py thành công.")

import importlib
import utils_dcrkt as utils_dcrkt
importlib.reload(utils_dcrkt)

from utils_dcrkt import (
    save_model_and_snapshot,
    save_val_metrics,
    export_mastery_csv

)

from collections import defaultdict
import pandas as pd
import torch
from torch.optim import Adam
from torch.nn import BCELoss
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score, f1_score


def get_time_split_data(interaction_df, val_ratio=0.2):
    train_rows, val_rows = [], []
    for user_id, user_df in interaction_df.groupby("user_idx"):
        user_df = user_df.sort_values("timestamp")
        n = len(user_df)
        split_point = int(n * (1 - val_ratio))
        train_rows.append(user_df.iloc[:split_point])
        val_rows.append(user_df.iloc[split_point:])
    return pd.concat(train_rows).reset_index(drop=True), pd.concat(val_rows).reset_index(drop=True)

def reassign_variable(student_id, history_id):
    # # Gán lại các biến cần thiết sau tiền xử lý
    global question_kc_df, kc_df, student_df, history_df
    question_kc_df = question_concept_df.copy()
    kc_df = concept_df.copy()
    student_df = interaction_df[interaction_df["user_idx"] == user2idx[student_id]].sort_values("timestamp")
    history_df = student_df[student_df["question_idx"] != question2idx[history_id]]

def create_dcrt(student_id):
    global model, learned_concepts
    
    # Check if model exists and is working
    if model is None:
        print("Model not available. Please load the model first.")
        return None
    
    # Check if student_id exists in user2idx mapping
    if student_id not in user2idx:
        print(f"Student {student_id} not found in user2idx mapping.")
        return None
    
    # Get the device that the model is on
    model_device = next(model.parameters()).device
    
    # Reset memory for the student
    model.reset_memory(student_id)

    # 1. Lấy lịch sử học sinh
    user_idx = user2idx[student_id]
    history_df = interaction_df[interaction_df["user_idx"] == user_idx]

    if len(history_df) == 0:
        print(f"No interaction history found for student {student_id}")
        return None

    # 2. Duyệt qua từng tương tác
    for _, row in history_df.iterrows():
        try:
            qid = int(row["question_id"])
            oid = int(row["answer_choice_id"])
            uid = max(0, oid - 1)
            score = int(row["answer_state"])
            timestamp = pd.to_datetime(row["start_time"]).timestamp()
            
            # Get concept IDs for this question
            concept_ids = question_concept_df[question_concept_df["question_id"] == qid]["knowledgecomponent_id"].tolist()

            # Map concept ids
            mapped = [kc2idx[k] for k in concept_ids if k in kc2idx]
            if mapped:
                model.forward_single_step(
                    student_id=student_id,
                    q_idx=torch.tensor(qid, device=model_device),
                    o_idx=torch.tensor(oid, device=model_device),
                    u_idx=torch.tensor(uid, device=model_device),
                    score=torch.tensor(score, device=model_device),
                    timestamp=torch.tensor(timestamp, device=model_device),
                    concept_ids=mapped
                )
        except Exception as e:
            print(f"Error processing interaction: {e}")
            continue

    # 3. Lấy trạng thái kiến thức hiện tại
    snapshot = model.get_snapshot(student_id)

    # 4. Các concept học sinh từng gặp
    # learned_concepts = set()
    # for qid in history_df["question_id"]:
    #     concept_ids = question_concept_df[question_concept_df["question_id"] == qid]["knowledgecomponent_id"].tolist()
    #     learned_concepts.update(kc2idx[k] for k in concept_ids if k in kc2idx)
    learned_concepts = set()
    if snapshot is not None:
        for cid, vec in enumerate(snapshot):
            if vec.norm().item() > 0:  # học sinh đã có tương tác ảnh hưởng đến concept này
                learned_concepts.add(cid)
    # 5. Tạo dữ liệu cho bảng
    concept_data = []
    global weak_concepts_global
    weak_concepts_global = []  # Reset weak concepts for this student
    
    for cid in sorted(learned_concepts):
        try:
            # Get concept info from kc_list
            concept_id = kc_list[cid]
            concept_row = concept_df[concept_df['id'] == concept_id]
            
            if not concept_row.empty:
                concept_name = concept_row['name'].values[0]
                concept_description = concept_row['description'].values[0] if 'description' in concept_row.columns else "No description"
            else:
                concept_name = f"Concept {concept_id}"
                concept_description = "No description"
            
            mv_norm = snapshot[cid].norm().item() if snapshot is not None else 0.0
            
            # Determine mastery level based on norm
            if mv_norm >= 2.0:
                mastery_level = "High"
                mastery_color = "green"
            elif mv_norm >= 1.0:
                mastery_level = "Medium"
                mastery_color = "orange"
                # Add to weak concepts (medium mastery)
                weak_concepts_global.append({
                    "concept_id": cid,
                    "concept_name": concept_name,
                    "mastery_value": mv_norm,
                    "mastery_level": mastery_level,
                    "color": mastery_color
                })
            else:
                mastery_level = "Low"
                mastery_color = "red"
                # Add to weak concepts (low mastery)
                weak_concepts_global.append({
                    "concept_id": cid,
                    "concept_name": concept_name,
                    "mastery_value": mv_norm,
                    "mastery_level": mastery_level,
                    "color": mastery_color
                })
            
            concept_data.append({
                "Concept ID": cid,
                "Concept Name": concept_name,
                "Description": concept_description,
                "Mastery (‖Mv‖)": f"{mv_norm:.4f}",
                "Mastery Level": mastery_level,
                "Mastery Color": mastery_color
            })
        except Exception as e:
            concept_data.append({
                "Concept ID": cid,
                "Concept Name": f"Concept {cid}",
                "Description": "Error loading description",
                "Mastery (‖Mv‖)": "0.0000",
                "Mastery Level": "Unknown",
                "Mastery Color": "gray"
            })

    # 6. Hiển thị kết quả
    print(f"\n Học sinh {student_id} đã học các khái niệm sau:")
    print("-" * 70)
    print(f"{'Concept Index':<15} | {'Tên khái niệm':<40} | {'‖Mv‖'}")
    print("-" * 70)
    for cid in sorted(learned_concepts):
        try:
            concept_id = kc_list[cid]
            concept_row = concept_df[concept_df['id'] == concept_id]
            if not concept_row.empty:
                concept_name = concept_row['name'].values[0]
            else:
                concept_name = f"Concept {concept_id}"
            mv_norm = snapshot[cid].norm().item() if snapshot is not None else 0.0
            print(f"{cid:<15} | {concept_name:<40} | {mv_norm:.4f}")
        except Exception as e:
            print(f"{cid:<15} | [ lỗi]: {e}")

    return model, concept_data

def draw_graph(student_id):

    # 1. Tạo đồ thị kiến thức tổng từ snapshot và graph của mô hình
    G_full = visualize_student_knowledge_graph(
        student_id=student_id,
        step=-1,
        model=model,
        show_prerequisite=False
    )

    # 2. Lọc đồ thị: chỉ giữ các node mà học sinh 39 đã từng học
    G_learned = G_full.subgraph(learned_concepts).copy()

    # 3. Hiển thị đồ thị đã lọc
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_learned, seed=42)

    node_colors = [G_learned.nodes[n]["color"] for n in G_learned.nodes]
    node_labels = {n: G_learned.nodes[n]["name"] for n in G_learned.nodes}

    nx.draw_networkx_nodes(G_learned, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_edges(G_learned, pos, arrows=True)
    nx.draw_networkx_labels(G_learned, pos, labels=node_labels, font_size=10)

    plt.title(f"Bản đồ kiến thức của học sinh {student_id} (chỉ những khái niệm đã học)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def get_question_concepts(question_id):
    """
    Get the concepts associated with a specific question.
    Returns a list of concept dicts: [{"id": ..., "name": ..., "description": ...}, ...]
    """
    # Find all knowledgecomponent_id for the question
    kc_ids = question_concept_df[question_concept_df["question_id"] == question_id]["knowledgecomponent_id"].tolist()
    # Get concept info from concept_df
    concepts = []
    for kc_id in kc_ids:
        row = concept_df[concept_df["id"] == kc_id]
        if not row.empty:
            concept = {
                "id": kc_id,
                "name": row.iloc[0]["name"],
                "description": row.iloc[0]["description"]
            }
            concepts.append(concept)
    return concepts
#
def predict_question(student_id, question_id):
    """
        get question concepts 
    """
    concepts = get_question_concepts(question_id)
    print(f"\n📚 Câu hỏi {question_id} liên quan đến các khái niệm: {concepts}")

    """
    Predict the probability of a student answering a question correctly
    """
    try:
        # Use the correct question index mapping
        if question_id not in question2idx:
            print(f"Warning: Question {question_id} not found in question2idx mapping")
            return 0.5
        
        q_idx = question2idx[question_id]  # Use the correct mapping
        
        # Get the question concepts using the question ID
        question_concepts = question_concept_df[question_concept_df["question_id"] == question_id]["knowledgecomponent_id"].tolist()
        concept_ids = [kc2idx[k] for k in question_concepts if k in kc2idx]
        
        if not concept_ids:
            print(f"Warning: No concepts found for question {question_id}")
            return 0.5  # Default probability if no concepts
        
        # Get options for this question
        options = option_df[option_df["question_id"] == question_id]
        if len(options) < 2:
            print(f"Warning: Not enough options for question {question_id}")
            return 0.5
        
        # Use actual option IDs like in model14_6.py
        option_ids = options["id"].tolist()
        o_idx = torch.tensor(option_ids[0])  # First option ID
        u_idx = torch.tensor(option_ids[1])  # Second option ID (unchosen)
        
        # Use current timestamp like in model14_6.py
        timestamp_now = torch.tensor(pd.Timestamp.now().timestamp())
        
        # Make prediction with consistent parameters
        with torch.no_grad():
            pred = model.forward_single_step(
                student_id=student_id,
                q_idx=torch.tensor(q_idx),
                o_idx=o_idx,
                u_idx=u_idx,
                score=torch.tensor(1.0),  # Use 1.0 for prediction (assuming correct)
                timestamp=timestamp_now,
                concept_ids=concept_ids
            )
            
            prediction_prob = pred.item()
            print(f"\n[Dự đoán] Xác suất học sinh {student_id} trả lời đúng câu hỏi {question_id}: {prediction_prob:.4f}")
            return prediction_prob
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0.5  # Default probability on error


    # """
    # Analyze detailed performance metrics for a student
    # """
    # print(f"\n📊 DETAILED PERFORMANCE ANALYSIS FOR STUDENT {student_id}")
    # print("=" * 70)
    
    # # Get student data
    # student_data = interaction_df[interaction_df["user_idx"] == user2idx[student_id]]
    
    # if len(student_data) == 0:
    #     print(f"❌ No data found for student {student_id}")
    #     return
    
    # # Basic statistics
    # total_questions = len(student_data)
    # correct_answers = student_data['is_correct'].sum()
    # accuracy = (correct_answers / total_questions) * 100
    
    # print(f"📈 Basic Statistics:")
    # print(f"   • Total Questions Attempted: {total_questions}")
    # print(f"   • Correct Answers: {correct_answers}")
    # print(f"   • Accuracy: {accuracy:.2f}%")
    
    # # Time analysis
    # if 'timestamp' in student_data.columns:
    #     first_time = student_data['timestamp'].min()
    #     last_time = student_data['timestamp'].max()
    #     time_span = last_time - first_time
    #     print(f"   • Time Span: {time_span/3600:.1f} hours")
    #     print(f"   • Average Time per Question: {time_span/total_questions:.0f} seconds")
    
    # # Concept mastery analysis
    # concept_performance = {}
    # for _, row in student_data.iterrows():
    #     concepts = row['concept_idxs']
    #     is_correct = row['is_correct']
        
    #     for concept in concepts:
    #         if concept not in concept_performance:
    #             concept_performance[concept] = {'correct': 0, 'total': 0}
    #         concept_performance[concept]['total'] += 1
    #         if is_correct:
    #             concept_performance[concept]['correct'] += 1
    
    # print(f"\n🎯 Concept Performance:")
    # print(f"{'Concept ID':<12} | {'Concept Name':<35} | {'Correct':<8} | {'Total':<6} | {'Accuracy':<10}")
    # print("-" * 80)
    
    # for concept_id, stats in sorted(concept_performance.items()):
    #     try:
    #         concept_name = kc_df[kc_df['id'] == kc_list[concept_id]]['name'].values[0]
    #         concept_acc = (stats['correct'] / stats['total']) * 100
    #         print(f"{concept_id:<12} | {concept_name:<35} | {stats['correct']:<8} | {stats['total']:<6} | {concept_acc:<10.1f}%")
    #     except:
    #         concept_acc = (stats['correct'] / stats['total']) * 100
    #         print(f"{concept_id:<12} | {'Unknown':<35} | {stats['correct']:<8} | {stats['total']:<6} | {concept_acc:<10.1f}%")

  
# Helper function to get concept name from CSV
def get_concept_name(concept_index, concept_df, kc_list):
    """Get concept name from the KCs.csv file using concept index"""
    if concept_index >= len(kc_list):
        return f"Concept {concept_index}"
    
    # Get the real concept ID from kc_list
    concept_id_real = kc_list[concept_index]
    
    # Get concept name from the CSV file
    concept_row = concept_df[concept_df["id"] == concept_id_real]
    if not concept_row.empty:
        name = concept_row["name"].values[0]
        return name
    else:
        return f"Concept {concept_id_real}"


# GPT Integration for Learning Path
def generate_learning_path(weak_concepts, concept_df, kc_list, time_period="7 days", student_id=None, force_refresh=False):
    """
    Generate a personalized learning path for a student based on weak concepts
    """
    # Configure your OpenAI API key
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    
    if not api_key:
        return "Please configure your OpenAI API key in Streamlit secrets to use this feature."
    
    # Format weak concepts with correct names from CSV
    weak_concept_names = []
    weak_concept_ids = []  # Track concept IDs for cache key
    for idx in weak_concepts:
        if idx < len(kc_list):
            kc_id = kc_list[idx]
            concept_name = get_concept_name(idx, concept_df, kc_list)
            weak_concept_names.append(concept_name)
            weak_concept_ids.append(kc_id)
    
    if not weak_concept_names:
        return "No weak concepts identified (‖Mv‖ < 1.0). Keep up the good work!"
    
    # Create cache directory if it doesn't exist
    os.makedirs("learning_path_cache", exist_ok=True)
    
    # Create a cache key based on student_id, weak concepts and time period
    if student_id is not None:
        # Sort concept IDs to ensure consistent cache key regardless of order
        cache_key = f"{student_id}_{'-'.join(sorted([str(cid) for cid in weak_concept_ids]))}_{time_period}"
        cache_file = os.path.join("learning_path_cache", f"{cache_key}.txt")
        
        # Check if cache exists and not forced to refresh
        if os.path.exists(cache_file) and not force_refresh:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_content = f.read()
                st.success("Using cached learning path. Click 'Refresh' to generate a new one.")
                return cached_content
            except Exception as e:
                print(f"Error reading cache: {e}")
                # Continue with generating a new path if cache read fails
    
    # Calculate spaced repetition intervals based on time period
    if time_period == "7 days":
        intervals = ", ".join([f"Day {i+1}" for i in range(7)])
        final_day = "Day 7"
        days = 7
    elif time_period == "15 days":
        intervals = ", ".join([f"Day {i+1}" for i in range(15)])
        final_day = "Day 15"
        days = 15
    elif time_period == "1 month":
        intervals = ", ".join([f"Day {i+1}" for i in range(30)])
        final_day = "Day 30"
        days = 30
    else:
        intervals = ", ".join([f"Day {i+1}" for i in range(7)])
        final_day = "Day 7"
        days = 7
    
    # Update the prompt to require a full plan for the selected period
    content = f"""
    As an AI tutor specializing in evidence-based learning techniques, create a personalized learning path focused on ACTIVE RECALL strategies for a student who needs to improve on these concepts:
    {', '.join(weak_concept_names)}
    
    The learning path MUST be a COMPLETE, DETAILED, DAY-BY-DAY plan for the entire {time_period} period (not just a summary or a few days). For each day, specify exactly what the student should do for each concept, including specific active recall activities, retrieval practice, and spaced repetition. Do not skip any days. The plan should be actionable and easy to follow.
    
    For each concept, provide SPECIFIC active recall activities:
    
    1. **Retrieval Practice Exercises:**
       - Write down everything you know about the concept from memory
       - Create self-quizzes with specific questions
       - Use the Feynman Technique (explain to someone else)
       - Practice problems without looking at solutions first
    
    2. **Spaced Repetition Schedule ({time_period}):**
       - {intervals}
       - Final recall: {final_day} (consolidation)
    
    3. **Active Learning Methods:**
       - Concept mapping from memory
       - Teaching the concept to a peer
       - Creating flashcards with spaced repetition
       - Problem-solving without reference materials
    
    4. **Progress Tracking:**
       - Self-assessment rubrics for each concept
       - Confidence ratings before and after practice
       - Error analysis and correction strategies
    
    For each concept, provide:
    - 2-3 specific active recall questions
    - 1-2 practical application problems
    - A confidence-building exercise
    
    Format as a clear, actionable plan that emphasizes DOING and PRACTICING rather than passive reading.
    Focus on the weakest concepts first and build up to more challenging ones.
    Ensure the schedule fits within the {time_period} timeframe and covers every day with specific tasks for each concept. The plan should be detailed and not skip any days.
    """
    
    try:
        # Use the new OpenAI client for API version 1.0.0+
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational AI tutor specializing in active recall learning strategies and spaced repetition. You create detailed, actionable study plans based on cognitive science research."},
                {"role": "user", "content": content}
            ],
            temperature=0.7,  # Add some creativity but keep focused
            max_tokens=2000   # Allow for longer, more detailed responses
        )
        
        result = response.choices[0].message.content
        
        # Cache the result if student_id is provided
        if student_id is not None:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"Cached learning path for student {student_id}")
            except Exception as e:
                print(f"Error caching learning path: {e}")
        
        return result
    except Exception as e:
        return f"Error generating learning path: {e}\n\nPlease check your OpenAI API key configuration."

def handle_sidebar():
    global model
    st.sidebar.title("DCRKT Knowledge Tracing Model")
    
    # Show model status
    if model is not None:
        st.sidebar.success("✅ Model is loaded and ready")
        st.sidebar.caption(f"Model loaded from: full_model_fold1.pt")
        st.sidebar.caption(f"Device: {device}")
    else:
        st.sidebar.error("❌ Model not loaded")
        st.sidebar.caption("Please ensure full_model_fold1.pt exists")
    
    # Model loading buttons
    if st.sidebar.button("🔄 Reload Model"):
        try:
            # Add safe globals for PyTorch 2.6 compatibility
            torch.serialization.add_safe_globals([argparse.Namespace])
            
            # Use the new checkpoint loading approach
            checkpoint = torch.load('outputs/full_model_fold1.pt', map_location=device, weights_only=False)
            args = checkpoint['args']
            state_dict = checkpoint['model_state_dict']

            # Debug: Print the args to see what dimensions were used
            print(f"Loaded model args: q_dim={args.q_dim}, g_dim={args.g_dim}, concept={args.concept}, question={args.question}, option={args.option}")

            # Recreate knowledge_mask if needed
            _, knowledge_mask = create_concept_graph(relation_df, kc2idx, args.concept)

            model = DCRKT(
                num_c=args.concept,
                num_q=args.question,
                num_o=args.option,
                dim_q=args.q_dim,
                dim_g=args.g_dim,
                num_heads=args.heads,
                top_k=args.top_k,
                dropout=args.dropout,
                knowledge_mask=knowledge_mask
            )
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            st.sidebar.success("✅ Model reloaded successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")
    
    st.sidebar.button("Load Selected Model")
    st.sidebar.button("Load Uploaded Model")
    
    # Show system info
    st.sidebar.markdown("---")
    st.sidebar.caption("System Information")
    st.sidebar.caption(f"PyTorch: {torch.__version__}")
    st.sidebar.caption(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.sidebar.caption(f"CUDA Device: {torch.cuda.get_device_name()}")

def handle_question_info(question_id):
    """
    Get question details from questions.csv
    """
    try:
        # Load questions.csv if not already loaded
        questions_df = pd.read_csv("input/raw/questions.csv")
        
        # Find the question by ID
        question_data = questions_df[questions_df["id"] == question_id]
        
        if question_data.empty:
            return None
        
        # Return the first (and should be only) row
        question_text = question_data.iloc[0]['question_text']
        # Display question info in a box similar to the image
        st.markdown("""
            <style>
            .question-box {
                border: 2px solid black;
                padding: 10px;
                margin-bottom: 20px;
            }
            .question-title {
                text-align: center;
                font-weight: bold;
                font-size: 1.2em;
                margin-bottom: 10px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }
            .concept-tag {
                display: inline-block;
                background-color: #FFFACD;
                border: 1px solid #FFD700;
                border-radius: 15px;
                padding: 2px 10px;
                margin: 3px;
                font-size: 0.9em;
            }
            </style>
            """, unsafe_allow_html=True)
        with st.container():
                # Create question box similar to the image
            st.markdown(
                    f"<div class='question-box'>"
                    f"<div class='question-title'>Target Question (Q{question_id})</div>"
                    f"<p><strong>Q.</strong> {question_text}</p>",
                    unsafe_allow_html=True)

    # Load question choices from Question_Choices.csv
 
        choices_df = pd.read_csv("input/raw/Question_Choices.csv")
        
        # Filter choices for the current question
        question_choices = choices_df[choices_df["question_id"] == question_id]
        
        if not question_choices.empty:
            st.markdown("<p><strong>Choices:</strong></p>", unsafe_allow_html=True)
            
            # Display each choice with letter labels (A, B, C, D...)
            choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            
            for idx, choice in question_choices.iterrows():
                choice_letter = choice_letters[idx % len(choice_letters)]
                choice_text = choice['choice_text']
                is_correct = choice['is_correct']
                
                # Style correct answers with green background
                if is_correct:
                    st.markdown(
                        f"<div style='background-color: #d4edda; border: 1px solid #c3e6cb; "
                        f"border-radius: 5px; padding: 8px; margin: 2px 0;'>"
                        f"<strong>{choice_letter}.</strong> {choice_text} ✓</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background-color: #f8f9fa; border: 1px solid #dee2e6; "
                        f"border-radius: 5px; padding: 8px; margin: 2px 0;'>"
                        f"<strong>{choice_letter}.</strong> {choice_text}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.warning("⚠️ No choices found for this question.")
            
    except FileNotFoundError:
        st.error("❌ Question_Choices.csv file not found in input/raw/")
    except Exception as e:
        st.error(f"❌ Error loading Question_Choices.csv: {e}")
    except FileNotFoundError:
        st.error("❌ questions.csv file not found in input/raw/")
        return None
    except Exception as e:
        st.error(f"❌ Error loading questions.csv: {e}")
        return None
def show_question_option_probabilities(model, student_id, question_id):
    """
    Predict the probability of correct answers for a given student and question.
    Do not update model.student_memory or save any new memory vector.
    Only show the prediction probabilities results
    """
    try:
        # Check if model is available
        if model is None:
            st.warning("Model not available. Please ensure the model is loaded.")
            return

        # Get the device that the model is on
        model_device = next(model.parameters()).device

        # Get question index and concept ids
        if question_id not in question2idx:
            st.warning(f"Question ID {question_id} not found!")
            return
        q_idx = question2idx[question_id]
        concept_ids = [
            kc2idx[k]
            for k in question_concept_df[question_concept_df["question_id"] == question_id]["knowledgecomponent_id"].tolist()
            if k in kc2idx
        ]
        options = option_df[option_df["question_id"] == question_id]
        option_ids = options["id"].tolist()
        option_texts = options["choice_text"].tolist()

        # Use current timestamp for delta_t calculation (not for saving memory)
        timestamp_now = torch.tensor(pd.Timestamp.now().timestamp(), device=model_device)

        predictions = []
        for i, (oid, _) in enumerate(zip(option_ids, option_texts)):
            o_idx = torch.tensor(oid, device=model_device)
            u_idx = torch.tensor(option_ids[(i + 1) % len(option_ids)], device=model_device)
            with torch.no_grad():
                # Do not update model.student_memory inside forward_single_step
                prob = model.forward_single_step(
                    student_id=student_id,
                    q_idx=torch.tensor(q_idx, device=model_device),
                    o_idx=o_idx,
                    u_idx=u_idx,
                    score=torch.tensor(1.0, device=model_device),  # Use 1.0 for "correct assumption"
                    timestamp=timestamp_now,
                    concept_ids=concept_ids
                )
            predictions.append(prob.item())

        # Display the prediction probabilities for each option
        st.markdown(
            "<div class='question-box'>"
            "<div class='question-title'>Prediction Results</div>",
            unsafe_allow_html=True
        )

        # if option_texts and predictions:
        #     for idx, (option_text, prob) in enumerate(zip(option_texts, predictions)):
        #         letter = chr(65 + idx)  # A, B, C, ...
        #         st.markdown(
        #             f"<p><strong>{letter}.</strong> {option_text} "
        #             f"<span style='color: green;'>(Probability: {prob:.2%})</span></p>",
        #             unsafe_allow_html=True
        #         )
        st.markdown(
            f"<p><strong>Predicted probability of correct answer:</strong> "
            f"<span style='color: blue; font-weight: bold;'>{prob.item():.2%}</span></p>",
            unsafe_allow_html=True
            )
      
        # Close the prediction box
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error predicting option probabilities: {e}")
        st.info("This error usually occurs when the model is not properly initialized for the selected student.")

def handle_select_student():
    global selected_student, model
    st.subheader("Student Selection")

    # Get all available student IDs
    available_students = sorted(user2idx.keys())

    # Use session_state to store and react to student/question selection
    if "selected_student" not in st.session_state:
        st.session_state.selected_student = available_students[0] if available_students else None
    if "selected_question_display" not in st.session_state:
        st.session_state.selected_question_display = None

    # Student selection dropdown
    selected_student = st.selectbox(
        "Select a student:",
        options=available_students,
        index=available_students.index(39) if 39 in available_students else 0,
        key="student_select"
    )
    st.session_state.selected_student = selected_student

    # Get all available questions for the selected student
    student_data = interaction_df[interaction_df["user_idx"] == user2idx[selected_student]]
    available_questions = sorted(student_data['question_idx'].unique())

    # Create question selection dropdown with +1 display
    question_display_options = [q + 1 for q in available_questions]
    default_question_index = 0  # Mặc định chọn phần tử đầu tiên

    # Nếu trước đó người dùng đã chọn câu hỏi nào đó (và vẫn nằm trong danh sách), thì giữ lại
    if "question_select" in st.session_state:
        try:
            previous_value = st.session_state.question_select
            if previous_value in question_display_options:
                default_question_index = question_display_options.index(previous_value)
        except:
            pass

    selected_question_display = st.selectbox(
        "Select a question:",
        options=question_display_options,
        index=default_question_index,  # An toàn, không bị vượt ngoài danh sách
        key="question_select"
    )    
    # selected_question_display = st.selectbox(
    #     "Select a question:",
    #     options=question_display_options,
    #     index=8 if len(question_display_options) > 0 else None,
    #     key="question_select"
    # )
    st.session_state.selected_question_display = selected_question_display

    if selected_question_display is not None:
        selected_question = selected_question_display - 1  # Convert back to original index
        handle_question_info(selected_question_display)
        # Now show concepts section
        st.markdown("<div class='question-title'>Concepts</div>", unsafe_allow_html=True)
        concept_names = get_question_concepts(selected_question_display)
        if concept_names:
            for concept in concept_names:
                st.markdown(f"<span class='concept-tag'>{concept['name']}</span>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No concepts found for this question.")

        # Ensure model is initialized for the selected student before making predictions
        if model is None:
            print("Model not loaded. Please ensure the model is properly loaded.")
        else:
            # Initialize model for the selected student if not already done
            try:
                # Check if model has the required methods
                if not hasattr(model, 'forward_single_step') or not hasattr(model, 'student_memory'):
                    st.error("Model is not properly initialized. Please reload the model.")
                    return
                
                # Check if student memory exists in the model
                if selected_student not in model.student_memory or len(model.student_memory[selected_student]) == 0:
                    with st.spinner(f"Initializing model for student {selected_student}..."):
                        result = create_dcrt(selected_student)
                        if result is not None:
                            st.success(f"Model initialized for student {selected_student}")
                        else:
                            st.error(f"Failed to initialize model for student {selected_student}")
                            return
                
                # Call the function to display probabilities
                show_question_option_probabilities(model, selected_student, selected_question_display)
            except Exception as e:
                st.error(f"Error initializing model for student {selected_student}: {e}")
                st.info("Please try selecting a different student or reload the model.")
    
            

def create_mastery_legend():
    legend_html = """
    <div style="background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
        <h4 style="text-align: center;">Mức độ thành thạo</h4>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgba(255, 50, 50, 0.8); margin-right: 10px;"></div>
            <div>Thấp</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgba(255, 150, 50, 0.8); margin-right: 10px;"></div>
            <div>Trung bình</div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgba(50, 200, 50, 0.8); margin-right: 10px;"></div>
            <div>Cao</div>
        </div>
    </div>
    """
    return legend_html


def simple_draw_graph_streamlit(student_id):
    """
    Vẽ bản đồ kiến thức các khái niệm học sinh đã học sau huấn luyện
    """
    try:
        with st.spinner("Đang tính toán bản đồ kiến thức..."):
            # Bước 1: Tạo DCRT và kiểm tra dữ liệu học
            result = create_dcrt(student_id)
            if result is None:
                st.warning(f"Không có dữ liệu học cho học sinh {student_id}.")
                return

            model_updated, concept_data = result

            if model is None or student_id not in model.snapshots or len(model.snapshots[student_id]) == 0:
                st.warning(f"Không có snapshot cho học sinh {student_id}.")
                return

            # Bước 2: Lấy snapshot vector và danh sách khái niệm đã học
            mv = model.get_snapshot(student_id, step=-1)

            _, _ = create_dcrt(student_id)
            mv = model.get_snapshot(student_id)
            global learned_concepts
            learned_concepts = [cid for cid in learned_concepts if mv[cid].norm().item() > 0]


            # Bước 3: Vẽ đồ thị đầy đủ và lọc theo concept đã học
            G_full = visualize_student_knowledge_graph(
                student_id=student_id,
                step=-1,
                model=model,
                show_prerequisite=False
            )

            G_learned = G_full.subgraph(learned_concepts).copy()

            if len(G_learned.nodes()) == 0:
                st.info(f"Không có khái niệm nào đủ điều kiện hiển thị.")
                return

            # Bước 4: Vẽ đồ thị bằng matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(G_learned, seed=42)
            node_colors = [G_learned.nodes[n]["color"] for n in G_learned.nodes]
            node_labels = {n: G_learned.nodes[n]["name"] for n in G_learned.nodes}
            nx.draw_networkx_nodes(G_learned, pos, node_color=node_colors, node_size=700)
            nx.draw_networkx_edges(G_learned, pos, arrows=True)
            nx.draw_networkx_labels(G_learned, pos, labels=node_labels, font_size=10)
            plt.title(f"🧠 Bản đồ kiến thức đã học – Học sinh {student_id}", fontsize=14)
            plt.axis("off")
            plt.tight_layout()

        st.pyplot(fig)
        plt.close()
        st.success(f"✅ Đã tạo bản đồ kiến thức cho học sinh {student_id}")

    except Exception as e:
        st.error(f"❌ Lỗi khi tạo bản đồ: {e}")
        st.info("1. Học sinh chưa có lịch sử học")
        st.info("2. Mô hình chưa được khởi tạo đúng")
        st.info("3. Dữ liệu snapshot hoặc khái niệm bị thiếu")


def handle_knowledge_concept_map():
    st.header("Knowledge Concept Map")
    st.write("This section will display the knowledge concept map for the selected student.")
    
    if selected_student:
        # Add visualization controls
        viz_col1, viz_col2 = st.columns([4, 1])
        with viz_col1:
            # Simple call to draw the graph
            simple_draw_graph_streamlit(selected_student)
        with viz_col2:
            st.markdown(create_mastery_legend(), unsafe_allow_html=True)
    else:
        st.info("👆 Please select a student to view their knowledge concept map")


def handle_learning_path_recommendation(tab2):
    with tab2:
        st.header("Learning Path Recommendation")
        
        # Check if model is loaded
        if model is None:
            st.warning("Please load a model first.")
            return
        
        # Use the current selected student from session state
        if "selected_student" not in st.session_state or st.session_state.selected_student is None:
            st.warning("Please select a student in the Knowledge Testing tab first.")
            return
        
        selected_student_lp = st.session_state.selected_student
        
        # Check if we have weak concepts data for this student
        if not weak_concepts_global:
            st.warning(f"No weak concepts data available for student {selected_student_lp}. Please analyze the student in the Knowledge Testing tab first.")
            return
        
        # Display weak concepts using the global array
        st.subheader(f"📚 Weak Concepts for Student {selected_student_lp}")
        
        # Create two columns for layout: 1:3 ratio
        col_weak, col_path = st.columns([1, 3])
        
        # Display weak concepts in left column
        with col_weak:
            st.subheader("Concepts Needing Improvement")
            
            # Add a card-like styling
            st.markdown("""
            <style>
            .concept-card {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
                border-left: 4px solid #ff6b6b;
            }
            .concept-card-medium {
                border-left: 4px solid #ffa500;
            }
            .concept-card-low {
                border-left: 4px solid #ff0000;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if weak_concepts_global:
                # Display weak concepts with color coding
                for concept in weak_concepts_global:
                    card_class = "concept-card-low" if concept["mastery_level"] == "Low" else "concept-card-medium"
                    st.markdown(f"""
                    <div class="concept-card {card_class}">
                        <strong>C<sub>{concept['concept_id']}</sub></strong> {concept['concept_name']}<br>
                        <small>Mastery: {concept['mastery_value']:.3f} ({concept['mastery_level']})</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show count
                low_count = len([c for c in weak_concepts_global if c["mastery_level"] == "Low"])
                medium_count = len([c for c in weak_concepts_global if c["mastery_level"] == "Medium"])
                st.caption(f"Total: {len(weak_concepts_global)} concepts need improvement")
                st.caption(f"Low: {low_count}, Medium: {medium_count}")
            else:
                st.success("No weak concepts identified!")
                st.caption("All concepts have high mastery levels")
        
        # Display learning path in right column
        with col_path:
            st.subheader("Personalized Learning Path")
            
            # Time period selection
            time_period = st.radio(
                "Select time period for learning path",
                ["7 days", "15 days", "1 month"],
                horizontal=True,
                key="time_period_lp"
            )
            
            # Automatically generate learning path
            try:
                # Add a refresh button
                col_refresh, col_status = st.columns([1, 3])
                with col_refresh:
                    refresh = st.button("Refresh Learning Path", key="refresh_lp")
                
                with col_status:
                    if refresh:
                        st.info("Generating new learning path...")
                
                with st.spinner("Generating personalized learning path..."):
                    # Extract concept IDs for the learning path function
                    weak_concept_ids = [c["concept_id"] for c in weak_concepts_global]
                    
                    # Pass student_id for caching
                    learning_path = generate_learning_path(
                        weak_concept_ids, 
                        concept_df, 
                        kc_list, 
                        time_period,
                        student_id=selected_student_lp,
                        force_refresh=refresh
                    )
                    
                    # Format the learning path
                    st.markdown("""
                    <style>
                    .learning-path {
                        background-color: white;
                        border-radius: 5px;
                        padding: 15px;
                        border: 1px solid #ddd;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="learning-path">{learning_path}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating learning path: {str(e)}")
                st.info("Suggestions for fixing this error:")
                st.info("1. Check if the OpenAI API key is correctly set in Streamlit secrets")
                st.info("2. Make sure you have the latest version of the OpenAI package: `pip install --upgrade openai`")
                
                # Show a simple list of weak concepts as a fallback
                if weak_concepts_global:
                    st.markdown("## General Study Recommendations")
                    st.write("Based on the identified weak concepts, here are some general recommendations:")
                    
                    recommendations = []
                    for concept in weak_concepts_global[:5]:  # Limit to first 5 concepts
                        recommendations.append(f"• Study more about **{concept['concept_name']}** - focus on foundational principles and practical examples")
                    
                    if len(weak_concepts_global) > 5:
                        recommendations.append(f"• ...and {len(weak_concepts_global) - 5} more concepts")
                    
                    st.markdown("\n".join(recommendations))

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize global variables
model = None
question_kc_df = question_concept_df.copy()
weak_concepts_global = []  # Global array to store weak concepts
interaction_df = None
user2idx = {}
question2idx = {}
kc2idx = {}
kc_list = []
knowledge_mask = None

def initialize_model_and_data():
    """Initialize model and data preprocessing from finalthesis (3).py"""
    global model, interaction_df, user2idx, question2idx, kc2idx, kc_list, knowledge_mask, question_df, option_df, concept_df, relation_df, question_concept_df
    
    try:
        # 0. Load all data files first
        print("Loading data files...")
        folder_path = 'input/raw'
        
        question_df = pd.read_csv(f"{folder_path}/Questions.csv")
        option_df = pd.read_csv(f"{folder_path}/Question_Choices.csv")
        concept_df = pd.read_csv(f"{folder_path}/KCs.csv")
        relation_df = pd.read_csv(f"{folder_path}/KC_Relationships.csv")
        question_concept_df = pd.read_csv(f"{folder_path}/Question_KC_Relationships.csv")
        interaction_df = pd.read_csv(f"{folder_path}/Transaction.csv")
        specialization_df = pd.read_csv(f"{folder_path}/Specialization.csv")
        student_specialization_df = pd.read_csv(f"{folder_path}/Student_Specialization.csv")
        
        print(" Data files loaded successfully")
        
        # 1. Run data preprocessing (same as finalthesis (3).py)
        print(" Running data preprocessing...")
        snapshots, concept_graph, concept_memory, interaction_df, num_q, num_c, num_o, relation_dict, knowledge_mask = run_all_steps()
        
        # 2. Register DCRKT class for model loading
        torch.serialization.add_safe_globals([('DCRKT', DCRKT)])
        
        # 3. Load the trained model
        print(" Loading trained model...")
        model_path = "outputs/full_model_fold1.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        print(f" Model loaded successfully from {model_path}")
        
        # 4. Load student memory
        print(" Loading student memory...")
        memory_path = "outputs/memory_fold1.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, "rb") as f:
                student_memory = pickle.load(f)
            print(f" Student memory loaded for {len(student_memory)} students")
        else:
            print(" Student memory file not found, will use model's internal memory")
        
        # 5. Load snapshots
        print(" Loading snapshots...")
        snapshots_path = "outputs/snapshots_fold1.pkl"
        if os.path.exists(snapshots_path):
            with open(snapshots_path, "rb") as f:
                snapshots = pickle.load(f)
            print(f" Snapshots loaded for {len(snapshots)} students")
        else:
            print(" Snapshots file not found")
        
        # 6. Load concept list
        print(" Loading concept list...")
        concept_csv_path = "input/raw/KCs.csv"
        concept_df = pd.read_csv(concept_csv_path)
        kc_list = concept_df["id"].tolist()
        print(f" Loaded {len(kc_list)} concepts")
        
        # 7. Create mappings
        user_list = sorted(interaction_df["student_id"].unique())
        question_list = sorted(question_df["id"].unique())
        kc_list_sorted = sorted(concept_df["id"].unique())
        
        user2idx = {u: i for i, u in enumerate(user_list)}
        question2idx = {q: i for i, q in enumerate(question_list)}
        kc2idx = {k: i for i, k in enumerate(kc_list_sorted)}
        
        print(f"✅ Mappings created - Users: {len(user2idx)}, Questions: {len(question2idx)}, Concepts: {len(kc2idx)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing model and data: {e}")
        return False

# Initialize model and data
if not initialize_model_and_data():
    print("❌ Failed to initialize model and data. Please ensure all required files exist.")
    model = None
else:
    print("✅ Model and data initialized successfully!")


def handle_knowledge_testing(tab1):
    with tab1:
        st.header("Test Student Knowledge")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            handle_select_student()
        
        with col2:
            handle_knowledge_concept_map()


# Main App
def main():
    # Configure page
    st.set_page_config(layout="wide", page_title="DCRKT Knowledge Tracing System")
    st.title("Knowledge Tracing and Learning Path System")

    # Check model status at startup
    global model
    if model is None:
        st.warning("⚠️ Model not loaded. Attempting to load model...")
        try:
            # Try to initialize model and data
            if initialize_model_and_data():
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please ensure all required files exist.")
                st.info("Required files:")
                st.info("• outputs/full_model_fold1.pt")
                st.info("• outputs/memory_fold1.pkl")
                st.info("• outputs/snapshots_fold1.pkl")
                st.info("• input/raw/KCs.csv")
                st.info("• input/raw/Questions.csv")
                st.info("• input/raw/Transaction.csv")
                st.info("• input/raw/Question_KC_Relationships.csv")
                st.info("• input/raw/KC_Relationships.csv")
                st.info("• input/raw/Question_Choices.csv")
                return
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("Please check the model file and try again.")
            return
    else:
        st.success("✅ Model is ready!")

    handle_sidebar()

    # Tabs for different sections
    tab1, tab2 = st.tabs(["Knowledge Testing", "Learning Path Recommendation"])

    handle_knowledge_testing(tab1)
    handle_learning_path_recommendation(tab2)

# Run the console interface
if __name__ == "__main__":
    main()
    # run_console_analysis()


