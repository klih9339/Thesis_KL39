import numpy as np
import pandas as pd
import os
import torch
from torch.nn import BCELoss, Embedding, Sequential, Linear, ReLU, Dropout, LayerNorm, Sigmoid
from torch.optim import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from collections import defaultdict
from tqdm import tqdm
import gc
from argparse import Namespace
import multiprocessing
import concurrent.futures
from functools import partial
import psutil
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import one_hot, softmax
from torch_geometric.nn import GAT
from torch_geometric.utils import dense_to_sparse

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up optimized CUDA parameters if GPU is available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"CUDA optimizations enabled. Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Create directories for checkpoints and model output
os.makedirs("trainer", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Get available CPU cores and set num_workers for parallel processing
num_cpus = multiprocessing.cpu_count()
num_workers = max(1, num_cpus - 2)  # Leave 2 cores free for system processes
print(f"Using {num_workers} workers for parallel processing (out of {num_cpus} CPUs)")

# Get system memory info for automatic batch size selection
mem_info = psutil.virtual_memory()
total_ram_gb = mem_info.total / (1024 ** 3)
print(f"System RAM: {total_ram_gb:.2f} GB")

# Auto-adjust batch size based on available RAM
def get_optimal_batch_size():
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem < 4:
            return 16
        elif gpu_mem < 8:
            return 32
        else:
            return 64
    else:
        if total_ram_gb < 8:
            return 16
        elif total_ram_gb < 16:
            return 32
        else:
            return 64

# MODULE: Multi-head Attention
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

    def forward(self, qt, d_t):
        qt_hat = self.attn_question(qt, qt, qt)
        h_t = self.attn_state(qt_hat, qt_hat, d_t)
        if qt_hat.dim() == 3:
            return qt_hat[:, :-1, :], h_t[:, :-1, :]
        else:
            return qt_hat, h_t

# MODULE 3: Per-Concept Memory with Forget Gate + Time Decay
class MemoryUpdater(nn.Module):
    def __init__(self, dim_g):
        super().__init__()
        self.forget_gate = Linear(dim_g + 1, 1)
    def forward(self, memory_value, delta_t):
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)  # [num_c, 1]
        time_feat = delta_t.float()
        gate_input = torch.cat([memory_value, time_feat], dim=-1)
        forget_weight = torch.sigmoid(self.forget_gate(gate_input))
        return memory_value * forget_weight

# MODULE 4: Dynamic Latent Concept Graph Builder
class DynamicConceptGraphBuilder(nn.Module):
    def __init__(self, num_c, top_k):
        super().__init__()
        self.num_c = num_c
        self.top_k = top_k

    def forward(self, memory_value):
        normed = memory_value / memory_value.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sim_matrix = torch.matmul(normed, normed.T)
        mask = torch.zeros_like(sim_matrix)
        topk = torch.topk(sim_matrix, self.top_k + 1, dim=-1).indices
        for i in range(self.num_c):
            mask[i, topk[i]] = 1.0
        sim_matrix = sim_matrix * mask
        edge_index, edge_weight = dense_to_sparse(sim_matrix)
        return edge_index, edge_weight

# MODULE 5: Prediction
class AttentionBasedPredictor(nn.Module):
    def __init__(self, dim_q, dim_g):
        super().__init__()
        self.query_proj = Linear(dim_q, dim_g)
        self.sigmoid = Sigmoid()

    def forward(self, qt_hat, memory_key, memory_value):
        pred_query = self.query_proj(qt_hat)
        if pred_query.dim() == 2:
            sim = torch.einsum('bd,cd->bc', pred_query, memory_key)       # [1, num_c]
            attn = torch.softmax(sim, dim=-1)
            mastery = torch.einsum('bc,cd->bd', attn, memory_value)       # [1, dim_g]
            logits = torch.sum(pred_query * mastery, dim=-1)              # [1]
        else:
            sim = torch.einsum('btd,cd->btc', pred_query, memory_key)
            attn = torch.softmax(sim, dim=-1)
            mastery = torch.einsum('btc,cd->btd', attn, memory_value)
            logits = torch.sum(pred_query * mastery, dim=-1)
        return self.sigmoid(logits)

# Define DCRKT model 
class DCRKT(nn.Module):
    def __init__(self, num_c, num_q, num_o, dim_q, dim_g, num_heads, top_k, dropout):
        super().__init__()
        self.num_c = num_c
        self.num_q = num_q
        self.num_o = num_o
        self.dim_g = dim_g

        self.question_emb = Embedding(num_q + 1, dim_q)
        self.response_emb = Embedding(num_q * num_o + 2, dim_q, padding_idx=-1)

        self.encoder = DisentangledResponseEncoder(dim_q, dropout)
        self.retriever = KnowledgeRetriever(dim_q, num_heads, dropout)
        self.memory_updater = MemoryUpdater(dim_g)
        self.graph_builder = DynamicConceptGraphBuilder(num_c, top_k)
        self.predictor = AttentionBasedPredictor(dim_q, dim_g)

        self.memory_key = nn.Parameter(torch.randn(num_c, dim_g))

        # === Same as training.py: Per-student memory ===
        self.student_memory = defaultdict(lambda: torch.zeros(num_c, dim_g))
        self.last_update_time = defaultdict(lambda: torch.zeros(num_c))
        self.snapshots = defaultdict(list)  # dict[student_id] → list of Mv snapshot over time

        self.gat = GAT(in_channels=dim_g, hidden_channels=dim_g // 2,
                       out_channels=dim_g, num_layers=2, dropout=dropout)
        
        self.name = "DCRKT"

    def reset_memory(self, student_id):
        self.student_memory[student_id] = torch.zeros(self.num_c, self.dim_g)
        self.last_update_time[student_id] = torch.zeros(self.num_c)
        self.snapshots[student_id] = []

    def forward_single_step(self, student_id, q_idx, o_idx, u_idx, score, timestamp, concept_ids):
        q_idx = torch.clamp(q_idx, 0, self.num_q - 1)
        o_idx = torch.clamp(o_idx, 0, self.num_o - 1)
        u_idx = torch.clamp(u_idx, 0, self.num_o - 1)

        # === Tính chỉ số embedding ===
        response_idx = q_idx * self.num_o + o_idx
        unchosen_idx = q_idx * self.num_o + u_idx

        max_idx = self.response_emb.num_embeddings - 1
        if response_idx > max_idx or unchosen_idx > max_idx:
            raise ValueError(f"[Embedding Overflow] response_idx={response_idx}, unchosen_idx={unchosen_idx}, max={max_idx}")
            
        qt = self.question_emb(q_idx)
        ot = self.response_emb(q_idx * self.num_o + o_idx)
        ut = self.response_emb(q_idx * self.num_o + u_idx)

        d_t = self.encoder(ot.unsqueeze(0), ut.unsqueeze(0), score.unsqueeze(0))
        qt_hat, h_t = self.retriever(qt.unsqueeze(0), d_t)

        mk = self.memory_key
        mv = self.student_memory[student_id]  # [num_c, dim_g]
        last_time = self.last_update_time[student_id]  # [num_c]

        # === UPDATE MEMORY CHỈ TRÊN CONCEPT LIÊN QUAN ===
        current_time = timestamp.item()  # thời gian hiện tại
        delta_t = torch.zeros_like(last_time)

        for cid in concept_ids:
            delta_t[cid] = current_time - last_time[cid]
            last_time[cid] = current_time

        edge_index, edge_weight = self.graph_builder(mv)

        # ==== Đảm bảo tất cả tensor trên đúng device ====
        device = mk.device
        mv = mv.to(device)
        edge_index = edge_index.to(device)
        delta_t = delta_t.to(device)
        last_time = last_time.to(device)

        mv_propagated = self.gat(mv, edge_index)
        mv_propagated_updated = self.memory_updater(mv_propagated, delta_t)

        # Ghi kiến thức mới
        mv_updated = mv_propagated_updated.clone()
        h_update = h_t.squeeze(0)
        for cid in concept_ids:
            mv_updated[cid] = mv_propagated_updated[cid] + h_update
            
        # Cập nhật
        self.student_memory[student_id] = mv_updated.detach()
        self.last_update_time[student_id] = last_time
        self.snapshots[student_id].append(mv_updated.detach().clone())

        # === Dự đoán ===
        pred = self.predictor(qt_hat, mk, mv_updated)
        return pred.squeeze()

    def get_snapshot(self, student_id, step=-1):
        return self.snapshots[student_id][step] if self.snapshots[student_id] else None

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
def process_row_answer(row, option_map):
    qid = row["question_id"]
    aid = row["answer_choice_id"]
    if qid not in option_map or pd.isna(aid):
        return None
    try:
        return option_map[qid].index(aid)
    except ValueError:
        return None

def map_answers_parallel():
    global interaction_df, option_df

    option_df_sorted = option_df.sort_values(["question_id", "id"])
    option_map = option_df_sorted.groupby("question_id")["id"].apply(list).to_dict()
    
    # Split dataframe into chunks for parallel processing
    chunk_size = max(1000, len(interaction_df) // (num_workers * 2))
    chunks = [interaction_df.iloc[i:i+chunk_size] for i in range(0, len(interaction_df), chunk_size)]
    
    # Process chunks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        partial_func = partial(process_chunk_answers, option_map=option_map)
        results = list(executor.map(partial_func, chunks))
    
    # Combine results
    interaction_df["answer_idx"] = pd.concat(results)
    
    print(f"[map_answers_parallel] Processed {len(interaction_df)} rows with {num_workers} workers")

def process_chunk_answers(chunk, option_map):
    return chunk.apply(lambda x: process_row_answer(x, option_map), axis=1)

# ----------------------- MAP CONCEPT -----------------------
def process_row_concepts(qid, qkc_map, kc2idx):
    return [kc2idx[k] for k in qkc_map.get(qid, []) if k in kc2idx]

def map_concepts_parallel():
    global interaction_df, question_concept_df, kc2idx
    
    qkc_map = question_concept_df.groupby("question_id")["knowledgecomponent_id"].apply(list).to_dict()
    
    # Process in parallel
    chunk_size = max(1000, len(interaction_df) // (num_workers * 2))
    chunks = [interaction_df.iloc[i:i+chunk_size] for i in range(0, len(interaction_df), chunk_size)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        partial_func = partial(process_chunk_concepts, qkc_map=qkc_map, kc2idx=kc2idx)
        results = list(executor.map(partial_func, chunks))
    
    # Combine results
    interaction_df["concept_idxs"] = pd.concat(results)
    
    # Filter out rows with no concepts
    interaction_df = interaction_df[interaction_df["concept_idxs"].map(len) > 0]
    
    print(f"[map_concepts_parallel] Processed {len(interaction_df)} rows with {num_workers} workers")

def process_chunk_concepts(chunk, qkc_map, kc2idx):
    return chunk["question_id"].apply(lambda qid: process_row_concepts(qid, qkc_map, kc2idx))

# ----------------------- SORT EVENTS -----------------------
def sort_events():
    global interaction_df
    interaction_df.sort_values(["user_idx", "timestamp"], inplace=True)
    interaction_df.reset_index(drop=True, inplace=True)

# ----------------------- TẠO GRAPH KIẾN THỨC -----------------------
def create_concept_graph():
    relation_df["src_idx"] = relation_df["from_knowledgecomponent_id"].map(kc2idx)
    relation_df["tar_idx"] = relation_df["to_knowledgecomponent_id"].map(kc2idx)
    rel_df = relation_df.dropna()

    edges = list(zip(rel_df["src_idx"], rel_df["tar_idx"]))
    directed = []
    undirected = []

    for s, t in edges:
        if (t, s) in edges and s < t:
            undirected.append((s, t))
        elif (t, s) not in edges:
            directed.append((s, t))

    concept_graph = {"directed": directed, "undirected": undirected}
    print(f"[create_concept_graph] Directed: {len(directed)}, Undirected: {len(undirected)}")
    return concept_graph

# ----------------------- HÀM CHÍNH TIỀN XỬ LÝ -----------------------
def run_all_steps():
    clean_data()
    map_ids()
    process_events()
    
    # Use parallel processing for expensive operations
    print("Using parallel processing for data preparation...")
    map_answers_parallel()  # Parallel version
    map_concepts_parallel() # Parallel version
    
    sort_events()
    create_concept_graph()
    
    print(f"Dataset dimensions: {len(user_list)} users, {len(question_list)} questions, "
          f"{interaction_df['answer_idx'].nunique()} options, {len(kc_list)} concepts")
    
    return len(question_list), len(kc_list), interaction_df["answer_idx"].nunique()

# ----------------------- TRAINING FUNCTIONS -----------------------
def evaluate_metrics(y_true, y_pred, total_loss, total_samples):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_bin = (y_pred >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    acc = accuracy_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

    return {
        "AUC": round(auc, 4),
        "ACC": round(acc, 4),
        "F1": round(f1, 4),
        "Loss": round(avg_loss, 4),
    }

# Optimized dataset class for student interactions
class StudentInteractionDataset(Dataset):
    def __init__(self, user_data, max_seq_len=50):
        self.samples = []
        self.max_seq_len = max_seq_len
        
        # Create samples for each student
        for student_id, interactions in user_data.items():
            # Sort interactions by timestamp
            interactions = sorted(interactions, key=lambda x: x["timestamp"])
            
            # Process in chunks for more efficient training
            for i in range(0, len(interactions), max_seq_len):
                chunk = interactions[i:i+max_seq_len]
                if chunk:  # Ensure the chunk is not empty
                    self.samples.append({
                        "student_id": student_id,
                        "interactions": chunk
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Custom collate function for variable-length sequences
def custom_collate_fn(batch):
    return batch

# Optimized group_by_user function with chunking for memory efficiency
def group_by_user_optimized(df):
    user_data = defaultdict(list)
    
    # Process dataframe in chunks to save memory
    chunk_size = 10000
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(num_chunks), desc="Grouping by user"):
        chunk = df.iloc[i*chunk_size:(i+1)*chunk_size]
        
        for row in chunk.itertuples(index=False):
            try:
                q_idx = int(row.question_idx)
                o_idx = int(row.answer_idx) if pd.notna(row.answer_idx) else 0
                u_idx = int(row.unchosen_idx) if hasattr(row, "unchosen_idx") and pd.notna(row.unchosen_idx) else o_idx
                score = float(row.is_correct)
                timestamp = float(row.timestamp)
                concept_ids = eval(row.concept_idxs) if isinstance(row.concept_idxs, str) else row.concept_idxs

                user_data[row.user_idx].append({
                    "q_idx": q_idx,
                    "o_idx": o_idx,
                    "u_idx": u_idx,
                    "score": score,
                    "timestamp": timestamp,
                    "concept_ids": concept_ids
                })
            except Exception as e:
                print(f"[ERROR] Bỏ qua row lỗi: {row}")
                print(e)
        
        # Clear memory after each chunk
        if i % 5 == 0:
            gc.collect()
    
    print(f"Grouped {len(df)} interactions for {len(user_data)} students")
    return user_data

# Evaluate with DataLoader for better memory efficiency
def evaluate_dcrkt_with_dataloader(model, val_dataset, device, batch_size=32):
    model.eval()
    model.to(device)
    loss_fn = BCELoss()
    
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    y_true_all, y_pred_all = [], []
    total_loss, total_samples = 0.0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for sample in batch:
                student_id = sample["student_id"]
                interactions = sample["interactions"]
                
                # Reset student memory
                model.reset_memory(student_id)
                
                # Process each interaction
                for interaction in interactions:
                    try:
                        q_idx = torch.tensor(interaction["q_idx"], dtype=torch.long).to(device)
                        o_idx = torch.tensor(interaction["o_idx"], dtype=torch.long).to(device)
                        u_idx = torch.tensor(interaction["u_idx"], dtype=torch.long).to(device)
                        score = torch.tensor(interaction["score"], dtype=torch.float).to(device)
                        timestamp = torch.tensor(interaction["timestamp"], dtype=torch.float).to(device)
                        concept_ids = interaction["concept_ids"]
                        
                        pred = model.forward_single_step(
                            student_id=student_id,
                            q_idx=q_idx,
                            o_idx=o_idx,
                            u_idx=u_idx,
                            score=score,
                            timestamp=timestamp,
                            concept_ids=concept_ids
                        )
                        
                        y_true_all.append(score.item())
                        y_pred_all.append(pred.item())
                        total_loss += loss_fn(pred, score).item()
                        total_samples += 1
                    except Exception as e:
                        print(f"[EVAL ERROR]: {e}")
                        continue
    
    return evaluate_metrics(y_true_all, y_pred_all, total_loss, total_samples)

# === HÀM TẠO FOLD DỮ LIỆU ===
def get_fold_data(interaction_df, fold, n_splits=5):
    user_ids = interaction_df["user_idx"].unique()
    if fold >= n_splits or fold < 0:
        raise ValueError(f"Invalid fold index: {fold}. Must be between 0 and {n_splits-1}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(user_ids)):
        if i == fold:
            train_users = set(user_ids[train_idx])
            val_users = set(user_ids[val_idx])
            train_df = interaction_df[interaction_df["user_idx"].isin(train_users)].reset_index(drop=True)
            val_df = interaction_df[interaction_df["user_idx"].isin(val_users)].reset_index(drop=True)
            return train_df, val_df
    raise RuntimeError("Fold not found in KFold split.")

# Optimized training function using DataLoader
def train_dcrkt_optimized(args, interaction_df, fold):
    device = args.device
    batch_size = args.batch
    
    # Create train/val split
    train_df, val_df = get_fold_data(interaction_df, fold, n_splits=args.k_fold)
    print(f"Split data: {len(train_df)} training samples, {len(val_df)} validation samples")
    
    # Group by user
    print("Grouping training data by user...")
    train_user_data = group_by_user_optimized(train_df)
    
    print("Grouping validation data by user...")
    val_user_data = group_by_user_optimized(val_df)
    
    # Free memory
    del train_df, val_df
    gc.collect()
    
    # Initialize model - use the model definition matching training.py
    model = DCRKT(
        num_c=args.concept,
        num_q=args.question,
        num_o=args.option,
        dim_q=args.q_dim,
        dim_g=args.g_dim,
        num_heads=args.heads,
        top_k=args.top_k,
        dropout=args.dropout
    ).to(device)
    
    # Initialize optimizer with same parameters as training.py
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = BCELoss()
    
    # Learning rate scheduler - simpler version to match training.py
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    best_val_loss = float("inf")
    patience = 0
    
    # Free memory before training
    torch.cuda.empty_cache()
    gc.collect()
    
    # === Training loop - closer to training.py approach ===
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Process each student sequentially, like in training.py
        for student_id, interactions in tqdm(train_user_data.items(), desc=f"[Fold {fold}] Epoch {epoch+1}"):
            model.reset_memory(student_id)
            interactions = sorted(interactions, key=lambda x: x["timestamp"])
            
            # Prepare batch tensors for all interactions of this student
            q_idx = torch.tensor([s["q_idx"] for s in interactions], dtype=torch.long, device=device)
            o_idx = torch.tensor([s["o_idx"] for s in interactions], dtype=torch.long, device=device)
            u_idx = torch.tensor([s["u_idx"] for s in interactions], dtype=torch.long, device=device)
            score = torch.tensor([s["score"] for s in interactions], dtype=torch.float, device=device)
            timestamp = torch.tensor([s["timestamp"] for s in interactions], dtype=torch.float, device=device)
            concept_ids_list = [s["concept_ids"] for s in interactions]
            
            # Process each step sequentially, like in training.py
            for i in range(len(interactions)):
                pred = model.forward_single_step(
                    student_id=student_id,
                    q_idx=q_idx[i],
                    o_idx=o_idx[i],
                    u_idx=u_idx[i],
                    score=score[i],
                    timestamp=timestamp[i],
                    concept_ids=concept_ids_list[i]
                )
                
                loss = loss_fn(pred, score[i])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_samples += 1
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Evaluate on validation set
        print("\nEvaluating model...")
        val_metrics = evaluate_dcrkt(model, val_user_data, device)
        val_loss = val_metrics["Loss"]
        
        # Print metrics
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Metrics: AUC={val_metrics['AUC']:.4f}, ACC={val_metrics['ACC']:.4f}, F1={val_metrics['F1']:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            print(f"Mô hình improved tại epoch {epoch+1} (Val Loss: {val_loss:.4f})")
            # Save to both locations for compatibility
            torch.save(model.state_dict(), f"trainer/dcrkt_model_fold_{fold}.pt")
            torch.save(model.state_dict(), f"dcrkt_model_fold_{fold}.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping tại epoch {epoch+1}")
                break
        
        # Clear memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    return best_val_loss, val_metrics, model

# Use the evaluate_dcrkt function from training.py
def evaluate_dcrkt(model, user_data, device):
    model.eval()
    model.to(device)
    loss_fn = BCELoss()

    y_true_all, y_pred_all = [], []
    total_loss, total_samples = 0.0, 0

    for student_id, interactions in user_data.items():
        model.reset_memory(student_id)
        interactions = sorted(interactions, key=lambda x: x["timestamp"])

        for sample in interactions:
            try:
                q_idx = torch.tensor(sample["q_idx"], dtype=torch.long).to(device)
                o_idx = torch.tensor(sample["o_idx"], dtype=torch.long).to(device)
                u_idx = torch.tensor(sample["u_idx"], dtype=torch.long).to(device)
                score = torch.tensor(sample["score"], dtype=torch.float).to(device)
                timestamp = torch.tensor(sample["timestamp"], dtype=torch.float).to(device)
                concept_ids = sample["concept_ids"]

                with torch.no_grad():
                    pred = model.forward_single_step(
                        student_id=student_id,
                        q_idx=q_idx,
                        o_idx=o_idx,
                        u_idx=u_idx,
                        score=score,
                        timestamp=timestamp,
                        concept_ids=concept_ids
                    )

                y_true_all.append(score.item())
                y_pred_all.append(pred.item())
                total_loss += loss_fn(pred, score).item()
                total_samples += 1
            except Exception as e:
                print("[EVAL ERROR]:", e)
                continue

    return evaluate_metrics(y_true_all, y_pred_all, total_loss, total_samples)

def visualize_student_knowledge(student_id, model):
    """Visualize a student's knowledge state (mastery levels)"""
    snapshot = model.get_snapshot(student_id)
    if snapshot is None:
        print(f"No snapshot available for student {student_id}")
        return
        
    # Get concept names from global variables
    student_df = interaction_df[interaction_df["user_idx"] == student_id]
    
    # Get the concepts this student has seen
    learned_concepts = set()
    for row in student_df["concept_idxs"]:
        learned_concepts.update(row)
    
    # Display results
    print(f"\n Học sinh {student_id} đã học các khái niệm sau:")
    print("-" * 70)
    print(f"{'Concept Index':<15} | {'Tên khái niệm':<40} | {'‖Mv‖'}")
    print("-" * 70)
    
    for cid in sorted(learned_concepts):
        try:
            concept_id_real = kc_list[cid]
            name = concept_df[concept_df["id"] == concept_id_real]["name"].values[0]
            mv_norm = snapshot[cid].norm().item()
            print(f"{cid:<15} | {name:<40} | {mv_norm:.4f}")
        except Exception as e:
            print(f"{cid:<15} | [ lỗi]: {e}")

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data from raw files
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

    # Process data
    print("Processing dataset...")
    num_questions, num_concepts, num_options = run_all_steps()
    
    # Set up training arguments to match training.py
    print("Setting up training configuration...")
    args = Namespace(
        device=device,
        k_fold=2,  # Lower value for faster training
        batch=32,  # Same as training.py
        epoch=1,   # Lower value for faster training
        lr=1e-3,   # Same as training.py
        patience=3,
        q_dim=64,  # Same as training.py
        g_dim=64,  # Same as training.py
        heads=4,   # Same as training.py
        top_k=5,   # Same as training.py
        dropout=0.2, # Same as training.py
        question=num_questions,
        option=num_options,
        concept=num_concepts
    )

    print(f"\n=== Training Configuration ===")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch}")
    print(f"K-Fold: {args.k_fold}")
    print(f"Epochs: {args.epoch}")
    print(f"Learning Rate: {args.lr}")
    print(f"Dataset: {len(interaction_df)} interactions, {len(user_list)} users")
    print(f"Model dimensions: Q={args.q_dim}, G={args.g_dim}, Heads={args.heads}")

    # Run training with sequential fold processing (like training.py)
    print(f"\n=== Starting Training ===")
    val_results = []
    val_losses = []

    for fold in range(args.k_fold):
        print(f"\n=== Training Fold {fold+1}/{args.k_fold} ===")
        val_loss, val_metrics, trained_model = train_dcrkt_optimized(args, interaction_df, fold)
        val_metrics["Fold"] = fold + 1
        val_results.append(val_metrics)
        val_losses.append(val_loss)
        
        # Clear memory between folds
        gc.collect()
        torch.cuda.empty_cache()

    # Print summary of results
    df = pd.DataFrame(val_results)
    print("\n=== Final Results ===")
    print("\nKết quả trung bình các fold:")
    print(df.drop(columns=["Fold"]).mean().round(4))

    print("\nĐộ lệch chuẩn các fold:")
    print(df.drop(columns=["Fold"]).std().round(4))
    print(df.round(4))
    
    # Demo: Visualize a student's knowledge state (using approach from training.py)
    try:
        # Find the fold with the best validation loss
        best_fold = np.argmin(val_losses)
        print(f"\n=== Using best model from fold {best_fold+1} for visualization ===")
        print("\nHiển thị trạng thái kiến thức của học sinh 39:")
        
        # Load model
        best_model = DCRKT(
            num_c=args.concept,
            num_q=args.question,
            num_o=args.option,
            dim_q=args.q_dim,
            dim_g=args.g_dim,
            num_heads=args.heads,
            top_k=args.top_k,
            dropout=args.dropout
        ).to("cpu")  # Use CPU for visualization
        
        # Try both possible model paths
        try:
            best_model.load_state_dict(torch.load(f"trainer/dcrkt_model_fold_{best_fold}.pt", map_location="cpu"))
        except:
            best_model.load_state_dict(torch.load(f"dcrkt_model_fold_{best_fold}.pt", map_location="cpu"))
        
        best_model.eval()
        
        # Process student 39 data exactly like in training.py
        student_id = 39
        best_model.reset_memory(student_id)
        
        # Get student history
        student_df = interaction_df[interaction_df["user_idx"] == student_id].sort_values("timestamp")
        
        # Process each interaction
        for _, row in student_df.iterrows():
            q_idx = int(row["question_idx"])
            o_idx = int(row["answer_idx"]) if pd.notna(row["answer_idx"]) else 0
            u_idx = max(0, o_idx - 1)
            score = int(row["is_correct"])
            timestamp = float(row["timestamp"])
            concept_ids = row["concept_idxs"]
            
            # Process each step exactly like in training.py
            best_model.forward_single_step(
                student_id=student_id,
                q_idx=torch.tensor(q_idx),
                o_idx=torch.tensor(o_idx),
                u_idx=torch.tensor(u_idx),
                score=torch.tensor(score),
                timestamp=torch.tensor(timestamp),
                concept_ids=concept_ids
            )
        
        # Visualize student knowledge
        visualize_student_knowledge(student_id, best_model)
                
    except Exception as e:
        print(f"Error visualizing student knowledge: {e}") 