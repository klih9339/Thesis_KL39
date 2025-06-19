import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import (Embedding, Sequential, Linear, Sigmoid, ReLU, Dropout, LayerNorm)
from torch_geometric.nn import GAT
from torch_geometric.utils import dense_to_sparse
from collections import defaultdict

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

        # Per-student memory
        self.student_memory = defaultdict(lambda: torch.zeros(num_c, dim_g))
        self.last_update_time = defaultdict(lambda: torch.zeros(num_c))
        self.snapshots = defaultdict(list)  # dict[student_id] → list of Mv snapshot over time

        self.gat = GAT(in_channels=dim_g, hidden_channels=dim_g // 2,
                       out_channels=dim_g, num_layers=2, dropout=dropout)

    def reset_memory(self, student_id):
        self.student_memory[student_id] = torch.zeros(self.num_c, self.dim_g)
        self.last_update_time[student_id] = torch.zeros(self.num_c)
        self.snapshots[student_id] = []

    def forward_single_step(self, student_id, q_idx, o_idx, u_idx, score, timestamp, concept_ids):
        # Ensure all inputs are on the same device as the model
        device = self.memory_key.device
        q_idx = q_idx.to(device)
        o_idx = o_idx.to(device)
        u_idx = u_idx.to(device)
        score = score.to(device)
        timestamp = timestamp.to(device)
        
        q_idx = torch.clamp(q_idx, 0, self.num_q - 1)
        o_idx = torch.clamp(o_idx, 0, self.num_o - 1)
        u_idx = torch.clamp(u_idx, 0, self.num_o - 1)

        # Calculate embedding indices
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
        mv = self.student_memory[student_id].to(device)  # [num_c, dim_g]
        last_time = self.last_update_time[student_id].to(device)  # [num_c]

        # Update memory only for related concepts
        current_time = timestamp.item()  # current time
        delta_t = torch.zeros_like(last_time)

        for cid in concept_ids:
            delta_t[cid] = current_time - last_time[cid]
            last_time[cid] = current_time

        edge_index, edge_weight = self.graph_builder(mv)

        # Ensure all tensors are on the correct device
        edge_index = edge_index.to(device)
        delta_t = delta_t.to(device)
        last_time = last_time.to(device)

        mv_propagated = self.gat(mv, edge_index)
        mv_propagated_updated = self.memory_updater(mv_propagated, delta_t)

        # Write new knowledge
        mv_updated = mv_propagated_updated.clone()
        h_update = h_t.squeeze(0)
        for cid in concept_ids:
            mv_updated[cid] = mv_propagated_updated[cid] + h_update
        # Update
        self.student_memory[student_id] = mv_updated.detach().cpu()  # Store on CPU for memory efficiency
        self.last_update_time[student_id] = last_time.cpu()
        self.snapshots[student_id].append(mv_updated.detach().clone().cpu())

        # Prediction
        pred = self.predictor(qt_hat, mk, mv_updated)
        return pred.squeeze()

    def get_snapshot(self, student_id, step=-1):
        return self.snapshots[student_id][step] if self.snapshots[student_id] else None 