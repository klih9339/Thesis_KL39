
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

