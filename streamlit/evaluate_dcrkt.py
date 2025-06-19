import numpy as np
import pandas as pd
import torch
from torch.nn import BCELoss
from collections import defaultdict
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import time

# Force CPU use
device = torch.device("cpu")
torch.set_num_threads(4)  # Limit number of threads for better performance

# Import model definitions from test_model.py
from test_model import DCRKT, load_model, load_data

def evaluate_metrics(y_true, y_pred, total_loss, total_samples):
    """Calculate evaluation metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_bin = (y_pred >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    acc = accuracy_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

    return {
        "AUC": round(auc, 4),
        "ACC": round(acc, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Loss": round(avg_loss, 4),
    }

def evaluate_model(model, df, concept_name_map=None, max_students=None):
    """Evaluate model on dataset"""
    model.eval()
    loss_fn = BCELoss()

    # Group data by student
    student_groups = df.groupby("user_idx")
    
    # Limit number of students if specified
    if max_students and max_students < len(student_groups):
        student_ids = list(student_groups.groups.keys())[:max_students]
    else:
        student_ids = list(student_groups.groups.keys())
    
    y_true_all, y_pred_all = [], []
    total_loss, total_samples = 0.0, 0
    
    # For calculating per-concept performance
    concept_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "sum_pred": 0.0})
    
    start_time = time.time()
    
    print(f"Evaluating on {len(student_ids)} students...")
    
    for i, student_id in enumerate(student_ids):
        if i % 10 == 0:
            print(f"Processing student {i+1}/{len(student_ids)}")
        
        # Get student data
        student_df = student_groups.get_group(student_id).sort_values("timestamp")
        
        # Reset student memory
        model.reset_memory(student_id)
        
        # Process each interaction
        for _, row in student_df.iterrows():
            try:
                qid = int(row["question_idx"])
                oid = int(row["answer_idx"]) if pd.notna(row["answer_idx"]) else 0
                uid = oid  # Use answer as unchosen for simplicity
                score = float(row["is_correct"])
                timestamp = float(row["timestamp"])
                concept_ids = row["concept_idxs"]
                
                q_idx = torch.tensor(qid, dtype=torch.long, device=device)
                o_idx = torch.tensor(oid, dtype=torch.long, device=device)
                u_idx = torch.tensor(uid, dtype=torch.long, device=device)
                score_tensor = torch.tensor(score, dtype=torch.float, device=device)
                timestamp_tensor = torch.tensor(timestamp, dtype=torch.float, device=device)
                
                with torch.no_grad():
                    pred = model.forward_single_step(
                        student_id=student_id,
                        q_idx=q_idx,
                        o_idx=o_idx,
                        u_idx=u_idx,
                        score=score_tensor,
                        timestamp=timestamp_tensor,
                        concept_ids=concept_ids
                    )
                
                # Update metrics
                pred_val = pred.item()
                y_true_all.append(score)
                y_pred_all.append(pred_val)
                total_loss += loss_fn(pred, score_tensor).item()
                total_samples += 1
                
                # Update per-concept metrics
                for cid in concept_ids:
                    concept_metrics[cid]["total"] += 1
                    concept_metrics[cid]["correct"] += int(score)
                    concept_metrics[cid]["sum_pred"] += pred_val
                
            except Exception as e:
                print(f"Error processing interaction: {e}")
                continue
    
    end_time = time.time()
    
    # Calculate overall metrics
    metrics = evaluate_metrics(y_true_all, y_pred_all, total_loss, total_samples)
    metrics["Time"] = round(end_time - start_time, 2)
    metrics["Students"] = len(student_ids)
    metrics["Samples"] = total_samples
    
    # Calculate per-concept metrics
    concept_performance = {}
    for cid, data in concept_metrics.items():
        if data["total"] > 10:  # Only include concepts with enough samples
            concept_name = concept_name_map.get(cid, f"Concept {cid}") if concept_name_map else f"Concept {cid}"
            concept_performance[concept_name] = {
                "Accuracy": round(data["correct"] / data["total"], 4),
                "Avg Prediction": round(data["sum_pred"] / data["total"], 4),
                "Samples": data["total"]
            }
    
    return metrics, concept_performance

def plot_concept_performance(concept_performance, output_dir="."):
    """Plot concept performance metrics"""
    # Skip if no concepts
    if not concept_performance:
        print("No concept performance data to plot")
        return
    
    # Sort concepts by accuracy
    sorted_concepts = sorted(
        concept_performance.items(),
        key=lambda x: x[1]["Accuracy"],
        reverse=True
    )
    
    # Only show top 20 concepts for readability
    sorted_concepts = sorted_concepts[:20]
    
    concepts = [c[:20] + "..." if len(c) > 20 else c for c, _ in sorted_concepts]
    accuracies = [p["Accuracy"] for _, p in sorted_concepts]
    predictions = [p["Avg Prediction"] for _, p in sorted_concepts]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    x = np.arange(len(concepts))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, accuracies, width, label='Actual Accuracy')
    plt.bar(x + width/2, predictions, width, label='Predicted Probability')
    
    # Add labels and title
    plt.xlabel('Concepts')
    plt.ylabel('Score')
    plt.title('Concept Performance: Actual vs. Predicted')
    plt.xticks(x, concepts, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "concept_performance.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DCRKT model")
    parser.add_argument("--model", default="dcrkt_model_fold_0.pt", help="Path to model file")
    parser.add_argument("--data", default="data/raw/processed_data.csv", help="Path to processed data")
    parser.add_argument("--max_students", type=int, default=10, help="Maximum number of students to evaluate")
    parser.add_argument("--output", default=".", help="Directory to save results")
    args = parser.parse_args()
    
    # Paths
    model_path = args.model
    data_path = args.data
    output_dir = args.output
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found")
        return
    
    # Load model and mappings
    print(f"Loading model from {model_path}...")
    model, user2idx, question2idx, concept_names = load_model(model_path, data_path)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    # Evaluate model
    print("Evaluating model...")
    metrics, concept_performance = evaluate_model(
        model, df, concept_names, args.max_students
    )
    
    # Print overall metrics
    print("\nOverall Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)
    
    # Save concept performance to CSV
    if concept_performance:
        concept_df = pd.DataFrame.from_dict(concept_performance, orient='index')
        concept_df.reset_index(inplace=True)
        concept_df.rename(columns={'index': 'Concept'}, inplace=True)
        concept_df.to_csv(os.path.join(output_dir, "concept_performance.csv"), index=False)
        
        # Plot concept performance
        plot_concept_performance(concept_performance, output_dir)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main() 