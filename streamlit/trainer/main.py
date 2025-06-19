import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from .knowledge_map import KnowledgeMapVisualizer
from .active_recall import ActiveRecallPlanner
from .dashboard import KnowledgeDashboard

def load_data(folder_path="input/raw"):
    """
    Load all necessary data from raw files
    """
    dfs = {}
    dfs["question_df"] = pd.read_csv(f"{folder_path}/Questions.csv")
    dfs["option_df"] = pd.read_csv(f"{folder_path}/Question_Choices.csv")
    dfs["concept_df"] = pd.read_csv(f"{folder_path}/KCs.csv")
    dfs["relation_df"] = pd.read_csv(f"{folder_path}/KC_Relationships.csv")
    dfs["question_concept_df"] = pd.read_csv(f"{folder_path}/Question_KC_Relationships.csv")
    dfs["interaction_df"] = pd.read_csv(f"{folder_path}/Transaction.csv")
    dfs["specialization_df"] = pd.read_csv(f"{folder_path}/Specialization.csv")
    dfs["student_specialization_df"] = pd.read_csv(f"{folder_path}/Student_Specialization.csv")
    
    # Clean data
    dfs["question_df"].drop_duplicates(subset=["id"], inplace=True)
    dfs["question_concept_df"].drop_duplicates(subset=["question_id", "knowledgecomponent_id"], inplace=True)
    dfs["concept_df"].drop_duplicates(subset=["id"], inplace=True)
    dfs["option_df"].drop_duplicates(subset=["id", "question_id"], inplace=True)
    dfs["interaction_df"].drop_duplicates(subset=["student_id", "question_id", "start_time"], inplace=True)
    dfs["relation_df"].drop_duplicates(subset=["from_knowledgecomponent_id", "to_knowledgecomponent_id"], inplace=True)

    # Fill missing values
    dfs["concept_df"]["name"] = dfs["concept_df"]["name"].fillna("Unknown Concept")
    dfs["question_df"]["question_text"] = dfs["question_df"]["question_text"].fillna("")
    dfs["option_df"]["choice_text"] = dfs["option_df"]["choice_text"].fillna("")
    
    return dfs

def preprocess_data(dfs):
    """
    Preprocess data and create mappings
    """
    # Process time in interaction_df
    if not pd.api.types.is_datetime64_any_dtype(dfs["interaction_df"]["start_time"]):
        dfs["interaction_df"]["start_time"] = pd.to_datetime(dfs["interaction_df"]["start_time"], errors="coerce")
    
    dfs["interaction_df"].dropna(subset=["start_time"], inplace=True)
    if pd.api.types.is_datetime64tz_dtype(dfs["interaction_df"]["start_time"]):
        dfs["interaction_df"]["start_time"] = dfs["interaction_df"]["start_time"].dt.tz_localize(None)
    
    # Filter out rows with missing important fields
    dfs["interaction_df"].dropna(subset=["student_id", "question_id", "answer_state"], inplace=True)
    
    # Create ID mappings
    mappings = {}
    mappings["user_list"] = sorted(dfs["interaction_df"]["student_id"].unique())
    mappings["question_list"] = sorted(dfs["question_df"]["id"].unique())
    mappings["kc_list"] = sorted(dfs["concept_df"]["id"].unique())
    
    mappings["user2idx"] = {u: i for i, u in enumerate(mappings["user_list"])}
    mappings["question2idx"] = {q: i for i, q in enumerate(mappings["question_list"])}
    mappings["kc2idx"] = {k: i for i, k in enumerate(mappings["kc_list"])}
    
    # Add additional fields to interaction_df
    dfs["interaction_df"]["user_idx"] = dfs["interaction_df"]["student_id"].map(mappings["user2idx"])
    dfs["interaction_df"]["question_idx"] = dfs["interaction_df"]["question_id"].map(mappings["question2idx"])
    dfs["interaction_df"]["timestamp"] = dfs["interaction_df"]["start_time"].apply(lambda x: int(x.timestamp()))
    dfs["interaction_df"]["is_correct"] = dfs["interaction_df"]["answer_state"].astype(int)
    
    # Map answers
    option_df_sorted = dfs["option_df"].sort_values(["question_id", "id"])
    option_map = option_df_sorted.groupby("question_id")["id"].apply(list).to_dict()
    
    def convert_answer(qid, aid):
        if qid not in option_map or pd.isna(aid):
            return 0  # Default to first option
        try:
            return option_map[qid].index(aid)
        except ValueError:
            return 0
    
    dfs["interaction_df"]["answer_idx"] = dfs["interaction_df"].apply(
        lambda x: convert_answer(x["question_id"], x["answer_choice_id"]), axis=1
    )
    
    # Map concepts
    qkc_map = dfs["question_concept_df"].groupby("question_id")["knowledgecomponent_id"].apply(list).to_dict()
    
    dfs["interaction_df"]["concept_idxs"] = dfs["interaction_df"]["question_id"].apply(
        lambda qid: [mappings["kc2idx"][k] for k in qkc_map.get(qid, []) if k in mappings["kc2idx"]]
    )
    
    # Filter out rows with no concepts
    dfs["interaction_df"] = dfs["interaction_df"][dfs["interaction_df"]["concept_idxs"].map(len) > 0]
    
    # Sort by user and timestamp
    dfs["interaction_df"].sort_values(["user_idx", "timestamp"], inplace=True)
    dfs["interaction_df"].reset_index(drop=True, inplace=True)
    
    return dfs, mappings

def load_model(model_path, num_c, num_q, num_o, device="cpu"):
    """
    Load trained DCRKT model
    """
    from .dcrkt_model import DCRKT
    
    # Create model with the same parameters as training
    model = DCRKT(
        num_c=num_c,
        num_q=num_q,
        num_o=num_o,
        dim_q=64,
        dim_g=64,
        num_heads=4,
        top_k=5,
        dropout=0.2
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def prepare_for_student(model, student_id, dfs, mappings):
    """
    Prepare model for a specific student by loading their interaction history
    """
    # Get student index
    student_idx = student_id
    if student_id in mappings["user2idx"]:
        student_idx = mappings["user2idx"][student_id]
    
    # Reset student memory
    model.reset_memory(student_idx)
    
    # Get student history
    student_df = dfs["interaction_df"][dfs["interaction_df"]["user_idx"] == student_idx]
    student_df = student_df.sort_values("timestamp")
    
    # Get the device from the model
    device = next(model.parameters()).device
    
    # Feed history to model
    for _, row in student_df.iterrows():
        q_idx = int(row["question_idx"])
        o_idx = int(row["answer_idx"])
        u_idx = max(0, o_idx - 1)  # Approximation of unchosen option
        score = int(row["is_correct"])
        timestamp = float(row["timestamp"])
        concept_ids = row["concept_idxs"]
        
        try:
            model.forward_single_step(
                student_id=student_idx,
                q_idx=torch.tensor(q_idx, device=device),
                o_idx=torch.tensor(o_idx, device=device),
                u_idx=torch.tensor(u_idx, device=device),
                score=torch.tensor(score, device=device),
                timestamp=torch.tensor(timestamp, device=device),
                concept_ids=concept_ids
            )
        except Exception as e:
            print(f"Error processing interaction: {e}")
            continue
    
    return model

def init_trainer(model_path=None, data_folder="input/raw", device="cpu"):
    """
    Initialize the trainer system
    """
    # Load and preprocess data
    dfs = load_data(data_folder)
    dfs, mappings = preprocess_data(dfs)
    
    # Get counts for model initialization
    num_c = len(mappings["kc_list"])
    num_q = len(mappings["question_list"])
    num_o = dfs["interaction_df"]["answer_idx"].max() + 1
    
    # Load model or create new one
    if model_path and os.path.exists(model_path):
        model = load_model(model_path, num_c, num_q, num_o, device)
    else:
        from .dcrkt_model import DCRKT
        model = DCRKT(
            num_c=num_c,
            num_q=num_q,
            num_o=num_o,
            dim_q=64,
            dim_g=64,
            num_heads=4,
            top_k=5,
            dropout=0.2
        ).to(device)
        print("Created new model (not trained)")
    
    # Create dashboard
    dashboard = KnowledgeDashboard(model, dfs, mappings)
    
    return model, dashboard, dfs, mappings

def create_interactive_session():
    """
    Create interactive session with the dashboard
    """
    # Try to use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize trainer with model
    model_path = "dcrkt_model_fold_0.pt"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Looking for model in checkpoints folder...")
        model_path = "checkpoints/dcrkt_model_fold_0.pt"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            model_path = None
    
    model, dashboard, dfs, mappings = init_trainer(model_path, device=device)
    
    # Create dashboard
    try:
        # If running in Jupyter notebook, show interactive dashboard
        from IPython.display import display
        display(dashboard.interactive_dashboard())
    except ImportError:
        # If not in Jupyter notebook, show example usage
        print("Interactive dashboard requires Jupyter notebook.")
        print("Example usage:")
        print("1. Prepare model for a student:")
        print("   model = prepare_for_student(model, student_id, dfs, mappings)")
        print("2. Display knowledge map:")
        print("   dashboard.display_knowledge_map(student_id, save=True)")
        print("3. Get study plan:")
        print("   dashboard.display_study_plan(student_id)")
        print("4. Predict performance:")
        print("   dashboard.display_prediction(student_id, question_id)")
    
    return model, dashboard, dfs, mappings

# If this file is run directly
if __name__ == "__main__":
    model, dashboard, dfs, mappings = create_interactive_session() 