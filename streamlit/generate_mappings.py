import pandas as pd
import json
import os
import numpy as np

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_mappings():
    print("Loading data files...")
    folder_path = 'input/raw'
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Directory {folder_path} does not exist")
        return
    
    try:
        # Load data
        question_df = pd.read_csv(f"{folder_path}/Questions.csv")
        option_df = pd.read_csv(f"{folder_path}/Question_Choices.csv")
        concept_df = pd.read_csv(f"{folder_path}/KCs.csv")
        interaction_df = pd.read_csv(f"{folder_path}/Transaction.csv")
        
        # Clean data
        question_df.drop_duplicates(subset=["id"], inplace=True)
        concept_df.drop_duplicates(subset=["id"], inplace=True)
        option_df.drop_duplicates(subset=["id", "question_id"], inplace=True)
        interaction_df.drop_duplicates(subset=["student_id", "question_id", "start_time"], inplace=True)
        
        # Get unique lists
        user_list = sorted(interaction_df["student_id"].unique())
        question_list = sorted(question_df["id"].unique())
        kc_list = sorted(concept_df["id"].unique())
        
        # Create mappings - convert numpy int64 to Python int
        user2idx = {int(u) if isinstance(u, np.integer) else u: i for i, u in enumerate(user_list)}
        question2idx = {int(q) if isinstance(q, np.integer) else q: i for i, q in enumerate(question_list)}
        kc2idx = {int(k) if isinstance(k, np.integer) else k: i for i, k in enumerate(kc_list)}
        
        # Count options
        num_options = int(option_df.groupby("question_id").size().max())
        
        # Create mappings object
        mappings = {
            "num_questions": len(question_list),
            "num_concepts": len(kc_list),
            "num_options": num_options,
            "user2idx": user2idx,
            "question2idx": question2idx,
            "kc2idx": kc2idx
        }
        
        # Save to file using custom encoder
        with open("mappings.json", "w") as f:
            json.dump(mappings, f, cls=NumpyEncoder)
        
        print(f"Mappings saved to mappings.json")
        print(f"Found {len(user_list)} users, {len(question_list)} questions, {len(kc_list)} concepts, {num_options} max options")
        
    except Exception as e:
        print(f"Error generating mappings: {e}")

if __name__ == "__main__":
    generate_mappings() 