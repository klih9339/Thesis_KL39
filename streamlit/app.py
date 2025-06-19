import streamlit as st
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from new_train_dcrkt import DCRKT, visualize_student_knowledge
import os
import json
import plotly.graph_objects as go
import openai
from datetime import datetime, timedelta
import math


# Configure page
st.set_page_config(layout="wide", page_title="DCRKT Knowledge Tracing System")

# Load data
@st.cache_data
def load_data():
    folder_path = 'input/raw'
    question_df = pd.read_csv(f"{folder_path}/Questions.csv")
    option_df = pd.read_csv(f"{folder_path}/Question_Choices.csv")
    concept_df = pd.read_csv(f"{folder_path}/KCs.csv")
    relation_df = pd.read_csv(f"{folder_path}/KC_Relationships.csv")
    question_concept_df = pd.read_csv(f"{folder_path}/Question_KC_Relationships.csv")
    interaction_df = pd.read_csv(f"{folder_path}/Transaction.csv")
    
    # Process timestamps
    interaction_df["start_time"] = pd.to_datetime(interaction_df["start_time"], errors="coerce")
    
    return question_df, option_df, concept_df, relation_df, question_concept_df, interaction_df

# Load model - updated to use a button for loading
def load_model_from_file(model_path):
    try:
        # Load mappings
        mappings = {}
        if os.path.exists("mappings.json"):
            with open("mappings.json", "r") as f:
                mappings = json.load(f)
        else:
            st.warning("Mappings file not found. Using default values.")
            mappings = {
                "num_questions": 500,
                "num_concepts": 100,
                "num_options": 5,
                "user2idx": {},
                "question2idx": {},
                "kc2idx": {}
            }
        
        # Initialize model with same parameters as in new_train_dcrkt.py
        model = DCRKT(
            num_c=mappings["num_concepts"],
            num_q=mappings["num_questions"],
            num_o=mappings["num_options"],
            dim_q=64,  # Same as in new_train_dcrkt.py
            dim_g=64,  # Same as in new_train_dcrkt.py
            num_heads=4,  # Same as in new_train_dcrkt.py
            top_k=5,  # Same as in new_train_dcrkt.py
            dropout=0.2  # Same as in new_train_dcrkt.py
        )
        
        # Load model weights
        st.info(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model, mappings
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_resource
def get_default_model():
    # This is just a placeholder for the cache_resource decorator
    # The actual loading happens in the main function with the button
    return None, None

# Visualization Functions
def create_knowledge_graph(student_memory, concept_df, kc2idx, kc_list, relations_df=None, related_concepts=None, include_connected=True, show_all=False):
    """
    Create a knowledge graph for visualization
    
    Parameters:
    - student_memory: Memory vectors for student
    - concept_df: DataFrame of concepts
    - kc2idx: Mapping from concept IDs to indices
    - kc_list: List of concept IDs
    - relations_df: DataFrame of relations between concepts (can be None to exclude prerequisite relationships)
    - related_concepts: Optional list of concept indices to filter (related to current question)
    - include_connected: If True and related_concepts is provided, also include concepts connected to related_concepts
    """
    G = nx.DiGraph()
    
    # Handle edge case where student_memory is None or empty
    if student_memory is None or len(student_memory) == 0:
        print("DEBUG: student_memory is None or empty, returning empty graph")
        return G
    
    # Calculate norms for all concepts - use the actual memory vector norm
    try:
        memory_norms = [tensor.norm().item() for tensor in student_memory]
    except Exception as e:
        print(f"DEBUG: Error calculating memory norms: {e}")
        memory_norms = []
    
    # If no valid norms, return empty graph
    if not memory_norms:
        print("DEBUG: No valid memory norms, returning empty graph")
        return G
    
    # Create a mapping from concept index to concept name for faster lookup
    idx_to_name = {}
    for idx, norm in enumerate(memory_norms):
        if idx >= len(kc_list):
            continue
            
        # kc_list[idx] gives us the real concept ID from the CSV
        concept_id_real = kc_list[idx]
        # Get concept name from the CSV file using the real concept ID
        concept_row = concept_df[concept_df["id"] == concept_id_real]
        if not concept_row.empty:
            concept_name = concept_row["name"].values[0]
        else:
            concept_name = f"Concept {concept_id_real}"
        
        idx_to_name[idx] = concept_name
    
    # If related_concepts is provided, include all of them directly
    connected_concepts = set()
    if related_concepts is not None:
        # Add all related concepts directly (this includes extended concepts from relationships)
        connected_concepts.update(related_concepts)
        print(f"DEBUG: Including all related concepts: {list(connected_concepts)}")
        
        if include_connected:
            # Find concepts with similar names to our related concepts
            for idx, name in idx_to_name.items():
                if idx in connected_concepts:
                    continue
                    
                # Check if this concept's name is similar to any of our related concepts
                for related_idx in related_concepts:
                    if related_idx not in idx_to_name:
                        continue
                    
                    related_name = idx_to_name[related_idx]
                    
                    if related_name and name:
                        # Check for common words in names (except very common words)
                        words1 = set(name.lower().split())
                        words2 = set(related_name.lower().split())
                        common_words = words1.intersection(words2) - {'a', 'an', 'the', 'of', 'and', 'or', 'in', 'on', 'with'}
                        
                        if common_words:
                            connected_concepts.add(idx)
                            print(f"DEBUG: Added concept {idx} ({name}) due to name similarity with {related_idx} ({related_name})")
                            break  # Once we find a match, no need to check other related concepts
            
            # Find concepts with similar mastery levels
            if related_concepts:
                # Get average mastery level of related concepts
                related_norms = [memory_norms[idx] for idx in related_concepts if idx < len(memory_norms)]
                if related_norms:
                    avg_mastery = sum(related_norms) / len(related_norms)
                    
                    # Find concepts with similar mastery levels
                    for idx, norm in enumerate(memory_norms):
                        if idx >= len(kc_list) or idx in connected_concepts:
                            continue
                        
                        # If mastery level is within 20% of average, consider it related
                        if abs(norm - avg_mastery) / avg_mastery < 0.2:
                            connected_concepts.add(idx)
                            print(f"DEBUG: Added concept {idx} ({idx_to_name.get(idx, '')}) due to similar mastery level")
            
        print(f"DEBUG: Final connected concepts: {list(connected_concepts)}")
    
    # First pass: Add all nodes with their mastery levels
    for idx, norm in enumerate(memory_norms):
        if idx >= len(kc_list):
            continue
        
        # Skip if we're filtering and this concept isn't related or connected
        if related_concepts is not None:
            if not show_all and idx not in connected_concepts:
                continue
            
        concept_name = idx_to_name.get(idx, f"Concept {idx}")
        
        # Use full concept name without truncation for better visualization
        # Truncate only if extremely long (for display purposes)
        if len(concept_name) > 30:
            display_name = concept_name[:27] + "..."
        else:
            display_name = concept_name
            
        # Use the actual mv_norm value directly
        mv_norm = norm
        
        # Check if this is a directly related concept (for highlighting)
        is_directly_related = related_concepts is not None and idx in related_concepts
        
        # Define node color based on mastery using thresholds for raw norm values
        if mv_norm < 1.0:
            color = "red"  # Weak knowledge
            category = "Thấp"
        elif mv_norm < 2.0:
            color = "orange"  # Medium knowledge
            category = "Trung bình"
        else:
            color = "green"  # Strong knowledge
            category = "Cao"
            
        G.add_node(idx, 
                  name=display_name, 
                  full_name=concept_name,
                  mastery=mv_norm, 
                  color=color,
                  category=category,
                  is_directly_related=is_directly_related)
    
    # Create edges between nodes
    if G.number_of_nodes() > 1:
        print("DEBUG: Creating edges between concepts")
        
        # Create a list of nodes
        nodes = list(G.nodes())
        
        # First, connect all directly related concepts to each other
        if related_concepts:
            related_nodes = [n for n in nodes if n in related_concepts]
            for i, node1 in enumerate(related_nodes):
                for node2 in related_nodes[i+1:]:
                    G.add_edge(node1, node2, weight=1.0, relation_type="direct")
                    G.add_edge(node2, node1, weight=1.0, relation_type="direct")
                    print(f"DEBUG: Added direct edge between related concepts {node1} <-> {node2}")
        
        # Then connect other nodes based on similarity
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Skip if already connected
                if G.has_edge(node1, node2) or G.has_edge(node2, node1):
                    continue
                    
                name1 = G.nodes[node1].get('full_name', '')
                name2 = G.nodes[node2].get('full_name', '')
                
                # Check for name similarity
                if name1 and name2:
                    words1 = set(name1.lower().split())
                    words2 = set(name2.lower().split())
                    common_words = words1.intersection(words2) - {'a', 'an', 'the', 'of', 'and', 'or', 'in', 'on', 'with'}
                    
                    if common_words:
                        G.add_edge(node1, node2, weight=1.0, relation_type="similar")
                        G.add_edge(node2, node1, weight=1.0, relation_type="similar")
                        print(f"DEBUG: Added similarity edge {node1} <-> {node2}")
                        continue
                
                # Check for mastery similarity
                mastery1 = G.nodes[node1].get('mastery', 0)
                mastery2 = G.nodes[node2].get('mastery', 0)
                
                if abs(mastery1 - mastery2) / (max(mastery1, mastery2) + 0.0001) < 0.2:
                    G.add_edge(node1, node2, weight=0.8, relation_type="mastery")
                    G.add_edge(node2, node1, weight=0.8, relation_type="mastery")
                    print(f"DEBUG: Added mastery edge {node1} <-> {node2}")
        
        print(f"DEBUG: Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Add edges based on relationships (if relationships are provided directly)
    if hasattr(create_knowledge_graph, '_relationships') and create_knowledge_graph._relationships:
        for source_idx, target_idx, relation_type in create_knowledge_graph._relationships:
            if source_idx in G.nodes and target_idx in G.nodes:
                G.add_edge(source_idx, target_idx, weight=1.0, relation_type=relation_type)
                print(f"DEBUG: Added {relation_type} edge from {source_idx} to {target_idx}")
    
    return G

def plot_interactive_knowledge_graph(G):
    # Create Plotly figure
    fig = go.Figure()
    
    # Check if graph has nodes
    if G.number_of_nodes() == 0:
        # Return empty figure with message
        fig.add_annotation(
            text="No concept data available to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Knowledge Concept Map",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        # Add a dummy trace to ensure fig.data is not None
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(opacity=0), showlegend=False))
        return fig
    
    # Use a more spread out layout similar to the image
    try:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    except Exception as e:
        print(f"Error creating layout: {e}")
        # Fallback to simpler layout if spring_layout fails
        try:
            pos = nx.random_layout(G, seed=42)
        except Exception as e:
            print(f"Error creating random layout: {e}")
            # If even random layout fails, add a dummy trace and return early
            fig.add_annotation(
                text="Error generating layout for the knowledge graph",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Knowledge Concept Map (Error)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            # Add a dummy trace to ensure fig.data is not None
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(opacity=0), showlegend=False))
            return fig
    
    # Color settings
    edge_color = '#666'  # Darker color for better visibility
    node_outline_color = '#333'
    
    # Add edges
    edge_trace_list = []
    
    # Update the create_arrow function to return better arrow parameters
    def create_arrow(x0, y0, x1, y1, arrow_length=0.05, arrow_angle=30):
        # Direction vector
        dx, dy = x1 - x0, y1 - y0
        # Normalize
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return []  # Avoid division by zero
            
        dx, dy = dx/length, dy/length
        
        # Calculate arrow head points
        cos_angle = math.cos(math.radians(arrow_angle))
        sin_angle = math.sin(math.radians(arrow_angle))
        
        # Adjust arrow position to be slightly before endpoint for better appearance
        x1_adj = x1 - 0.02 * dx
        y1_adj = y1 - 0.02 * dy
        
        # One point of the arrow head
        ax1 = x1_adj - arrow_length * (dx * cos_angle - dy * sin_angle)
        ay1 = y1_adj - arrow_length * (dy * cos_angle + dx * sin_angle)
        
        # Another point of the arrow head
        ax2 = x1_adj - arrow_length * (dx * cos_angle + dy * sin_angle)
        ay2 = y1_adj - arrow_length * (dy * cos_angle - dx * sin_angle)
        
        # Return points for arrow
        return [ax1, ay1, x1_adj, y1_adj, ax2, ay2]

    # Dictionary to map edge types to colors
    edge_color_map = {
        "prerequisite": "#666",  # Standard relationship
        "related": "#1E88E5",    # Blue for similarity-based
        "mastery_relation": "#8E24AA",  # Purple for mastery-based
        "sequence": "#43A047",   # Green for sequence
        "cross_connection": "#FB8C00"  # Orange for cross connections
    }
    
    # Group edges by type for better visualization
    edges_by_type = {}
    for edge in G.edges(data=True):
        edge_type = edge[2].get('relation_type', 'prerequisite')
        if edge_type not in edges_by_type:
            edges_by_type[edge_type] = []
        edges_by_type[edge_type].append((edge[0], edge[1], edge[2]))
    
    # Add edges with arrows, grouped by type
    for edge_type, edges in edges_by_type.items():
        # Get color for this edge type
        color = edge_color_map.get(edge_type, edge_color)
        
        # Add all edges of this type
        for edge_data in edges:
            source, target, attrs = edge_data
            try:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                
                # Create line for edge
                edge_trace = go.Scatter(
                    x=[x0, x1, None], 
                    y=[y0, y1, None],
                    line=dict(width=2.0, color=color),  # Thicker line
                    opacity=0.9,  # Higher opacity
                    hoverinfo='text',
                    text=f"{edge_type}",  # Show relation type on hover
                    mode='lines',
                    name=edge_type,  # Group by type in legend
                    showlegend=(edge_data == edges[0])  # Only show in legend once per type
                )
                edge_trace_list.append(edge_trace)
                
                # Create arrow for edge direction
                arrow_points = create_arrow(x0, y0, x1, y1)
                if arrow_points:
                    arrow_trace = go.Scatter(
                        x=[arrow_points[0], arrow_points[2], arrow_points[4], arrow_points[0]],
                        y=[arrow_points[1], arrow_points[3], arrow_points[5], arrow_points[1]],
                        line=dict(width=1.0, color=color),
                        fill="toself",
                        fillcolor=color,
                        opacity=0.9,
                        hoverinfo='none',
                        mode='lines',
                        showlegend=False
                    )
                    edge_trace_list.append(arrow_trace)
            except Exception as e:
                print(f"Error adding edge {source}->{target}: {e}")
                continue
    
    # Add all edge traces to the figure
    for trace in edge_trace_list:
        fig.add_trace(trace)
    
    # Separate nodes by mastery level and relationship to question
    # We'll now have 6 categories: Low/Medium/High + Related/Not-Related
    directly_related_low_x, directly_related_low_y, directly_related_low_text, directly_related_low_size = [], [], [], []
    directly_related_medium_x, directly_related_medium_y, directly_related_medium_text, directly_related_medium_size = [], [], [], []
    directly_related_high_x, directly_related_high_y, directly_related_high_text, directly_related_high_size = [], [], [], []
    
    low_x, low_y, low_text, low_size = [], [], [], []
    medium_x, medium_y, medium_text, medium_size = [], [], [], []
    high_x, high_y, high_text, high_size = [], [], [], []
    
    for node in G.nodes():
        try:
            x, y = pos[node]
            name = G.nodes[node].get('name', f'Node {node}')
            full_name = G.nodes[node].get('full_name', name)
            mastery = G.nodes[node].get('mastery', 0.0)
            is_directly_related = G.nodes[node].get('is_directly_related', False)
            hover_text = f"{full_name}<br>‖Mv‖: {mastery:.4f}"
            
            # Constrain node size to a reasonable range to avoid huge circles
            # Use a logarithmic scale to handle very large values while keeping small values visible
            if mastery > 0:
                # For positive values, use log scale with a base size
                node_size = 30 + min(15 * (1 + math.log10(1 + mastery)), 40)
            else:
                # For zero or negative values, use minimum size
                node_size = 30
            
            # Sort nodes by mastery level, category, and direct relation to question
            category = G.nodes[node].get('category', 'Thấp')
            
            if is_directly_related:
                # These are directly related to the question - make them stand out
                if category == "Thấp":  # Low mastery
                    directly_related_low_x.append(x)
                    directly_related_low_y.append(y)
                    directly_related_low_text.append(hover_text)
                    directly_related_low_size.append(node_size + 5)  # Make them bigger
                elif category == "Trung bình":  # Medium mastery
                    directly_related_medium_x.append(x)
                    directly_related_medium_y.append(y)
                    directly_related_medium_text.append(hover_text)
                    directly_related_medium_size.append(node_size + 5)  # Make them bigger
                else:  # High mastery
                    directly_related_high_x.append(x)
                    directly_related_high_y.append(y)
                    directly_related_high_text.append(hover_text)
                    directly_related_high_size.append(node_size + 5)  # Make them bigger
            else:
                # Regular nodes - not directly related to the question
                if category == "Thấp":  # Low mastery
                    low_x.append(x)
                    low_y.append(y)
                    low_text.append(hover_text)
                    low_size.append(node_size)
                elif category == "Trung bình":  # Medium mastery
                    medium_x.append(x)
                    medium_y.append(y)
                    medium_text.append(hover_text)
                    medium_size.append(node_size)
                else:  # High mastery
                    high_x.append(x)
                    high_y.append(y)
                    high_text.append(hover_text)
                    high_size.append(node_size)
        except Exception as e:
            print(f"Error processing node {node}: {e}")
            continue
    
    # Add regular low mastery nodes (red)
    if low_x:
        fig.add_trace(
            go.Scatter(
                x=low_x, y=low_y,
                mode='markers+text',  # Default to showing text
                hoverinfo='text',
                text=low_text,
                textposition="bottom center",
                textfont=dict(size=8, color='black'),
                marker=dict(
                    color='rgba(255, 50, 50, 0.8)',  # Brighter red with transparency
                    size=low_size,
                    line=dict(width=1.5, color=node_outline_color)
                ),
                name="Thấp"
            )
        )
    
    # Add regular medium mastery nodes (orange)
    if medium_x:
        fig.add_trace(
            go.Scatter(
                x=medium_x, y=medium_y,
                mode='markers+text',  # Default to showing text
                hoverinfo='text',
                text=medium_text,
                textposition="bottom center",
                textfont=dict(size=8, color='black'),
                marker=dict(
                    color='rgba(255, 150, 50, 0.8)',  # Brighter orange with transparency
                    size=medium_size,
                    line=dict(width=1.5, color=node_outline_color)
                ),
                name="Trung bình"
            )
        )
    
    # Add regular high mastery nodes (green)
    if high_x:
        fig.add_trace(
            go.Scatter(
                x=high_x, y=high_y,
                mode='markers+text',  # Default to showing text
                hoverinfo='text',
                text=high_text,
                textposition="bottom center",
                textfont=dict(size=8, color='black'),
                marker=dict(
                    color='rgba(50, 200, 50, 0.8)',  # Brighter green with transparency
                    size=high_size,
                    line=dict(width=1.5, color=node_outline_color)
                ),
                name="Cao"
            )
        )
    
    # Add directly related low mastery nodes (red with thicker border)
    if directly_related_low_x:
        fig.add_trace(
            go.Scatter(
                x=directly_related_low_x, y=directly_related_low_y,
                mode='markers+text',
                hoverinfo='text',
                text=directly_related_low_text,
                textposition="bottom center",
                textfont=dict(size=10, color='black'),
                marker=dict(
                    color='rgba(255, 50, 50, 1.0)',  # Full opacity red
                    size=directly_related_low_size,
                    line=dict(width=3, color='black'),  # Thicker border
                    symbol='circle'
                ),
                name="Liên quan - Thấp"
            )
        )
    
    # Add directly related medium mastery nodes (orange with thicker border)
    if directly_related_medium_x:
        fig.add_trace(
            go.Scatter(
                x=directly_related_medium_x, y=directly_related_medium_y,
                mode='markers+text',
                hoverinfo='text',
                text=directly_related_medium_text,
                textposition="bottom center",
                textfont=dict(size=10, color='black'),
                marker=dict(
                    color='rgba(255, 150, 50, 1.0)',  # Full opacity orange
                    size=directly_related_medium_size,
                    line=dict(width=3, color='black'),  # Thicker border
                    symbol='circle'
                ),
                name="Liên quan - Trung bình"
            )
        )
    
    # Add directly related high mastery nodes (green with thicker border)
    if directly_related_high_x:
        fig.add_trace(
            go.Scatter(
                x=directly_related_high_x, y=directly_related_high_y,
                mode='markers+text',
                hoverinfo='text',
                text=directly_related_high_text,
                textposition="bottom center",
                textfont=dict(size=10, color='black'),
                marker=dict(
                    color='rgba(50, 200, 50, 1.0)',  # Full opacity green
                    size=directly_related_high_size,
                    line=dict(width=3, color='black'),  # Thicker border
                    symbol='circle'
                ),
                name="Liên quan - Cao"
            )
        )
    
    # Update layout for a cleaner, more colorful graph
    fig.update_layout(
        title="Knowledge Concept Map",
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1.0,
            orientation="v",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        width=800,
        plot_bgcolor='rgba(255, 255, 255, 1)',
        template="plotly_white"
    )
    
    return fig

# Add a new function to create a legend for the knowledge graph
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

# Prediction Function (similar to use_trainer.py approach)
def predict_answer(model, student_id, question_idx, option_idx, concept_ids, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    # Filter out placeholder concept_ids (negative values)
    valid_concept_ids = [cid for cid in concept_ids if cid >= 0]
    
    # Print debug info
    print(f"DEBUG: Making prediction for student {student_id}, question {question_idx}, option {option_idx}")
    print(f"DEBUG: Using {len(valid_concept_ids)} valid concepts (filtered from {len(concept_ids)} total)")
    print(f"DEBUG: Valid concept IDs: {valid_concept_ids}")
    
    # If no valid concepts, return a default prediction
    if not valid_concept_ids:
        print("WARNING: No valid concepts for prediction, using default probability of 0.5")
        return 0.5
    
    # Convert to tensors
    q_idx = torch.tensor(question_idx)
    o_idx = torch.tensor(option_idx)
    u_idx = torch.tensor(option_idx)  # Using the same as option_idx for unchosen
    score = torch.tensor(0.0)  # Dummy score
    timestamp_tensor = torch.tensor(float(timestamp))
    
    # Make prediction
    try:
        with torch.no_grad():
            pred = model.forward_single_step(
                student_id=student_id,
                q_idx=q_idx,
                o_idx=o_idx,
                u_idx=u_idx,
                score=score,
                timestamp=timestamp_tensor,
                concept_ids=valid_concept_ids
            )
        return pred.item()
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return 0.5  # Return default probability on error

# Function to prepare model for a student (adapted from use_trainer.py)
def prepare_for_student(model, student_idx, question_df, option_df, question_concept_df, interaction_df, mappings):
    # Reset student memory
    model.reset_memory(student_idx)
    print(f"DEBUG: Reset memory for student {student_idx}")
    
    # Store model and student_id in session state for later use
    st.session_state.model = model
    st.session_state.current_student_id = student_idx
    
    # Get student's past interactions
    student_df = interaction_df[interaction_df["student_id"] == student_idx].sort_values("start_time")
    print(f"DEBUG: Found {len(student_df)} interactions for student {student_idx}")
    
    # Check if initial memory is zero
    initial_memory = model.student_memory.get(student_idx, None)
    if initial_memory is not None:
        print(f"DEBUG: Initial memory shape: {initial_memory.shape}")
        print(f"DEBUG: Initial memory type: {type(initial_memory)}")
        print(f"DEBUG: Initial memory sum: {initial_memory.sum().item()}")
        print(f"DEBUG: Initial memory[0] sum: {initial_memory[0].sum().item() if len(initial_memory) > 0 else 'N/A'}")
    else:
        print("DEBUG: Initial memory is None")
    
    # Process each interaction
    num_processed = 0
    for _, row in student_df.iterrows():
        q_id = row["question_id"]
        if q_id not in mappings["question2idx"]:
            continue
            
        q_idx = mappings["question2idx"][q_id]
        
        # Get answer options for this question
        q_options = option_df[option_df["question_id"] == q_id]
        if q_options.empty:
            continue
            
        # Get selected option
        ans_id = row["answer_choice_id"]
        if pd.isna(ans_id):
            continue
            
        o_idx = 0  # Default
        try:
            o_list = q_options["id"].tolist()
            o_idx = o_list.index(ans_id)
        except:
            pass
            
        u_idx = max(0, o_idx - 1)  # Simple unchosen
        score = float(row["answer_state"])
        timestamp = row["start_time"].timestamp()
        
        # Get concepts for this question
        q_concepts = question_concept_df[question_concept_df["question_id"] == q_id]
        c_ids = []
        for _, c_row in q_concepts.iterrows():
            kc_id = c_row["knowledgecomponent_id"]
            if kc_id in mappings["kc2idx"]:
                c_ids.append(mappings["kc2idx"][kc_id])
        
        if not c_ids:
            continue
            
        # Debug the concepts
        if num_processed < 3:  # Only show for first few interactions
            print(f"DEBUG: Interaction {num_processed+1} - Question {q_id}, Concepts: {c_ids}")
        
        # Process interaction
        try:
            result = model.forward_single_step(
                student_id=student_idx,
                q_idx=torch.tensor(q_idx),
                o_idx=torch.tensor(o_idx),
                u_idx=torch.tensor(u_idx),
                score=torch.tensor(score),
                timestamp=torch.tensor(timestamp),
                concept_ids=c_ids
            )
            
            # Check memory after each interaction
            if num_processed < 3:  # Only check first few interactions
                curr_memory = model.student_memory.get(student_idx, None)
                if curr_memory is not None:
                    print(f"DEBUG: After interaction {num_processed+1}, memory sum: {curr_memory.sum().item()}")
                    print(f"DEBUG: Memory for concepts {c_ids}: {[curr_memory[c_id].sum().item() for c_id in c_ids]}")
                    print(f"DEBUG: Prediction result: {result.item() if result is not None else 'None'}")
                else:
                    print(f"DEBUG: After interaction {num_processed+1}, memory is None")
            
            num_processed += 1
        except Exception as e:
            print(f"Error processing interaction: {e}")
    
    # Debug - check if memory was updated
    if student_idx in model.student_memory:
        memory = model.student_memory[student_idx]
        memory_norms = [tensor.norm().item() for tensor in memory]
        print(f"DEBUG: Processed {num_processed} interactions. Memory updated with {len(memory)} concepts.")
        print(f"DEBUG: Memory norms sample: {memory_norms[:5]}")
        print(f"DEBUG: Memory norms sum: {sum(memory_norms)}")
        print(f"DEBUG: Memory norms max: {max(memory_norms) if memory_norms else 'N/A'}")
        print(f"DEBUG: Memory norms min: {min(memory_norms) if memory_norms else 'N/A'}")
        print(f"DEBUG: Memory norms non-zero count: {sum(1 for n in memory_norms if n > 0.0001)}")
        
        # Get a snapshot directly from the model
        snapshot = model.get_snapshot(student_idx)
        if snapshot is not None:
            snapshot_norms = [tensor.norm().item() for tensor in snapshot]
            print(f"DEBUG: Snapshot norms sample: {snapshot_norms[:5]}")
            print(f"DEBUG: Snapshot sum: {snapshot.sum().item()}")
        else:
            print("DEBUG: No snapshot available")
    else:
        print(f"DEBUG: No memory found for student {student_idx} after processing")
    
    return model

# GPT Integration for Learning Path
def generate_learning_path(weak_concepts, concept_df, kc_list, time_period="7 days", student_id=None, force_refresh=False):
    """
    Generate a personalized learning path for a student based on weak concepts
    
    Parameters:
    - weak_concepts: List of concept indices that are weak
    - concept_df: DataFrame of concepts
    - kc_list: List of concept IDs
    - time_period: Time period for the learning path
    - student_id: ID of the student (for cache)
    - force_refresh: If True, ignore cache and generate a new learning path
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
    
    # Create the prompt with stronger focus on active recall strategies
    content = f"""
    As an AI tutor specializing in evidence-based learning techniques, create a personalized learning path focused on ACTIVE RECALL strategies for a student who needs to improve on these concepts:
    {', '.join(weak_concept_names)}
    
    The learning path should be designed for {time_period} and PRIORITIZE active recall over passive review. 
    
    Include specifically:
    
    1. A daily/weekly schedule with active recall activities such as:
       - Self-quizzing exercises with specific example questions
       - Concept mapping activities
       - Teaching the material to someone else (Feynman Technique)
       - Problem-solving challenges that apply the concepts
    
    2. Concrete practice exercises using these active recall methods:
       - Retrieval practice (writing answers from memory)
       - Flashcard systems with spaced repetition
       - Cloze deletion exercises
       - Application problems that force recall and synthesis
    
    3. Spaced repetition strategy with increasing intervals:
       - Initial recall: same day
       - Second recall: 1-2 days later
       - Third recall: 3-4 days later
       - Final recall: 7 days later
    
    4. Progress measurement methods:
       - Self-assessment rubrics
       - Practice tests with specific metrics
       - Knowledge mastery tracking system
    
    For each concept, provide at least 2 SPECIFIC examples of active recall questions or exercises.
    
    Format the response as a clear, structured plan that emphasizes DOING rather than passive reading or highlighting.
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
            max_tokens=1500   # Allow for longer, more detailed responses
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

# Update the main function to fix concept name display in the detailed mastery levels
def display_mastery_levels(student_memory, kc_list, concept_df, related_concepts=None):
    """
    Display mastery levels with correct concept names
    If related_concepts is provided, only show concepts related to the current question
    """
    # Handle case where student_memory is None or empty
    if student_memory is None or len(student_memory) == 0:
        print("DEBUG: student_memory is None or empty in display_mastery_levels")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Try to get a snapshot directly from the model in session state
    memory_to_use = student_memory
    
    # Try to use the snapshot if available
    if 'model' in st.session_state and 'current_student_id' in st.session_state:
        try:
            student_id = st.session_state.current_student_id
            snapshot = st.session_state.model.get_snapshot(student_id)
            if snapshot is not None:
                print(f"DEBUG: Using snapshot for student {student_id} instead of student_memory")
                memory_to_use = snapshot
            else:
                print("DEBUG: No snapshot available, using student_memory")
        except Exception as e:
            print(f"DEBUG: Error getting snapshot: {e}")
    
    # Calculate norms for all concepts - use the actual memory vector norm
    try:
        memory_norms = [tensor.norm().item() for tensor in memory_to_use]
    except Exception as e:
        print(f"DEBUG: Error calculating memory norms in display_mastery_levels: {e}")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Ensure related_concepts is valid
    if related_concepts is not None:
        try:
            # Convert to set for faster lookup
            related_set = set(related_concepts)
        except Exception as e:
            print(f"DEBUG: Error converting related_concepts to set: {e}")
            related_set = None  # Fall back to showing all concepts
    else:
        related_set = None  # Show all concepts
    
    # Debug: Check the values of memory_norms
    print(f"DEBUG: Memory norms: {memory_norms[:5]}")
    
    # Create a dataframe for display
    mastery_data = []
    for idx, norm in enumerate(memory_norms):
        if idx >= len(kc_list):
            continue
        
        # If related_concepts is provided, only show related concepts
        if related_set is not None and idx not in related_set and idx >= 0:
            continue
            
        # Get the real concept ID from kc_list
        try:
            concept_id_real = kc_list[idx]
            concept_name = get_concept_name(idx, concept_df, kc_list)
        except Exception as e:
            print(f"DEBUG: Error getting concept name for index {idx}: {e}")
            concept_id_real = f"Unknown-{idx}"
            concept_name = f"Concept {idx}"
        
        # Use the actual mv_norm value directly (same as in new_train_dcrkt.py)
        mv_norm = norm
        
        # Debug: Print each mv_norm value
        print(f"DEBUG: Concept {idx}, ID {concept_id_real}, Name {concept_name}, mv_norm = {mv_norm}")
        
        # Determine status based on the memory vector norm
        # Using thresholds that make sense for the raw norm values
        status = "Strong" if mv_norm > 2.0 else "Medium" if mv_norm > 1.0 else "Weak"
        
        mastery_data.append({
            "Concept ID": concept_id_real,
            "Concept": concept_name,
            "‖Mv‖": mv_norm,
            "Status": status
        })
    
    # Create DataFrame safely
    try:
        df = pd.DataFrame(mastery_data).sort_values("‖Mv‖", ascending=False)
    except Exception as e:
        print(f"DEBUG: Error creating mastery DataFrame: {e}")
        # Try with a simpler approach
        df = pd.DataFrame(mastery_data) if mastery_data else pd.DataFrame()
    
    # If filtering by related concepts but none were found, return empty dataframe with a message
    if related_set is not None and df.empty:
        st.info("No mastery data available for concepts related to this question.")
        
    return df

def get_extended_related_concepts(selected_question, question_concept_df, concept_df, relation_df, kc2idx, kc_list, depth=1):
    """
    Get concepts related to a question and their related concepts (extended network)
    
    Parameters:
    - selected_question: The question ID
    - question_concept_df: DataFrame mapping questions to concepts
    - concept_df: DataFrame of all concepts
    - relation_df: DataFrame of concept relationships
    - kc2idx: Mapping from concept IDs to indices
    - kc_list: List of concept IDs
    - depth: How many levels of relationships to explore (1 = direct + 1 level, 2 = direct + 2 levels, etc.)
    
    Returns:
    - extended_concept_ids: List of concept indices (direct + related)
    - extended_concept_names: List of concept names
    - direct_concept_ids: List of only the directly related concept indices
    - relationships: List of tuples (source_idx, target_idx, relation_type) for drawing edges
    """
    # First get concepts directly related to the question
    direct_concept_ids, direct_concept_names = get_related_concepts(selected_question, question_concept_df, concept_df, kc2idx, kc_list)
    
    # Initialize extended sets with direct concepts
    extended_concept_ids = set(direct_concept_ids)
    extended_concept_names = list(direct_concept_names)
    relationships = []
    
    # Get the actual concept IDs (not indices) for the direct concepts
    direct_kc_ids = []
    for idx in direct_concept_ids:
        if idx >= 0 and idx < len(kc_list):  # Valid index
            direct_kc_ids.append(kc_list[idx])
    
    print(f"DEBUG: Direct concepts for question {selected_question}: {direct_kc_ids}")
    
    # Explore relationships for the specified depth
    current_level_kc_ids = set(direct_kc_ids)
    
    for level in range(depth):
        next_level_kc_ids = set()
        
        # Check relation_df for relationships
        if relation_df is not None and not relation_df.empty:
            # Get column names from relation_df
            columns = relation_df.columns.tolist()
            print(f"DEBUG: Relation_df columns: {columns}")
            
            # Try to identify the correct column names
            source_col = None
            target_col = None
            relation_col = None
            
            # Common column name patterns
            for col in columns:
                col_lower = col.lower()
                if col_lower in ['source', 'from', 'kc1', 'concept1', 'prerequisite']:
                    source_col = col
                elif col_lower in ['target', 'to', 'kc2', 'concept2', 'dependent']:
                    target_col = col
                elif col_lower in ['relation_type', 'type', 'relationship', 'relation']:
                    relation_col = col
            
            # If we couldn't find standard names, use the first two columns
            if source_col is None and len(columns) >= 2:
                source_col = columns[0]
                target_col = columns[1]
                if len(columns) >= 3:
                    relation_col = columns[2]
            
            print(f"DEBUG: Using columns - source: {source_col}, target: {target_col}, relation: {relation_col}")
            
            if source_col and target_col:
                # Find concepts related to current level concepts
                for kc_id in current_level_kc_ids:
                    # Find where this concept is the source
                    source_relations = relation_df[relation_df[source_col] == kc_id]
                    for _, row in source_relations.iterrows():
                        target_kc_id = row[target_col]
                        relation_type = row[relation_col] if relation_col else 'related'
                        
                        # Convert to indices
                        source_idx = kc2idx.get(kc_id) or (kc_list.index(kc_id) if kc_id in kc_list else None)
                        target_idx = kc2idx.get(target_kc_id) or (kc_list.index(target_kc_id) if target_kc_id in kc_list else None)
                        
                        if source_idx is not None and target_idx is not None:
                            if target_idx not in extended_concept_ids:
                                extended_concept_ids.add(target_idx)
                                concept_name = get_concept_name(target_idx, concept_df, kc_list)
                                extended_concept_names.append(concept_name)
                                next_level_kc_ids.add(target_kc_id)
                                print(f"DEBUG: Added related concept {concept_name} (level {level+1})")
                            
                            relationships.append((source_idx, target_idx, relation_type))
                    
                    # Find where this concept is the target
                    target_relations = relation_df[relation_df[target_col] == kc_id]
                    for _, row in target_relations.iterrows():
                        source_kc_id = row[source_col]
                        relation_type = row[relation_col] if relation_col else 'related'
                        
                        # Convert to indices
                        source_idx = kc2idx.get(source_kc_id) or (kc_list.index(source_kc_id) if source_kc_id in kc_list else None)
                        target_idx = kc2idx.get(kc_id) or (kc_list.index(kc_id) if kc_id in kc_list else None)
                        
                        if source_idx is not None and target_idx is not None:
                            if source_idx not in extended_concept_ids:
                                extended_concept_ids.add(source_idx)
                                concept_name = get_concept_name(source_idx, concept_df, kc_list)
                                extended_concept_names.append(concept_name)
                                next_level_kc_ids.add(source_kc_id)
                                print(f"DEBUG: Added prerequisite concept {concept_name} (level {level+1})")
                            
                            relationships.append((source_idx, target_idx, relation_type))
        
        # Update for next iteration
        current_level_kc_ids = next_level_kc_ids
        if not current_level_kc_ids:  # No more concepts to explore
            break
    
    print(f"DEBUG: Extended network has {len(extended_concept_ids)} concepts and {len(relationships)} relationships")
    
    return list(extended_concept_ids), extended_concept_names, direct_concept_ids, relationships

# Update the concept display in related concepts section
def get_related_concepts(selected_question, question_concept_df, concept_df, kc2idx, kc_list):
    """Get concepts related to a question with correct names"""
    concepts = question_concept_df[question_concept_df["question_id"] == selected_question]
    concept_ids = []
    concept_names = []
    
    # Debug info
    print(f"DEBUG: Finding concepts for question {selected_question}")
    print(f"DEBUG: Found {len(concepts)} concept relations in question_concept_df")
    
    for _, concept_row in concepts.iterrows():
        kc_id = concept_row["knowledgecomponent_id"]
        # Debug
        print(f"DEBUG: Processing KC ID {kc_id}, type: {type(kc_id)}")
        
        # Try different ways to find the concept index
        concept_idx = None
        
        # Check if in kc2idx directly
        if kc_id in kc2idx:
            concept_idx = kc2idx[kc_id]
            print(f"DEBUG: Found KC {kc_id} directly in kc2idx")
        # Check string version
        elif str(kc_id) in kc2idx:
            concept_idx = kc2idx[str(kc_id)]
            print(f"DEBUG: Found KC {kc_id} as string in kc2idx")
        # Try to find in kc_list
        else:
            try:
                list_idx = kc_list.index(kc_id)
                concept_idx = list_idx
                print(f"DEBUG: Found KC {kc_id} at index {list_idx} in kc_list")
            except ValueError:
                # Try as string
                try:
                    list_idx = kc_list.index(str(kc_id))
                    concept_idx = list_idx
                    print(f"DEBUG: Found KC {kc_id} as string at index {list_idx} in kc_list")
                except ValueError:
                    print(f"DEBUG: KC {kc_id} not found in kc_list")
        
        if concept_idx is not None:
            concept_ids.append(concept_idx)
            # Use the concept index to get the correct name
            concept_name = get_concept_name(concept_idx, concept_df, kc_list)
            concept_names.append(concept_name)
            print(f"DEBUG: Added concept {concept_name} with index {concept_idx}")
        else:
            # If we can't find the index but the concept exists in the dataframe,
            # still add it with its name from the concept dataframe
            concept_row = concept_df[concept_df["id"] == kc_id]
            if not concept_row.empty:
                concept_name = concept_row["name"].values[0]
                # Use a placeholder index
                placeholder_idx = -1
                concept_ids.append(placeholder_idx)
                concept_names.append(f"{concept_name} (unmapped)")
                print(f"DEBUG: Added unmapped concept {concept_name}")
    
    print(f"DEBUG: Returning {len(concept_ids)} concept IDs and {len(concept_names)} concept names")
    return concept_ids, concept_names

# Main App
def main():
    st.title("Knowledge Tracing and Learning Path System")
    
    # Load data and model
    try:
        question_df, option_df, concept_df, relation_df, question_concept_df, interaction_df = load_data()
        
        # Model loading section with file uploader and button
        st.sidebar.header("Model Loading")
        
        # Display current model status
        if 'model' in st.session_state and st.session_state.model is not None:
            st.sidebar.success("✅ Model is loaded and ready")
            if hasattr(st.session_state, 'loaded_model_path'):
                st.sidebar.caption(f"Current model: {st.session_state.loaded_model_path}")
        else:
            st.sidebar.warning("⚠️ No model loaded")
        
        # Option to use a default model or upload a custom one
        model_option = st.sidebar.radio(
            "Choose model source:",
            ["Use default model", "Select from available models", "Upload custom model"]
        )
        
        if model_option == "Use default model":
            default_model_path = "streamlit/backup/dcrkt_model_fold_0.pt"
            if os.path.exists(default_model_path):
                if st.sidebar.button("Load Default Model"):
                    model, mappings = load_model_from_file(default_model_path)
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.mappings = mappings
                        st.session_state.loaded_model_path = default_model_path
                        st.sidebar.success(f"Default model loaded successfully!")
            else:
                st.sidebar.error(f"Default model not found at {default_model_path}")
                
        elif model_option == "Select from available models":
            # Find available model files
            model_files = []
            for file in os.listdir():
                if file.endswith(".pt"):
                    model_files.append(file)
            
            if not model_files:
                st.sidebar.warning("No model files (.pt) found in the current directory.")
            else:
                selected_model = st.sidebar.selectbox("Select model file:", model_files)
                if st.sidebar.button("Load Selected Model"):
                    model, mappings = load_model_from_file(selected_model)
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.mappings = mappings
                        st.session_state.loaded_model_path = selected_model
                        st.sidebar.success(f"Model loaded successfully from {selected_model}!")
        
        elif model_option == "Upload custom model":
            uploaded_file = st.sidebar.file_uploader("Upload model file (.pt)", type=["pt"])
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open("temp_model.pt", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.sidebar.button("Load Uploaded Model"):
                    model, mappings = load_model_from_file("temp_model.pt")
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.mappings = mappings
                        st.session_state.loaded_model_path = f"Uploaded: {uploaded_file.name}"
                        st.sidebar.success("Uploaded model loaded successfully!")
        
        # Check if model and mappings are in session state
        if 'model' not in st.session_state or 'mappings' not in st.session_state:
            st.warning("Please load a model before proceeding.")
            return
        
        model = st.session_state.model
        mappings = st.session_state.mappings
        
        if model is None or mappings is None:
            st.error("Failed to load model or mappings. Please check the files and try again.")
            return
    except Exception as e:
        st.error(f"Error loading data or initializing components: {e}")
        return
            
    # Extract mappings
    user2idx = mappings.get("user2idx", {})
    question2idx = mappings.get("question2idx", {})
    kc2idx = mappings.get("kc2idx", {})
        
    # In new_train_dcrkt.py, kc_list is defined as:
    # kc_list = sorted(concept_df["id"].unique())
    # So we need to use the actual concept IDs from the CSV, not just the keys of kc2idx
    kc_list = sorted(concept_df["id"].unique())
        
    # Create reverse mappings
    idx2user = {v: k for k, v in user2idx.items()}
    idx2question = {v: k for k, v in question2idx.items()}
        
    # Get unique students and questions
    unique_students = sorted(interaction_df["student_id"].unique())
    unique_questions = sorted(question_df["id"].unique())
        
    # Create dynamic student mappings if needed
    # This ensures that all students in the dataset can be used even if they're not in the original mappings
    student_mapping = {}
    for i, student_id in enumerate(unique_students):
        # Use existing mapping if available, otherwise create a new one
        if str(student_id) in user2idx:
            student_mapping[student_id] = user2idx[str(student_id)]
        elif student_id in user2idx:
            student_mapping[student_id] = user2idx[student_id]
        else:
            # Use position in unique_students as the index if not in mappings
            student_mapping[student_id] = i
        
    # Initialize with default student (39) if app is loading for the first time
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = True
        
        # Automatically load default model on first run
        default_model_path = "streamlit/backup/dcrkt_model_fold_0.pt"
        if os.path.exists(default_model_path):
            with st.spinner("Loading default model..."):
                try:
                    model, mappings = load_model_from_file(default_model_path)
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.mappings = mappings
                        st.session_state.loaded_model_path = default_model_path
                        st.success(f"✅ Default model loaded successfully! System is ready.")
                except Exception as e:
                    st.error(f"Error loading default model: {e}")
                    st.warning("Please load a model using the options in the sidebar before proceeding.")
        else:
            st.warning(f"Default model not found at {default_model_path}")
            st.info("Please load a model using the options in the sidebar before proceeding.")
        
        # Set some defaults for when a model is loaded
        st.session_state.default_student = 39
        st.session_state.default_question = 9
            
    # Tabs for different sections
    tab1, tab2 = st.tabs(["Knowledge Testing", "Learning Path Recommendation"])
        
    with tab1:
        st.header("Test Student Knowledge")
        
        # Check if model is loaded
        if 'model' not in st.session_state or st.session_state.model is None:
            st.warning("Please load a model first using the options in the sidebar.")
            return
            
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Get default student
            default_student = st.session_state.get('default_student', 39)
            
            # Student selection - default to student 39 as in use_trainer.py
            default_student_idx = unique_students.index(default_student) if default_student in unique_students else 0
            selected_student = st.selectbox("Select Student", unique_students, index=default_student_idx, key="student_select")
            
            # Get student index using our dynamic mapping
            student_idx = student_mapping.get(selected_student)
            if student_idx is None:
                st.error(f"Student {selected_student} could not be mapped to an index.")
                st.error("Available mappings: " + ", ".join([str(k) for k in list(user2idx.keys())[:10]]) + "...")
                return
                
            # Get default question
            default_question = st.session_state.get('default_question', 9)
                
            # Question selection - default to question 9 as in use_trainer.py
            default_question_idx = unique_questions.index(default_question) if default_question in unique_questions else 0
            selected_question = st.selectbox("Select Question", unique_questions, index=default_question_idx, key="question_select")
            
            # Get question index - handle both string and int keys
            question_idx = None
            if selected_question in question2idx:
                question_idx = question2idx[selected_question]
            elif str(selected_question) in question2idx:
                question_idx = question2idx[str(selected_question)]
            
            if question_idx is None:
                st.error(f"Question {selected_question} not found in model mappings.")
                return
                
            # Get question details
            q_text = question_df[question_df["id"] == selected_question]["question_text"].values[0]
            
            # Get options for question
            options = option_df[option_df["question_id"] == selected_question]
            
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
            
            # Get concepts related to the question
            concept_ids, concept_names = get_related_concepts(selected_question, question_concept_df, concept_df, kc2idx, kc_list)
            
            with st.container():
                # Create question box similar to the image
                st.markdown(
                    f"<div class='question-box'>"
                    f"<div class='question-title'>Target Question (Q{selected_question})</div>"
                    f"<p><strong>Q.</strong> {q_text}</p>",
                    unsafe_allow_html=True)
                
                # Display options
                option_texts = options["choice_text"].tolist()
                option_ids = options["id"].tolist()
                
                if option_texts:
                    # Format the options similar to the image
                    option_html = "<p><strong>Options:</strong></p><ul style='list-style-type: none; padding-left: 15px;'>"
                    for i, opt_text in enumerate(option_texts):
                        letter = chr(97 + i)  # a, b, c, d...
                        option_html += f"<li><strong>{letter}.</strong> {opt_text}</li>"
                    option_html += "</ul>"
                    st.markdown(option_html, unsafe_allow_html=True)
                    
                    # Now show concepts section
                    st.markdown("<div class='question-title'>Concepts</div>", unsafe_allow_html=True)
                    
                    # Display concepts as tags in a style similar to the image
                    concept_html = "<div style='margin: 10px 0; display: flex; flex-wrap: wrap;'>"
                    if concept_names:
                        # First row - top concepts
                        first_row_html = ""
                        for i, concept in enumerate(concept_names[:3]):  # First row with up to 3 concepts
                            concept_id = concept_ids[i] if i < len(concept_ids) else "?"
                            first_row_html += f"<span class='concept-tag'>C<sub>{concept_id}</sub> {concept}</span> "
                        
                        # Second row - additional concepts
                        second_row_html = ""
                        for i, concept in enumerate(concept_names[3:6]):  # Second row with up to 3 more concepts
                            idx = i + 3
                            concept_id = concept_ids[idx] if idx < len(concept_ids) else "?"
                            second_row_html += f"<span class='concept-tag'>C<sub>{concept_id}</sub> {concept}</span> "
                        
                        # Add a label for Q11 related if any concepts exist
                        if concept_names:
                            q_related_html = f"<div style='text-align: right; margin-top: 5px;'><em>(Q{selected_question} Related)</em></div>"
                        else:
                            q_related_html = ""
                        
                        # Put it all together with proper layout - fix unescaped HTML
                        concept_html += (
                            "<div style='display: flex; flex-direction: column; width: 100%;'>"
                            f"<div style='display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 5px;'>{first_row_html}</div>"
                            f"<div style='display: flex; flex-wrap: wrap; gap: 5px;'>{second_row_html}</div>"
                            f"{q_related_html}"
                            "</div>"
                        )
                    else:
                        concept_html += "<p>No concepts associated with this question.</p>"
                    concept_html += "</div>"
                    st.markdown(concept_html, unsafe_allow_html=True)
                    
                    # Close the box
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Create radio button for options
                    selected_option_text = st.radio("Select Answer", option_texts, key="option_select")
                    selected_option_id = option_ids[option_texts.index(selected_option_text)]
                    
                    # Get option index
                    option_idx = 0  # Default
                    all_options = option_df[option_df["question_id"] == selected_question]
                    if not all_options.empty:
                        option_list = all_options["id"].tolist()
                        try:
                            option_idx = option_list.index(selected_option_id)
                        except:
                            st.warning("Option not found in model mappings, using default.")
                else:
                    st.warning("No options found for this question.")
                    option_idx = 0
            
            # Display student info
            st.subheader("Student Information")
            st.write(f"Student ID: {selected_student}")
            st.write(f"Internal Index: {student_idx}")
            
            # Automatically process student history when student or question changes
            with st.spinner("Processing student history and making prediction..."):
                # Use the prepare_for_student function adapted from use_trainer.py
                model = prepare_for_student(
                    model, student_idx, 
                    question_df, option_df, question_concept_df, interaction_df,
                    mappings
                )
                
                # Store memory in session state
                if student_idx in model.student_memory:
                    st.session_state.student_memory = model.student_memory[student_idx]
                    st.success(f"Successfully processed student {selected_student}'s history!")
                else:
                    st.error(f"Failed to create memory for student {selected_student}. Please check the model configuration.")
                    return
                
                # Calculate predictions for all options
                predictions = []
                if option_texts:
                    for i, _ in enumerate(option_texts):
                        pred = predict_answer(model, student_idx, question_idx, i, concept_ids)
                        predictions.append(pred)
                
                # Display prediction results in a similar style to the question box
                st.markdown(
                    "<div class='question-box'>"
                    "<div class='question-title'>Prediction Results</div>",
                    unsafe_allow_html=True)
                
                if option_texts:
                    # Show predictions for each option
                    prediction_html = "<ul style='list-style-type: none; padding-left: 15px;'>"
                    for i, opt_text in enumerate(option_texts):
                        letter = chr(97 + i)  # a, b, c, d...
                        prob = predictions[i]
                        
                        # Determine color based on probability
                        if prob > 0.7:
                            color = "green"
                        elif prob > 0.4:
                            color = "orange"
                        else:
                            color = "red"
                            
                        prediction_html += f"<li><strong>{letter}.</strong> {opt_text} - <span style='color: {color}; font-weight: bold;'>Probability: {prob:.2f}</span></li>"
                    prediction_html += "</ul>"
                    st.markdown(prediction_html, unsafe_allow_html=True)
                    
                    # Highlight the most likely answer
                    if predictions:
                        most_likely_idx = predictions.index(max(predictions))
                        most_likely_letter = chr(97 + most_likely_idx)
                        most_likely_text = option_texts[most_likely_idx]
                        st.markdown(
                            f"<p style='font-weight: bold;'>Most Likely Answer: "
                            f"<span style='color: blue;'>{most_likely_letter}. {most_likely_text}</span> "
                            f"(Probability: {max(predictions):.2f})</p>", 
                            unsafe_allow_html=True)
                
                # Close the prediction box
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if 'student_memory' not in st.session_state:
                st.session_state.student_memory = None
                
            # Display knowledge graph when Submit is clicked
            if st.session_state.get('student_memory') is not None:
                # Get student memory
                st.subheader("Knowledge Concept Map")
                
                # Add debugging information
                memory_size = len(st.session_state.student_memory)
                st.write(f"Debug: Student memory contains {memory_size} concepts")
                
                # Check if we have valid data
                if memory_size == 0:
                    st.warning("No concept data found in student memory.")
                    return
                
                # Create knowledge graph
                try:
                    # Try to get a snapshot directly
                    memory_to_use = st.session_state.student_memory
                    if 'model' in st.session_state and 'current_student_id' in st.session_state:
                        try:
                            student_id = st.session_state.current_student_id
                            snapshot = st.session_state.model.get_snapshot(student_id)
                            if snapshot is not None:
                                print(f"DEBUG: Using snapshot for knowledge graph")
                                memory_to_use = snapshot
                        except Exception as e:
                            print(f"DEBUG: Error getting snapshot for graph: {e}")
                    
                    # Get concepts related to current question
                    try:
                        extended_concepts, extended_names, direct_concepts, relationships = get_extended_related_concepts(
                            selected_question, question_concept_df, concept_df, relation_df, kc2idx, kc_list, depth=1
                        )
                        
                        # Store relationships for the graph function to use
                        create_knowledge_graph._relationships = relationships
                        
                        print(f"DEBUG: Found {len(direct_concepts)} direct concepts and {len(extended_concepts)} total concepts")
                        print(f"DEBUG: Found {len(relationships)} relationships")
                        
                        # Use extended concepts instead of just related concepts
                        related_concepts = extended_concepts if extended_concepts else []
                        
                        # Ensure related_concepts is a valid list
                        if related_concepts is None or not isinstance(related_concepts, list):
                            print(f"DEBUG: Invalid related_concepts, using empty list")
                            related_concepts = []
                        elif related_concepts:
                            print(f"DEBUG: Found {len(related_concepts)} concepts in extended network for question {selected_question}")
                            for rc in related_concepts:
                                concept_name = get_concept_name(rc, concept_df, kc_list)
                                print(f"DEBUG: Extended concept {rc}: {concept_name}")
                        
                        # Add this after: related_concepts = extended_concepts if extended_concepts else []
                        print(f"DEBUG: Extended concepts being passed to graph: {related_concepts}")
                        print(f"DEBUG: Extended concept names: {[get_concept_name(rc, concept_df, kc_list) for rc in related_concepts if rc >= 0]}")
                    except Exception as e:
                        print(f"DEBUG: Error getting extended related concepts: {e}")
                        related_concepts = []  # Use empty list as fallback
                        create_knowledge_graph._relationships = []
                    
                    # Option to show only related concepts or all
                    show_all = st.checkbox("Show all concepts", value=False, key="show_all_concepts_main")
                    
                    # Add a debug message
                    print(f"DEBUG: Show all concepts: {show_all}")
                    print(f"DEBUG: Number of related concepts: {len(related_concepts)}")
                    
                    # Create knowledge graph
                    G = create_knowledge_graph(
                        memory_to_use,
                        concept_df,
                        kc2idx,
                        kc_list,
                        None,  # Don't include prerequisite relationships
                        related_concepts=related_concepts if not show_all else None,
                        include_connected=True,
                        show_all=show_all
                    )
                    
                    # Debug info about the graph
                    print(f"DEBUG: Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                    if G.number_of_nodes() > 0:
                        print("DEBUG: Node indices:", list(G.nodes()))
                        print("DEBUG: Node names:", [G.nodes[n].get('name', '') for n in G.nodes()])
                        
                    # Add visualization controls
                    viz_col1, viz_col2 = st.columns([4, 1])
                    
                    with viz_col2:
                        # Display custom legend
                        st.markdown(create_mastery_legend(), unsafe_allow_html=True)
                        
                        # Add optional controls
                        st.write("Visualization Options")
                        show_labels = st.checkbox("Show concept names", value=True, key="show_concept_labels_main_1")
                        if show_labels:
                            st.info("You can hover over nodes to see detailed concept information")
                    
                    with viz_col1:
                        # Plot interactive graph
                        try:
                            # Get the plot 
                            fig = plot_interactive_knowledge_graph(G)
                            
                            # Apply visualization options - now always show labels for directly related concepts,
                            # and conditionally for others based on checkbox
                            # Only process if fig.data exists and is not empty
                            if hasattr(fig, 'data') and fig.data is not None and len(fig.data) > 0:
                                # Loop through traces safely
                                for i in range(len(fig.data)):
                                    # Get the trace
                                    trace = fig.data[i]
                                    
                                    # Only modify if it has a mode attribute
                                    if hasattr(trace, 'mode') and trace.mode is not None:
                                        # Check if trace name exists and contains "Liên quan"
                                        if hasattr(trace, 'name') and trace.name is not None and 'Liên quan' in trace.name:
                                            # Directly related concepts always show text
                                            trace.mode = 'markers+text'
                                        else:
                                            # Others depend on checkbox
                                            if show_labels:
                                                trace.mode = 'markers+text'
                                            else:
                                                trace.mode = 'markers'
                        
                            # Display the plot
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error plotting knowledge graph: {e}")
                            st.write("Attempting to display a simple version...")
                            
                            # Fallback: Display a simple table of concepts
                            fallback_data = []
                            
                            # Try to get a snapshot directly
                            memory_to_use = st.session_state.student_memory
                            if 'model' in st.session_state and 'current_student_id' in st.session_state:
                                try:
                                    student_id = st.session_state.current_student_id
                                    snapshot = st.session_state.model.get_snapshot(student_id)
                                    if snapshot is not None:
                                        print(f"DEBUG: Using snapshot for fallback table")
                                        memory_to_use = snapshot
                                except Exception as e:
                                    print(f"DEBUG: Error getting snapshot for fallback table: {e}")
                            
                            # Make sure memory_to_use is valid
                            if memory_to_use is None or len(memory_to_use) == 0:
                                st.warning("No memory data available for this student.")
                                return
                                
                            # Calculate norms
                            try:
                                memory_norms = [tensor.norm().item() for tensor in memory_to_use]
                                
                                for idx, norm in enumerate(memory_norms[:10]):  # Show first 10
                                    if idx < len(kc_list):
                                        concept_name = get_concept_name(idx, concept_df, kc_list)
                                        fallback_data.append({
                                            "Concept": concept_name,
                                            "‖Mv‖": f"{norm:.4f}"
                                        })
                                
                                if fallback_data:
                                    st.table(pd.DataFrame(fallback_data))
                                else:
                                    st.warning("No concept data available to display.")
                            except Exception as e:
                                st.error(f"Error generating fallback table: {e}")
                                st.warning("Could not display concept data.")
                    
                    # Display mastery levels - keep only one instance
                    with st.expander("Detailed Mastery Levels"):
                        try:
                            # Get concepts related to the question
                            try:
                                related_concepts, _ = get_related_concepts(selected_question, question_concept_df, concept_df, kc2idx, kc_list)
                                # Ensure related_concepts is a valid list
                                if related_concepts is None or not isinstance(related_concepts, list):
                                    print(f"DEBUG: Invalid related_concepts for mastery levels, using empty list")
                                    related_concepts = []
                                elif related_concepts:
                                    print(f"DEBUG: Found {len(related_concepts)} concepts for mastery table")
                                    for rc in related_concepts:
                                        concept_name = get_concept_name(rc, concept_df, kc_list)
                                        print(f"DEBUG: Including concept {rc}: {concept_name}")
                            except Exception as e:
                                print(f"DEBUG: Error getting related concepts for mastery levels: {e}")
                                related_concepts = []  # Use empty list as fallback
                            
                            # Show mastery levels for directly related concepts only
                            mastery_df = display_mastery_levels(st.session_state.student_memory, kc_list, concept_df, related_concepts)
                            
                            if mastery_df.empty:
                                st.warning("No mastery data available for concepts related to this question.")
                            else:
                                # Add color coding to dataframe
                                def color_status(val):
                                    if val == "Strong":
                                        return 'background-color: rgba(50, 200, 50, 0.2)'
                                    elif val == "Medium":
                                        return 'background-color: rgba(255, 150, 50, 0.2)'
                                    else:
                                        return 'background-color: rgba(255, 50, 50, 0.2)'
                                
                                # Apply styling
                                styled_df = mastery_df.style.applymap(color_status, subset=['Status'])
                                
                                # Debug - print the DataFrame columns and first few rows
                                print(f"DEBUG: DataFrame columns: {mastery_df.columns.tolist()}")
                                print(f"DEBUG: First few rows:\n{mastery_df.head()}")
                                
                                # Display the dataframe with a title showing the number of concepts
                                st.write(f"Showing mastery levels for {len(mastery_df)} concepts related to Question {selected_question}:")
                                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                                
                                # Option to download as CSV
                                csv = mastery_df.to_csv(index=False)
                                st.download_button(
                                    "Download Mastery Data", 
                                    csv, 
                                    f"student_{selected_student}_mastery.csv",
                                    "text/csv", 
                                    key='download-csv'
                                )
                        except Exception as e:
                            st.error(f"Error displaying mastery levels: {e}")
                except Exception as e:
                    st.error(f"Error creating knowledge graph: {e}")
                    return
                
                # Remove the second instance of mastery levels section
            else:
                st.info("Process student history to visualize the knowledge state")
                
                # Add sample image with fixed parameter
                st.write("Example of knowledge concept map:")
                st.image("https://i.imgur.com/6P2RJK2.png", use_container_width=True, caption="Example Knowledge Concept Map")
        
    with tab2:
        st.header("Learning Path Recommendation")
        
        # Check if model is loaded
        if 'model' not in st.session_state or st.session_state.model is None:
            st.warning("Please load a model first using the options in the sidebar.")
            return
        
        # Time period selection
        time_period = st.radio(
            "Select time period for learning path",
            ["7 days", "15 days", "1 month"],
            horizontal=True,
            key="time_period"
        )
        
        if st.session_state.get('student_memory') is not None:
            # Try to get a snapshot directly
            memory_to_use = st.session_state.student_memory
            if 'model' in st.session_state and 'current_student_id' in st.session_state:
                try:
                    student_id = st.session_state.current_student_id
                    snapshot = st.session_state.model.get_snapshot(student_id)
                    if snapshot is not None:
                        print(f"DEBUG: Using snapshot for learning path")
                        memory_to_use = snapshot
                except Exception as e:
                    print(f"DEBUG: Error getting snapshot for learning path: {e}")
            
            # Get weak concepts
            memory_norms = [tensor.norm().item() for tensor in memory_to_use]
            
            weak_concepts = []
            for idx, norm in enumerate(memory_norms):
                if idx >= len(kc_list):
                    continue
                    
                # Use the actual mv_norm value directly
                mv_norm = norm
                
                # Consider concepts with low mastery as weak
                if mv_norm < 1.0:
                    weak_concepts.append(idx)
            
            # Create two columns for layout: 1:3 ratio
            col_weak, col_path = st.columns([1, 3])
            
            # Display weak concepts in left column
            with col_weak:
                st.subheader("Weak Concepts")
                
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
                </style>
                """, unsafe_allow_html=True)
                
                if weak_concepts:
                    # Display weak concepts in a compact format
                    for idx in weak_concepts:
                        if idx < len(kc_list):
                            concept_name = get_concept_name(idx, concept_df, kc_list)
                            st.markdown(f"""
                            <div class="concept-card">
                                <strong>C<sub>{idx}</sub></strong> {concept_name}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show count
                    st.caption(f"Total: {len(weak_concepts)} concepts need improvement")
                else:
                    st.success("No weak concepts identified!")
                    st.caption("‖Mv‖ values are all above 1.0")
            
            # Display learning path in right column
            with col_path:
                st.subheader("Personalized Learning Path")
                
                # Automatically generate learning path
                try:
                    # Add a refresh button
                    col_refresh, col_status = st.columns([1, 3])
                    with col_refresh:
                        refresh = st.button("Refresh Learning Path")
                    
                    with col_status:
                        if refresh:
                            st.info("Generating new learning path...")
                    
                    with st.spinner("Generating personalized learning path..."):
                        # Pass student_id for caching
                        student_id = st.session_state.current_student_id if 'current_student_id' in st.session_state else None
                        
                        learning_path = generate_learning_path(
                            weak_concepts, 
                            concept_df, 
                            kc_list, 
                            time_period,
                            student_id=student_id,
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
                    if weak_concepts:
                        st.markdown("## General Study Recommendations")
                        st.write("Based on the identified weak concepts, here are some general recommendations:")
                        
                        recommendations = []
                        for idx in weak_concepts[:5]:  # Limit to first 5 concepts
                            if idx < len(kc_list):
                                concept_name = get_concept_name(idx, concept_df, kc_list)
                                recommendations.append(f"• Study more about **{concept_name}** - focus on foundational principles and practical examples")
                        
                        if len(weak_concepts) > 5:
                            recommendations.append(f"• ...and {len(weak_concepts) - 5} more concepts")
                        
                        st.markdown("\n".join(recommendations))
        else:
            st.info("First process student history in the Knowledge Testing tab to analyze the knowledge state")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
