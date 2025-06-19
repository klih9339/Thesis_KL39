# DCRKT Model Usage

This repository contains code for using the Disentangled Concept Response Knowledge Tracing (DCRKT) model to predict student performance and analyze knowledge states.

## Requirements

```
torch
torch_geometric
pandas
numpy
matplotlib
scikit-learn
```

## Files

- `test_model.py`: Contains the DCRKT model implementation and utility functions for loading and using the model.
- `use_dcrkt_model.py`: Script for analyzing individual students and making predictions.
- `evaluate_dcrkt.py`: Script for evaluating the model performance on the dataset.

## Usage

### Analyzing a Student

To analyze a student's knowledge state:

```bash
python use_dcrkt_model.py --model dcrkt_model_fold_0.pt --data data/raw/processed_data.csv --action analyze
```

This will:
1. Load the trained model
2. Select a student with a reasonable amount of interactions
3. Process the student's interaction history
4. Generate a visualization of the student's knowledge state
5. Display the top concepts the student has mastered

You can specify a particular student to analyze with the `--student_id` parameter.

### Predicting Performance

To predict a student's performance on new questions:

```bash
python use_dcrkt_model.py --model dcrkt_model_fold_0.pt --data data/raw/processed_data.csv --action predict
```

This will:
1. Load the trained model
2. Select a student
3. Process the student's interaction history
4. Find questions the student hasn't seen that contain concepts they've encountered
5. Predict performance on these new questions
6. Output recommendations from hardest to easiest questions

### Evaluating the Model

To evaluate the model's performance on the entire dataset:

```bash
python evaluate_dcrkt.py --model dcrkt_model_fold_0.pt --data data/raw/processed_data.csv --output results
```

This will:
1. Load the trained model
2. Process all student interactions
3. Calculate performance metrics (AUC, Accuracy, F1, etc.)
4. Generate per-concept performance statistics
5. Create visualizations of concept performance
6. Save results to the specified output directory

You can limit the number of students to evaluate with `--max_students`.

## Model Details

The DCRKT model consists of several components:

1. **Disentangled Response Encoder**: Processes student responses
2. **Knowledge Retriever**: Extracts knowledge from interactions
3. **Memory Updater**: Updates the knowledge state with time decay
4. **Dynamic Concept Graph Builder**: Creates a graph of related concepts
5. **Attention-Based Predictor**: Predicts performance on new questions

The model maintains a separate knowledge state for each student, allowing for personalized predictions. 