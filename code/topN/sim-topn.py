import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import os

# Set device for model execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SentenceTransformer model
model_name = 'sentence-transformers/all-mpnet-base-v2'
sentence_model = SentenceTransformer(model_name).to(device)

# Load main dataset
df = pd.read_csv('cleaned_data_lcsh.csv', encoding='utf-8-sig')

# Add Category column if not present
if 'Category' not in df.columns:
    df['Category'] = df['lcc'].str[0]

# Load previously predicted label count file
predicted_counts = pd.read_csv('/home/jl1609@students.ad.unt.edu/label_count_predictions_linear-new-all-toc.csv')

# Merge predicted label counts into the main dataframe
df = df.merge(predicted_counts, left_index=True, right_index=True, how='left')

# Initialize results list and evaluation metrics
category_results = {}

# Process each LCC category
for category in df['Category'].unique():
    print(f"Processing category: {category}")
    df_category = df[df['Category'] == category]

    # Check if precomputed vectors exist
    lcsh_vectors_file = f'lcsh_vectors_category_{category}.npy'
    lcsh_subjects_file = f'lcsh_subjects_category_{category}.txt'
    group_vectors_file = f'group_vectors_category_{category}.npy'

    if os.path.exists(lcsh_vectors_file) and os.path.exists(lcsh_subjects_file) and os.path.exists(group_vectors_file):
        # Load precomputed vectors
        lcsh_matrix = np.load(lcsh_vectors_file)
        with open(lcsh_subjects_file, 'r', encoding='utf-8-sig') as f:
            lcsh_subjects = [line.strip() for line in f]
        group_vectors = np.load(group_vectors_file)
        print(f"Loaded saved vectors for category {category}.")
    else:
        # Generate unique LCSH subject vectors
        unique_lcsh_subjects = list(set(subject.strip() for subjects in df_category['lcsh_subject_headings'].dropna() for subject in subjects.split(';') if subject.strip()))
        lcsh_vectors = {subject: sentence_model.encode(subject, convert_to_tensor=True).cpu().numpy() for subject in tqdm(unique_lcsh_subjects, desc=f"Calculating LCSH vectors for category {category}")}
        lcsh_subjects, lcsh_matrix = zip(*lcsh_vectors.items())
        lcsh_matrix = np.stack(lcsh_matrix)
        np.save(lcsh_vectors_file, lcsh_matrix)
        with open(lcsh_subjects_file, 'w', encoding='utf-8-sig') as f:
            for subject in lcsh_subjects:
                f.write(subject + '\n')

        # Generate vectors for all texts
        group_texts = [f"{row['title']} {row['abstract']} {row['toc']}" for _, row in tqdm(df_category.iterrows(), total=len(df_category), desc=f"Processing Texts for category {category}")]
        group_vectors = sentence_model.encode(group_texts, convert_to_tensor=True).cpu().numpy()
        np.save(group_vectors_file, group_vectors)

    # Initialize evaluation metrics
    precisions, recalls, f1_scores, results = [], [], [], []

    # Process each document in the current category
    for i in tqdm(range(len(group_vectors)), desc=f"Processing Documents for category {category}"):
        group_vector = group_vectors[i]
        similarities = cosine_similarity([group_vector], lcsh_matrix)

        true_lcsh_subjects = [subject.strip() for subject in str(df_category.iloc[i]['lcsh_subject_headings']).split(';') if subject.strip()]
        if not true_lcsh_subjects:
            print(f"Debug: Skipping document with index {i} due to empty true labels")
            continue

        predicted_label_count = int(df_category.iloc[i]['Predicted_Label_Count']) if not pd.isna(df_category.iloc[i]['Predicted_Label_Count']) else 5
        top_indices = np.argsort(similarities[0])[-predicted_label_count:][::-1]
        top_similar_subjects = [lcsh_subjects[idx] for idx in top_indices]

        all_subjects = list(set(true_lcsh_subjects + top_similar_subjects))
        y_true = [1 if subject in true_lcsh_subjects else 0 for subject in all_subjects]
        y_pred = [1 if subject in top_similar_subjects else 0 for subject in all_subjects]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        true_label_ranks = {true_label: np.where(np.argsort(similarities[0])[::-1] == lcsh_subjects.index(true_label))[0][0] + 1 for true_label in true_lcsh_subjects if true_label in lcsh_subjects}

        results.append({
            'title': df_category.iloc[i]['title'],
            'abstract': df_category.iloc[i]['abstract'],
            'toc': df_category.iloc[i]['toc'],
            'true_lcsh_subjects': true_lcsh_subjects,
            'top_similar_lcsh_subjects': top_similar_subjects,
            'category': df_category.iloc[i]['Category'],
            'true_label_ranks': true_label_ranks
        })

    average_precision = np.mean(precisions)
    average_recall = np.mean(recalls)
    average_f1 = np.mean(f1_scores)

    category_results[category] = {
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1': average_f1,
        'results': results
    }

    result_df = pd.DataFrame(results)
    result_df.to_csv(f'lcsh_new_sim_with_ranks_category_{category}_topn.csv', index=False, encoding='utf-8-sig')

# Print overall evaluation metrics
for category, metrics in category_results.items():
    print(f"Category: {category}")
    print(f"  Average Precision: {metrics['average_precision']}")
    print(f"  Average Recall: {metrics['average_recall']}")
    print(f"  Average F1 Score: {metrics['average_f1']}")
