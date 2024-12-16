import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SentenceTransformer model
model_name = 'sentence-transformers/all-mpnet-base-v2'
sentence_model = SentenceTransformer(model_name).to(device)

# Read CSV file
df = pd.read_csv('cleaned_data_lcsh.csv', encoding='utf-8-sig')

# Add Category column if not present
if 'Category' not in df.columns:
    df['Category'] = df['lcc'].str[0]

# Initialize result list and metrics
category_results = {}

# Process each LCC category
for category in df['Category'].unique():
    print(f"Processing category: {category}")
    df_category = df[df['Category'] == category]

    # Check if saved vector files exist
    lcsh_vectors_file = f'lcsh_vectors_category_{category}.npy'
    lcsh_subjects_file = f'lcsh_subjects_category_{category}.txt'
    group_vectors_file = f'group_vectors_category_{category}.npy'

    if os.path.exists(lcsh_vectors_file) and os.path.exists(lcsh_subjects_file) and os.path.exists(group_vectors_file):
        # Load saved vectors
        lcsh_matrix = np.load(lcsh_vectors_file)
        with open(lcsh_subjects_file, 'r', encoding='utf-8') as f:
            lcsh_subjects = [line.strip() for line in f]
        group_vectors = np.load(group_vectors_file)
        print(f"Loaded saved vectors for category {category}.")
    else:
        # Extract unique LCSH subject headings and convert to vectors
        unique_lcsh_subjects = list(set(subject.strip() for subjects in df_category['lcsh_subject_headings'].dropna() for subject in subjects.split(';') if subject.strip()))

        # Compute LCSH subject vectors
        lcsh_vectors = {subject: sentence_model.encode(subject, convert_to_tensor=True).cpu().numpy() for subject in tqdm(unique_lcsh_subjects, desc=f"Calculating LCSH vectors for category {category}")}

        # Convert LCSH vectors to a matrix
        lcsh_subjects, lcsh_matrix = zip(*lcsh_vectors.items())
        lcsh_matrix = np.stack(lcsh_matrix)

        # Save LCSH vectors to files
        np.save(lcsh_vectors_file, lcsh_matrix)
        with open(lcsh_subjects_file, 'w', encoding='utf-8') as f:
            for subject in lcsh_subjects:
                f.write(subject + '\n')

        # Encode group texts into vectors
        group_texts = [str(row['title']) + " " + str(row['abstract']) + " " + str(row['toc']) for _, row in tqdm(df_category.iterrows(), total=len(df_category), desc=f"Processing Texts for category {category}")]
        group_vectors = sentence_model.encode(group_texts, convert_to_tensor=True).cpu().numpy()

        # Save group vectors to file
        np.save(group_vectors_file, group_vectors)

    precisions, recalls, f1_scores, results = [], [], [], []

    # Process each document in the category
    for i in tqdm(range(len(group_vectors)), desc=f"Processing Documents for category {category}"):
        group_vector = group_vectors[i]
        similarities = cosine_similarity([group_vector], lcsh_matrix)

        # Extract true LCSH subjects
        true_lcsh_subjects = [subject.strip() for subject in str(df_category.iloc[i]['lcsh_subject_headings']).split(';') if subject.strip()]
        if not true_lcsh_subjects:
            print(f"Debug: Skipping document with index {i} due to empty true labels")
            continue

        # Get top 50 most similar subjects
        top_indices = np.argsort(similarities[0])[-5:][::-1]
        top_similar_subjects = [lcsh_subjects[idx] for idx in top_indices]

        # Create binary labels for evaluation
        all_subjects = list(set(true_lcsh_subjects + top_similar_subjects))
        y_true = [1 if subject in true_lcsh_subjects else 0 for subject in all_subjects]
        y_pred = [1 if subject in top_similar_subjects else 0 for subject in all_subjects]

        # Calculate precision, recall, and F1 score
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Compute ranks of true labels
        true_label_ranks = {}
        for true_label in true_lcsh_subjects:
            if true_label in lcsh_subjects:
                true_label_index = lcsh_subjects.index(true_label)
                rank = np.where(np.argsort(similarities[0])[::-1] == true_label_index)[0][0] + 1
                true_label_ranks[true_label] = rank

        # Save document results
        results.append({
            'title': df_category.iloc[i]['title'],
            'abstract': df_category.iloc[i]['abstract'],
            'toc': df_category.iloc[i]['toc'],
            'true_lcsh_subjects': true_lcsh_subjects,
            'top_similar_lcsh_subjects': top_similar_subjects,
            'category': df_category.iloc[i]['Category'],
            'true_label_ranks': true_label_ranks
        })

    # Compute average metrics for the category
    average_precision = np.mean(precisions)
    average_recall = np.mean(recalls)
    average_f1 = np.mean(f1_scores)

    # Save metrics and results
    category_results[category] = {
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1': average_f1,
        'results': results
    }

    # Save results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'lcsh_new_sim_with_ranks_category_{category}_top5.csv', index=False, encoding='utf-8-sig')

# Print overall metrics for each category
for category, metrics in category_results.items():
    print(f"Category: {category}")
    print(f"  Average Precision: {metrics['average_precision']}")
    print(f"  Average Recall: {metrics['average_recall']}")
    print(f"  Average F1 Score: {metrics['average_f1']}")
