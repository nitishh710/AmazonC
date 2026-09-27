import pandas as pd
import numpy as np
import random
from matching_features import extract_features
from matching_model import CustomLogisticRegression

def compute_f05(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def main():
    print("Loading datasets for fast validation...")
    s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t').head(100)
    s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
    s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
    gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

    cands_df = pd.concat([s2, s3], ignore_index=True)
    cands_df.set_index('entity_id', inplace=True)

    gt_map = {}
    for _, row in gt.iterrows():
        matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
        gt_map[row['source1_entity_id']] = set(m.strip() for m in matches if m.strip())

    print("Building training pairs...")
    X, y = [], []
    
    # Fast mock candidate generation for validation
    for _, row1 in s1.iterrows():
        s1_id = row1['entity_id']
        true_matches = gt_map.get(s1_id, set())
        
        # Add positive samples
        for m_id in true_matches:
            if m_id in cands_df.index:
                X.append(extract_features(row1.to_dict(), cands_df.loc[m_id].to_dict()))
                y.append(1.0)
                
        # Add random negative samples to train the model
        neg_samples = cands_df.sample(n=len(true_matches) + 5).index
        for n_id in neg_samples:
            if n_id not in true_matches:
                X.append(extract_features(row1.to_dict(), cands_df.loc[n_id].to_dict()))
                y.append(0.0)

    print("Training Custom Logistic Regression...")
    model = CustomLogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X, y)

    print("\n--- Threshold Sweep (F0.5 Optimization) ---")
    probas = model.predict_proba(X)
    
    for thresh in np.arange(0.50, 0.95, 0.05):
        preds = (probas >= thresh).astype(int)
        
        tp = np.sum((preds == 1) & (np.array(y) == 1))
        fp = np.sum((preds == 1) & (np.array(y) == 0))
        fn = np.sum((preds == 0) & (np.array(y) == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = compute_f05(precision, recall)
        
        print(f"Threshold: {thresh:.2f} | P: {precision:.3f} | R: {recall:.3f} | F0.5: {f05:.3f}")

if __name__ == "__main__":
    main()