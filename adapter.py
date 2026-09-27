import pandas as pd
from pathlib import Path
from typing import Sequence, Mapping, Iterable
from dataclasses import asdict, is_dataclass
from matching_features import extract_features
from matching_model import CustomLogisticRegression

class CompetitionMatcher:
    def __init__(self):
        print("--- Initializing & Training Matcher (On-the-fly) ---")
        self.model = CustomLogisticRegression(learning_rate=0.1, epochs=1000)
        self.threshold = 0.30
        self._targets = None  # Cache for test data to survive chunking
        
        # Fast training on the local training set
        s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t').head(1000)
        s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
        s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
        gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
        
        cands_df = pd.concat([s2, s3], ignore_index=True).set_index('entity_id')
        
        gt_map = {}
        for _, row in gt.iterrows():
            matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
            gt_map[row['source1_entity_id']] = set(m.strip() for m in matches if m.strip())

        X, y = [], []
        for _, row1 in s1.iterrows():
            s1_id = row1['entity_id']
            true_matches = gt_map.get(s1_id, set())
            
            # Positive samples
            for m_id in true_matches:
                if m_id in cands_df.index:
                    X.append(extract_features(row1.to_dict(), cands_df.loc[m_id].to_dict()))
                    y.append(1.0)
            
            # Negative samples
            neg_samples = cands_df.sample(n=len(true_matches) + 2).index
            for n_id in neg_samples:
                if n_id not in true_matches:
                    X.append(extract_features(row1.to_dict(), cands_df.loc[n_id].to_dict()))
                    y.append(0.0)
        
        self.model.fit(X, y)
        print("--- Training Complete ---")

    def predict(
        self,
        *,
        source1_rows: Sequence[any],
        source2_path: Path,
        source3_path: Path,
        candidate_pairs: Mapping[str, Sequence[str]],
    ) -> Mapping[str, Iterable[str]]:
        
        # Load and cache targets once 
        if self._targets is None:
            print("Loading test targets into memory...")
            s2_test = pd.read_csv(source2_path, sep='\t', dtype=str).fillna("")
            s3_test = pd.read_csv(source3_path, sep='\t', dtype=str).fillna("")
            self._targets = pd.concat([s2_test, s3_test], ignore_index=True).set_index('entity_id').to_dict('index')

        results = {}
        for row1 in source1_rows:
            # Handle the pipeline's Record format dynamically
            r1_dict = asdict(row1) if is_dataclass(row1) else dict(row1)
            s1_id = r1_dict['entity_id']
            results[s1_id] = []
            
            cands = candidate_pairs.get(s1_id, [])
            for cand_id in cands:
                if cand_id not in self._targets:
                    continue
                    
                r2_dict = self._targets[cand_id]
                features = extract_features(r1_dict, r2_dict)
                proba = self.model.predict_proba([features])[0]
                
                if proba >= self.threshold:
                    results[s1_id].append(cand_id)
        
        return results