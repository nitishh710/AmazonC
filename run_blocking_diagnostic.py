import pandas as pd
from collections import defaultdict
import blocking

print("Loading data...")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t").head(500)
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

# Map ground truth
gt_map = {}
for _, row in gt.iterrows():
    matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip() != '' else []
    gt_map[row['source1_entity_id']] = set(m.strip() for m in matches if m.strip())

print(f"Running blocking on sample of {len(s1)} Source 1 entities...")

# Run candidate generation (inspect blocking function call)
candidates_by_s1 = defaultdict(list)

# Generate candidates using blocking.py interface
if hasattr(blocking, 'generate_candidates'):
    candidates = blocking.generate_candidates(s1, s2, s3)
    for s1_id, cand_id in candidates:
        candidates_by_s1[s1_id].append(cand_id)
elif hasattr(blocking, 'get_candidates'):
    candidates_by_s1 = blocking.get_candidates(s1, s2, s3)
else:
    print("Check blocking.py for the exact function name (e.g. generate_candidates or get_candidates).")
    exit()

# Evaluate candidate recall across different caps
caps = [20, 50, 100, 200, 500, 1000, 2000]
print("\n--- Diagnostic Results ---")
for cap in caps:
    total_true = 0
    captured_true = 0
    candidate_counts = []
    
    for s1_id in s1['entity_id']:
        true_matches = gt_map.get(s1_id, set())
        total_true += len(true_matches)
        
        cands = candidates_by_s1.get(s1_id, [])[:cap]
        candidate_counts.append(len(cands))
        captured_true += len(true_matches.intersection(set(cands)))
    
    recall = (captured_true / total_true) if total_true > 0 else 0
    avg_cands = sum(candidate_counts) / len(candidate_counts)
    print(f"Cap: {cap:4d} | Recall: {recall * 100:6.2f}% | Captured: {captured_true:4d}/{total_true:4d} | Avg Candidates: {avg_cands:.1f}")

