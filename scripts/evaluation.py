import numpy as np
from tqdm.auto import tqdm
def recall_at_k_single(ranked, relevant, k=10):
 relevant=set(relevant); return (sum(x in relevant for x in ranked[:k])/len(relevant)) if relevant else None
def evaluate_recall(predictions,qrels,k=10,show_progress=True):
 vals=[]; it=predictions.items(); it=tqdm(it,desc=f"Recall@{k}",leave=False) if show_progress else it
 for qid,ranked in it:
  s=recall_at_k_single(ranked,qrels.get(qid,[]),k);
  if s is not None: vals.append(s)
 return {f"Recall@{k}":round(float(np.mean(vals)),4) if vals else 0.0,"n_queries":len(predictions)}
