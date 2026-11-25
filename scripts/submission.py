import csv
from pathlib import Path
from config import TOP_K,SUBMISSION_PATH
def write_submission(test_queries,predictions,catalog_id_set,path=SUBMISSION_PATH,top_k=TOP_K):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); cols=["query_id"]+[f"item_id_{i}" for i in range(1,top_k+1)]
 with path.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
  for q in test_queries:
   ids=predictions[q["query_id"]][:top_k]; w.writerow({"query_id":q["query_id"],**{f"item_id_{i}":p for i,p in enumerate(ids,1)}})
 with path.open(newline="",encoding="utf-8") as f: r=csv.DictReader(f); rows=list(r); assert r.fieldnames==cols
 assert len(rows)==len(test_queries) and [x["query_id"] for x in rows]==[q["query_id"] for q in test_queries]
 for row in rows:
  ids=[row[f"item_id_{i}"] for i in range(1,top_k+1)]; assert len(set(ids))==top_k and all(x in catalog_id_set for x in ids)
 return path
