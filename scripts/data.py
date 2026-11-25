import json, random
from pathlib import Path
from .config import DATA_DIR, SEED, VAL_SIZE
def load_json(path):
 with Path(path).open(encoding="utf-8") as f: return json.load(f)
def load_data(data_dir=DATA_DIR):
 d=Path(data_dir); return load_json(d/"catalog.json"),load_json(d/"train_gold.json"),load_json(d/"test_queries.json")
def validate_data(catalog, train_gold, test_queries):
 ids=[x["item_id"] for x in catalog]; qids=[q["query_id"] for q in test_queries]; tids=[q["query_id"] for q in train_gold]
 assert len(ids)==50000 and len(set(ids))==len(ids)
 assert len(test_queries)==19858 and len(set(qids))==len(qids)
 assert len(train_gold)==20000 and len(set(tids))==len(tids)
 assert all(len(q["reviews"])==5 for q in train_gold+test_queries)
 return ids,set(ids),qids
def split_train_validation(data,val_size=VAL_SIZE,seed=SEED):
 idx=list(range(len(data))); random.Random(seed).shuffle(idx)
 return [data[i] for i in idx[val_size:]],[data[i] for i in idx[:val_size]]
