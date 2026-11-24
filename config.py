from pathlib import Path
import random, numpy as np, torch
SEED=42; DATA_DIR=Path("."); OUTPUT_DIR=Path("outputs"); MODEL_DIR=OUTPUT_DIR/"models"
SUBMISSION_PATH=Path("2022320319_final.csv"); TOP_K=10; VAL_SIZE=2000
BASE_MODEL_NAME="sentence-transformers/msmarco-distilbert-base-v4"
BASE_MAX_SEQ_LENGTH=512; FT_MAX_SEQ_LENGTH=384; EPOCHS=2; BATCH_SIZE=64
EMBEDDING_BATCH_SIZE=256; RETRIEVAL_CHUNK_SIZE=1000
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed=SEED):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
