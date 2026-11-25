from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import InputExample,SentenceTransformer,losses
from .config import DEVICE,BASE_MODEL_NAME,FT_MAX_SEQ_LENGTH,BATCH_SIZE,EPOCHS,MODEL_DIR
from .text_utils import get_item_text,get_query_text
def build_training_examples(train_split,bm25_preds,catalog):
 lookup={x["item_id"]:get_item_text(x) for x in catalog}; ex=[]
 for q in train_split:
  pos=q["item_id"]
  if pos not in lookup: continue
  neg=next((p for p in bm25_preds.get(q["query_id"],[]) if p!=pos),None); texts=[get_query_text(q),lookup[pos]]
  if neg in lookup: texts.append(lookup[neg])
  ex.append(InputExample(texts=texts))
 return ex
def fine_tune_model(examples,model_name=BASE_MODEL_NAME,epochs=EPOCHS,batch_size=BATCH_SIZE,max_seq_length=FT_MAX_SEQ_LENGTH):
 m=SentenceTransformer(model_name,device=DEVICE); m.max_seq_length=max_seq_length; dl=DataLoader(examples,shuffle=True,batch_size=batch_size); loss=losses.MultipleNegativesRankingLoss(m)
 m.fit(train_objectives=[(dl,loss)],epochs=epochs,warmup_steps=int(len(dl)*epochs*.1),use_amp=DEVICE.type=="cuda",show_progress_bar=True); return m
def save_model(model,path=None):
 p=Path(path or MODEL_DIR/"fine_tuned"); p.mkdir(parents=True,exist_ok=True); model.save(str(p)); return p
