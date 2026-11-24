import torch
from sentence_transformers import SentenceTransformer
from config import DEVICE,BASE_MODEL_NAME,EMBEDDING_BATCH_SIZE,RETRIEVAL_CHUNK_SIZE
from text_utils import get_item_text,get_query_text
class DenseRetriever:
 def __init__(self,model,catalog,catalog_ids,batch_size=EMBEDDING_BATCH_SIZE,chunk_size=RETRIEVAL_CHUNK_SIZE):
  self.model=model; self.ids=list(catalog_ids); self.batch_size=batch_size; self.chunk_size=chunk_size
  self.emb=model.encode([get_item_text(x) for x in catalog],batch_size=batch_size,show_progress_bar=True,convert_to_tensor=True,normalize_embeddings=True)
 @classmethod
 def from_pretrained(cls,catalog,catalog_ids,model_name=BASE_MODEL_NAME,max_seq_length=512,**kw):
  m=SentenceTransformer(model_name,device=DEVICE); m.max_seq_length=max_seq_length; return cls(m,catalog,catalog_ids,**kw)
 def retrieve(self,queries,top_k=10):
  qe=self.model.encode([get_query_text(q) for q in queries],batch_size=self.batch_size,show_progress_bar=True,convert_to_tensor=True,normalize_embeddings=True); out={}
  for s in range(0,len(queries),self.chunk_size):
   e=min(s+self.chunk_size,len(queries)); ix=torch.topk(torch.matmul(qe[s:e],self.emb.T),k=top_k,dim=1).indices.cpu().numpy()
   for off,row in enumerate(ix): out[queries[s+off]["query_id"]]=[self.ids[i] for i in row]
  return out
