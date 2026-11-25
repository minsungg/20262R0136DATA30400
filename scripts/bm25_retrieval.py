import bm25s
from .text_utils import get_item_text,get_query_text
class BM25Retriever:
 def __init__(self,catalog,catalog_ids,k1=1.5,b=0.75):
  self.ids=list(catalog_ids); toks=bm25s.tokenize([get_item_text(x) for x in catalog],stopwords="en"); self.index=bm25s.BM25(k1=k1,b=b); self.index.index(toks)
 def retrieve(self,queries,top_k=10):
  toks=bm25s.tokenize([get_query_text(q) for q in queries],stopwords="en"); res,_=self.index.retrieve(toks,corpus=self.ids,k=top_k); return {q["query_id"]:list(res[i]) for i,q in enumerate(queries)}
