def get_item_text(item):
 return " ".join(str(item[k]) for k in ("title","name","description","text") if item.get(k))
def get_query_text(instance):
 return " ".join(str(r["text"]) for r in instance["reviews"] if r.get("text"))
