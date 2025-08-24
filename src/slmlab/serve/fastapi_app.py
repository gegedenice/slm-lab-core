from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()
_tok = AutoTokenizer.from_pretrained("runs/adapter")
_mod = AutoModelForCausalLM.from_pretrained("runs/adapter")
_gen = pipeline("text-generation", model=_mod, tokenizer=_tok, device=-1)

class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/generate")
def generate(q: Query):
    out = _gen(q.prompt, max_new_tokens=q.max_new_tokens)[0]["generated_text"]
    return {"text": out}