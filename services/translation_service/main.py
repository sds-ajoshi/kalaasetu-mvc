from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

app = FastAPI(title="Indic Trans Translation Service")

# Supported languages subset (at least 12)
SUPPORTED_LANGUAGES = {
    "hin_Deva":"Hindi", "mar_Deva":"Marathi", "tam_Taml":"Tamil",
    "tel_Telu":"Telugu", "ben_Beng":"Bengali", "guj_Gujr":"Gujarati",
    "kan_Knda":"Kannada", "mal_Mlym":"Malayalam", "pan_Guru":"Punjabi",
    "ory_Orya":"Odia", "asm_Beng":"Assamese", "urd_Arab":"Urdu"
}

# Initialize models globally
models = {}

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@app.on_event("startup")
async def startup_event():
    logging.info("Loading IndicTrans2 models...")
    models["indic_en"] = load_model("ai4bharat/indictrans2-indic-en-1B")
    models["en_indic"] = load_model("ai4bharat/indictrans2-en-indic-1B")
    logging.info("IndicTrans2 models loaded.")

class TranslationRequest(BaseModel):
    text: str
    src_lang: str  # e.g. "hin_Deva" or "eng_Latn"
    tgt_lang: str  # e.g. "eng_Latn" or one supported Indic code

class TranslationResponse(BaseModel):
    translated_text: str
    src_lang: str
    tgt_lang: str

@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text is required")
    if req.src_lang == req.tgt_lang:
        return TranslationResponse(translated_text=req.text, src_lang=req.src_lang, tgt_lang=req.tgt_lang)
    model_key = None
    if req.src_lang in SUPPORTED_LANGUAGES and req.tgt_lang == "eng_Latn":
        model_key = "indic_en"
    elif req.src_lang == "eng_Latn" and req.tgt_lang in SUPPORTED_LANGUAGES:
        model_key = "en_indic"
    else:
        raise HTTPException(status_code=400, detail="Language pair not supported")

    tokenizer, model = models[model_key]
    inputs = tokenizer(req.text, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, max_length=512, num_beams=5)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translation = decoded[0]
    return TranslationResponse(translated_text=translation, src_lang=req.src_lang, tgt_lang=req.tgt_lang)