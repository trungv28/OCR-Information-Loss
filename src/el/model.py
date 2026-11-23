import pickle

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import PARAMS

MODEL_NAME = PARAMS['models']['el']
TITLE2QID_PATH = 'lang_title2wikidataID-normalized_with_redirect.pkl'
BEAM = PARAMS['el']['beam']
MAX_NEW_TOKENS = PARAMS['el']['max_new_tokens']
START, END = '[START]', '[END]'
MAX_LEN = PARAMS['el']['max_len']
KEEP_BEFORE = PARAMS['el']['keep_before']


def load_linker():
    print('loading title to QID mapping and ' + MODEL_NAME, flush=True)
    with open(TITLE2QID_PATH, 'rb') as f:
        title2qid = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()
    print('model on', device, flush=True)
    return title2qid, tokenizer, model, device


def mark_entity(text: str, start: int, end: int) -> str:
    return text[:start] + START + ' ' + text[start:end] + ' ' + END + text[end:]


def fit_marker(text: str, tokenizer, max_len: int = MAX_LEN, keep_before: int = KEEP_BEFORE) -> str:
    '''cut leading context so the marked entity is not truncated away, no-op if it already fits'''
    i = text.find(START)
    if i < 0:
        return text
    j = text.find(END, i)
    j = j + len(END) if j >= 0 else i + len(START)
    if len(tokenizer(text[:j], add_special_tokens=True)['input_ids']) <= max_len:
        return text  # entity already fully inside the model's window
    # cut as much as possible but keep keep_before tokens before the marker
    lo, hi = 0, i
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer(text[mid:i], add_special_tokens=False)['input_ids']) > keep_before:
            lo = mid
        else:
            hi = mid - 1
    sp = text.rfind(' ', 0, lo)
    return text[sp + 1:] if sp > 0 else text[lo:]


def title_to_qid(title2qid: dict, title: str, lang: str) -> str | None:
    for lg in (lang, 'en'):
        val = title2qid.get((lg, title))
        if val:
            return next(iter(val)) if isinstance(val, set) else val
    return None


def decoded_title(text: str) -> str:
    return text.split('>>')[0].strip() if '>>' in text else text.strip()


def el_score(model, tokenizer, title2qid, device, text, ent_s, ent_e, ref_qid, lang) -> float:
    '''probability of the reference qid among the beams, 0 when no beam reaches it'''
    marked = fit_marker(mark_entity(text, ent_s, ent_e), tokenizer)
    enc = tokenizer([marked], return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**enc, num_beams=BEAM, num_return_sequences=BEAM,
                             max_new_tokens=MAX_NEW_TOKENS, output_scores=True,
                             return_dict_in_generate=True)
    decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    scores = out.sequences_scores.cpu().tolist()
    best = None
    for k in range(BEAM):
        if title_to_qid(title2qid, decoded_title(decoded[k]), lang) == ref_qid:
            if best is None or scores[k] > best:
                best = scores[k]
    return float(np.exp(best)) if best is not None else 0.0


def check_recovered(model, tokenizer, title2qid, device, text, ent_s, ent_e, ref_qid, lang) -> bool:
    '''greedy decode links the entity to the reference qid'''
    marked = fit_marker(mark_entity(text, ent_s, ent_e), tokenizer)
    enc = tokenizer([marked], return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**enc, num_beams=1, max_new_tokens=MAX_NEW_TOKENS)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return title_to_qid(title2qid, decoded_title(decoded), lang) == ref_qid
