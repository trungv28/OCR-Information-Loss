import torch
from transformers import AutoTokenizer, pipeline

from src.config import PARAMS
from src.ner.outcomes import classify

MODEL_NAME = PARAMS['models']['ner']
MODELS = {'en': MODEL_NAME, 'fr': MODEL_NAME, 'de': MODEL_NAME}

LABEL_MAP = {
    'pers': 'pers', 'person': 'pers', 'per': 'pers',
    'org': 'org', 'organisation': 'org', 'organization': 'org',
    'loc': 'loc', 'location': 'loc',
    'time': 'time',
    'prod': 'prod', 'product': 'prod',
}

MAX_TOKENS = PARAMS['ner']['max_tokens']
cached = None
tok = None


def load_model():
    '''one cached pipeline shared by all languages'''
    global cached, tok
    if cached is None:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        device = 0 if torch.cuda.is_available() else -1
        cached = pipeline('generic-ner', model=MODEL_NAME, tokenizer=tok,
                           trust_remote_code=True, device=device)
    return cached


def fits(text: str) -> bool:
    return len(tok(text, add_special_tokens=True)['input_ids']) <= MAX_TOKENS


def chunk_end(text: str, start: int) -> int:
    '''largest end where text[start:end] fits MAX_TOKENS, snapped to a word boundary'''
    end = len(text)
    if fits(text[start:end]):
        return end
    lo, hi = start, end
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(text[start:mid]):
            lo = mid
        else:
            hi = mid - 1
    boundary = text.rfind('\n', start, lo)
    if boundary <= start:
        boundary = text.rfind(' ', start, lo)
    return boundary if boundary > start else lo


def resolve_overlaps(ents: list[dict]) -> list[dict]:
    '''this model can tag a phrase and a fragment inside it, so keep whichever is more confident'''
    ordered = sorted(ents, key=lambda e: (-e['score'], -(e['end'] - e['start'])))
    kept = []
    for e in ordered:
        if not any(e['start'] < k['end'] and k['start'] < e['end'] for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e['start'])


def norm(ent: dict, offset: int) -> dict | None:
    '''impresso schema: type / surface / lOffset / rOffset / confidence_ner'''
    label = LABEL_MAP.get((ent.get('type') or '').lower().split('.')[0])
    if label is None:
        return None
    start, end = ent.get('lOffset'), ent.get('rOffset')
    if start is None or end is None:
        return None
    score = float(ent.get('confidence_ner', 100.0)) / 100  # this checkpoint reports confidence * 100
    return {'label': label, 'start': int(start) + offset, 'end': int(end) + offset,
            'score': round(score, 4),
            'text': ent.get('surface') or ''}


def run(ner, text: str, offset: int) -> list[dict]:
    '''predict on one text, chunking by token count to stay under the model limit'''
    if not text.strip():
        return []
    if fits(text):
        out = []
        for e in ner(text):
            n = norm(e, offset)
            if n is not None:
                out.append(n)
        return resolve_overlaps(out)
    entities, start = [], 0
    while start < len(text):
        end = chunk_end(text, start)
        entities.extend(run(ner, text[start:end], offset + start))
        start = end
    return entities


def predict_batch(ner, texts: list[str]) -> list[list[dict]]:
    # the custom pipeline has no batch mode, so this is a loop
    return [run(ner, t, 0) for t in texts]


def check_recovered(ner, text: str, ent_s: int, ent_e: int, label: str) -> bool:
    outcome, _ = classify(run(ner, text, 0), ent_s, ent_e, label)
    return outcome == 'correct'
