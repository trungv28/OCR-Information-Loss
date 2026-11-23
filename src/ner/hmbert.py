import json
from pathlib import Path

import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

from src.config import PARAMS
from src.data import load

MODELS = PARAMS['models']['hmbert']

LABEL_MAP = {
    'per': 'pers', 'pers': 'pers',
    'org': 'org',
    'loc': 'loc',
    'prod': 'prod',
}

CHUNK_SIZE = PARAMS['ner']['hmbert_chunk_chars']


def predict(tagger: SequenceTagger, text: str) -> list[dict]:
    if len(text) <= CHUNK_SIZE:
        return run(tagger, text, 0)
    entities, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind('\n', start, end)
            if boundary <= start:
                boundary = text.rfind(' ', start, end)
            if boundary > start:
                end = boundary
        entities.extend(run(tagger, text[start:end], start))
        start = end
    return entities


def run(tagger: SequenceTagger, chunk: str, offset: int) -> list[dict]:
    sentence = Sentence(chunk)
    tagger.predict(sentence)
    results = []
    for span in sentence.get_spans('ner'):
        label = LABEL_MAP.get(span.tag.lower().split('.')[0])
        if label is None:
            continue
        results.append({
            'text': span.text,
            'label': label,
            'start': span.start_position + offset,
            'end': span.end_position + offset,
            'score': round(span.score, 4),
        })
    return results


if __name__ == '__main__':
    import sys
    preview = '--preview' in sys.argv
    split = 'test' if '--test' in sys.argv else 'dev'
    use_ocr = '--ocr' in sys.argv
    text_key = 'ocr_hypothesis' if use_ocr else 'ground_truth'
    tag = 'ocr' if use_ocr else 'gt'

    flair.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using {flair.device} | split={split} | text={tag}')

    taggers = {lang: SequenceTagger.load(model_id) for lang, model_id in MODELS.items()}

    records = [
        rec for rec in load(split)
        if rec['ground_truth']['transcription_unit'].strip()
        and rec['document_metadata']['language'] in MODELS
    ]

    if preview:
        records = [r for r in records if r['document_metadata']['language'] == 'en'][:5]

    if preview:
        for rec in tqdm(records, desc='annotating'):
            meta = rec['document_metadata']
            text = rec[text_key]['transcription_unit']
            entities = predict(taggers[meta['language']], text)
            print(f"\n--- {meta['primary_dataset_name']} ({meta['language']}) | {meta['date']} ---")
            print(f'TEXT: {text}')
            if entities:
                for ent in entities:
                    match = 'OK' if text[ent['start']:ent['end']] == ent['text'] else 'MISMATCH'
                    print(f"[{ent['label']}] {ent['text']!r} ({ent['start']}:{ent['end']}) {match}")
            else:
                print('(no entities)')
    else:
        out_path = Path(f'data/annotations/hmbert_{tag}_{split}.jsonl')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    done.add(json.loads(line)['document_id'])
        records = [r for r in records if r['document_metadata']['document_id'] not in done]
        with open(out_path, 'a') as f:
            for rec in tqdm(records, desc='annotating'):
                meta = rec['document_metadata']
                text = rec[text_key]['transcription_unit']
                row = {
                    'document_id': meta['document_id'],
                    'dataset': meta['primary_dataset_name'],
                    'language': meta['language'],
                    'date': meta['date'],
                    'entities': predict(taggers[meta['language']], text),
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
        print(f'Saved to {out_path}')
