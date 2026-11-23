import json
import sys
from pathlib import Path

from gliner import GLiNER
from tqdm import tqdm

from src.config import PARAMS, PROMPTS
from src.data import load

MODEL = PARAMS['models']['gliner']
LABEL_MAP = PROMPTS['ner']['gliner_labels']
GLINER_LABELS = list(LABEL_MAP.keys())
CHUNK_SIZE = PARAMS['annotation']['chunk_chars']['gliner']


def predict(model: GLiNER, text: str) -> list[dict]:
    if len(text) <= CHUNK_SIZE:
        return model.predict_entities(text, GLINER_LABELS)
    chunks, start, entities = [], 0, []
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind('\n', start, end)
            if boundary <= start:
                boundary = text.rfind(' ', start, end)
            if boundary > start:
                end = boundary
        chunks.append((start, text[start:end]))
        start = end
    for offset, chunk in chunks:
        for ent in model.predict_entities(chunk, GLINER_LABELS):
            ent['start'] += offset
            ent['end'] += offset
            entities.append(ent)
    return entities


if __name__ == '__main__':
    preview = '--preview' in sys.argv

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GLiNER.from_pretrained(MODEL).to(device)
    model.eval()
    print(f'using {device}')

    split = 'test' if '--test' in sys.argv else 'dev'
    records = [
        rec for rec in load(split)
        if rec['ground_truth']['transcription_unit'].strip()
    ]

    if preview:
        records = [r for r in records if r['document_metadata']['language'] == 'en'][:5]

    if preview:
        for rec in tqdm(records, desc='annotating'):
            meta = rec['document_metadata']
            gt = rec['ground_truth']['transcription_unit']
            entities = predict(model, gt)
            print(f"\n--- {meta['primary_dataset_name']} ({meta['language']}) | {meta['date']} ---")
            print(f'TEXT: {gt}')
            if entities:
                for ent in entities:
                    label = LABEL_MAP.get(ent['label'], ent['label'])
                    match = 'OK' if gt[ent['start']:ent['end']] == ent['text'] else 'MISMATCH'
                    print(f"[{label}] {ent['text']!r} ({ent['start']}:{ent['end']}) {match}")
            else:
                print('(no entities)')
    else:
        out_path = Path(f'data/annotations/gliner_multi_{split}.jsonl')
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
                gt = rec['ground_truth']['transcription_unit']
                entities = predict(model, gt)
                row = {
                    'document_id': meta['document_id'],
                    'dataset': meta['primary_dataset_name'],
                    'language': meta['language'],
                    'date': meta['date'],
                    'entities': [
                        {
                            'text': ent['text'],
                            'label': LABEL_MAP.get(ent['label'], ent['label']),
                            'start': ent['start'],
                            'end': ent['end'],
                            'score': round(ent['score'], 4),
                        }
                        for ent in entities
                    ],
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
        print(f'Saved to {out_path}')
