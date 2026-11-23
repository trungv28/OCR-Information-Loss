import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from src.data import load
from src.ner.tagger import MODEL_NAME, MODELS, load_model, run

LANGS = set(MODELS)


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'
    on_ocr = '--ocr' in sys.argv
    preview = '--preview' in sys.argv
    tag = 'ocr' if on_ocr else 'gt'
    text_key = 'ocr_hypothesis' if on_ocr else 'ground_truth'

    print(f'loading {MODEL_NAME}')
    ner = load_model()
    print(f'cuda={torch.cuda.is_available()} | split={split} | text={tag}')

    records = [
        rec for rec in load(split)
        if rec['ground_truth']['transcription_unit'].strip()
        and rec['document_metadata']['language'] in LANGS
    ]

    if preview:
        for rec in [r for r in records if r['document_metadata']['language'] == 'en'][:5]:
            meta = rec['document_metadata']
            text = rec[text_key]['transcription_unit']
            print(f"\n--- {meta['primary_dataset_name']} ({meta['language']}) | {meta['date']} ---")
            print(f'TEXT: {text[:400]}')
            ents = run(ner, text, 0)
            if not ents:
                print('(no entities)')
            for e in ents[:15]:
                ok = 'OK' if text[e['start']:e['end']] == e['text'] else 'MISMATCH'
                print(f"[{e['label']}] {e['text']!r} ({e['start']}:{e['end']}) {ok}")
        sys.exit(0)

    out_path = Path(f'data/annotations/ner_{tag}_{split}.jsonl')
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
                'entities': run(ner, text, 0),
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
    print(f'Saved to {out_path}')
