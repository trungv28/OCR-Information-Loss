import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from src.data import load
from src.el.model import BEAM, MAX_NEW_TOKENS, load_linker, mark_entity, title_to_qid, decoded_title

SPLIT = 'test' if '--test' in sys.argv else 'dev'
BATCH_SIZE = 32


def predict_batch(model, tokenizer, texts: list[str], title2qid: dict, langs: list[str], device) -> list[str | None]:
    enc = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            num_beams=BEAM,
            num_return_sequences=BEAM,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    # num_return_sequences outputs per input, in order
    qids = []
    for i, lang in enumerate(langs):
        candidates = decoded[i * BEAM: (i + 1) * BEAM]
        qid = None
        for text in candidates:
            qid = title_to_qid(title2qid, decoded_title(text), lang)
            if qid:
                break
        qids.append(qid)
    return qids


if __name__ == '__main__':
    title2qid, tokenizer, model, device = load_linker()

    consensus = {
        json.loads(line)['entity_id']: json.loads(line)
        for line in open(f'data/annotations/consensus_el_{SPLIT}.jsonl')
    }
    accepted_ids = {eid for eid, r in consensus.items() if r['status'] == 'accepted'}

    proj_by_doc = {}
    for line in open(f'data/annotations/projected_{SPLIT}.jsonl'):
        d = json.loads(line)
        proj_by_doc[d['document_id']] = d
    text_by_id = {
        rec['document_metadata']['document_id']: rec
        for rec in load(SPLIT)
    }

    entities = []
    for doc_id, doc in proj_by_doc.items():
        rec = text_by_id.get(doc_id)
        if not rec:
            continue
        gt_full = rec['ground_truth']['transcription_unit']
        ocr_full = rec['ocr_hypothesis']['transcription_unit']
        lang = doc['language']
        for ent in doc['entities']:
            eid = ent['entity_id']
            if eid not in accepted_ids or ent['status'] != 'projected' or ent['ocr_start'] is None:
                continue
            entities.append({
                'entity_id': eid,
                'lang': lang,
                'gt_input': mark_entity(gt_full, ent['gt_start'], ent['gt_end']),
                'ocr_input': mark_entity(ocr_full, ent['ocr_start'], ent['ocr_end']),
            })

    print(f'{len(entities)} entities')

    for mode in ('gt', 'ocr'):
        out_path = Path(f'data/annotations/nel_{mode}_{SPLIT}.jsonl')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        done = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    done.add(json.loads(line)['entity_id'])

        batch_ents, batch_texts, batch_langs = [], [], []

        def flush(f):
            if not batch_ents:
                return
            qids = predict_batch(model, tokenizer, batch_texts, title2qid, batch_langs, device)
            for ent, qid in zip(batch_ents, qids):
                f.write(json.dumps({'entity_id': ent['entity_id'], 'qid': qid}) + '\n')
            f.flush()
            batch_ents.clear(); batch_texts.clear(); batch_langs.clear()

        with open(out_path, 'a') as f:
            for ent in tqdm(entities, desc=f'nel-{mode}'):
                if ent['entity_id'] in done:
                    continue
                batch_ents.append(ent)
                batch_texts.append(ent[f'{mode}_input'])
                batch_langs.append(ent['lang'])
                if len(batch_ents) >= BATCH_SIZE:
                    try:
                        flush(f)
                    except Exception as e:
                        tqdm.write(f'[warn] batch failed: {e}')
                        batch_ents.clear(); batch_texts.clear(); batch_langs.clear()
            try:
                flush(f)
            except Exception as e:
                tqdm.write(f'[warn] final batch failed: {e}')

        print(f'Saved to {out_path}')
