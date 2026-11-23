import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import PARAMS
from src.data import load, shard_arg
from src.ner.lime import ner_score_from_preds
from src.ner.tagger import run, predict_batch, load_model, check_recovered
from src.spans import apply_corruptions, track_entity, apply_repairs, pop_auc

SPLIT = 'test'
MAX_J = PARAMS['repair']['max_j']


def trajectory(tagger, ocr_win, gt_win, spans, ranked, ent_s, ent_e, label):
    m = len(spans)
    traj = []
    for k in range(m + 1):
        r = np.zeros(m)
        for i in ranked[:k]:
            r[i] = 1.0
        text, ns, ne = apply_repairs(ocr_win, gt_win, spans, r, ent_s, ent_e)
        traj.append(float(check_recovered(tagger, text, ns, ne, label)))
    return traj


if __name__ == '__main__':
    print('device', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)
    tagger = load_model()

    units = {}
    with open('data/results/outcomes_' + SPLIT + '.jsonl') as f:
        for l in f:
            r = json.loads(l)
            if r['in_eval']:
                units[r['entity_id']] = r
    attribution = {r['entity_id']: r for r in
                   (json.loads(l) for l in open('data/results/attribution_' + SPLIT + '.jsonl'))}
    text_by_id = {rec['document_metadata']['document_id']: rec for rec in load(SPLIT)}

    shard = shard_arg()
    suffix = '.shard' + str(shard[0]) + 'of' + str(shard[1]) if shard else ''
    out_path = Path('data/results/occlusion_ner_' + SPLIT + suffix + '.jsonl')

    done = set()
    for p in (out_path, Path('data/results/occlusion_ner_' + SPLIT + '.jsonl')):
        if p.exists():
            done |= {json.loads(l)['entity_id'] for l in open(p)}

    # longest first then stride, so every shard gets a mix of big and small units
    todo = sorted([(e, r) for e, r in units.items() if e in attribution],
                  key=lambda x: -len(attribution[x[0]]['edit_spans']))
    if shard:
        todo = todo[shard[0]::shard[1]]
    todo = [(e, r) for e, r in todo if e not in done]
    print(len(units), 'clean-ceiling units |', len(todo), 'to do', flush=True)

    with open(out_path, 'a') as fout:
        for eid, ent in tqdm(todo, desc='occlusion-ner'):
            attr = attribution[eid]
            spans = attr.get('edit_spans', [])
            m = len(spans)
            rec = text_by_id.get(ent['document_id'])
            if m == 0 or rec is None:
                continue

            gt_text = rec['ground_truth']['transcription_unit']
            ocr_text = rec['ocr_hypothesis']['transcription_unit']
            gt_win = gt_text[attr['gt_win_start']:attr['gt_win_end']]
            ocr_win = ocr_text[attr['ocr_win_start']:attr['ocr_win_end']]
            g_s, g_e = attr['ent_win_start'], attr['ent_win_end']
            label = ent['label']

            # gain of span s = score on gt minus score with only s corrupted
            texts, coords = [], []
            for i in range(m):
                z = np.zeros(m)
                z[i] = 1.0
                texts.append(apply_corruptions(gt_win, ocr_win, spans, z))
                coords.append(track_entity(ocr_win, spans, z, g_s, g_e))
            base_gt = ner_score_from_preds(run(tagger, gt_win, 0), g_s, g_e, label)
            preds = predict_batch(tagger, texts)
            raw = [base_gt - ner_score_from_preds(p, ns, ne, label) for p, (ns, ne) in zip(preds, coords)]
            gains = [round(float(g), 6) for g in raw]

            row = {
                'entity_id': eid,
                'document_id': ent['document_id'],
                'n_edit_spans': m,
                'score_gt': round(float(base_gt), 6),
                'gains': gains,
                'occ_damage': round(float(sum(max(0.0, g) for g in raw)), 6),
                'ner_failed': int(ent['outcome'] != 'correct'),
            }

            # repair order by max(0, gain), clipped like lime, replayed from ocr
            if ent['outcome'] != 'correct':
                o_s = attr.get('ent_ocr_win_start')
                o_e = attr.get('ent_ocr_win_end')
                if o_s is None:
                    o_s, o_e = g_s, g_e
                ranked = sorted(range(m), key=lambda i: -max(0.0, gains[i]))
                traj = trajectory(tagger, ocr_win, gt_win, spans, ranked, o_s, o_e, label)
                row['traj_occlusion_gt'] = [round(v, 4) for v in traj]
                row['auc_occlusion_gt'] = round(pop_auc(traj, MAX_J), 4)

            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            fout.flush()

    print('saved', out_path)
