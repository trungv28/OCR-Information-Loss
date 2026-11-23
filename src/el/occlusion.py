import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import PARAMS
from src.data import load
from src.el.model import load_linker, el_score, check_recovered
from src.spans import apply_corruptions, track_entity, apply_repairs, pop_auc

SPLIT = 'test'
MAX_J = PARAMS['repair']['max_j']


if __name__ == '__main__':
    title2qid, tokenizer, model, device = load_linker()

    el_outcomes = {}
    with open('data/results/outcomes_el_' + SPLIT + '.jsonl') as f:
        for l in f:
            r = json.loads(l)
            if r['in_eval']:
                el_outcomes[r['entity_id']] = r
    ner_outcomes = {r['entity_id']: r for r in
                    (json.loads(l) for l in open('data/results/outcomes_' + SPLIT + '.jsonl'))}
    consensus = {r['entity_id']: r for r in
                 (json.loads(l) for l in open('data/annotations/consensus_el_' + SPLIT + '.jsonl'))}
    attribution = {r['entity_id']: r for r in
                   (json.loads(l) for l in open('data/results/attribution_el_' + SPLIT + '.jsonl'))}
    text_by_id = {rec['document_metadata']['document_id']: rec for rec in load(SPLIT)}

    out_path = Path('data/results/occlusion_el_' + SPLIT + '.jsonl')
    done = set()
    if out_path.exists():
        done = {json.loads(l)['entity_id'] for l in open(out_path)}

    todo = sorted([(e, r) for e, r in el_outcomes.items() if e in attribution],
                  key=lambda x: -len(attribution[x[0]]['edit_spans']))
    todo = [(e, r) for e, r in todo if e not in done]
    print(len(el_outcomes), 'EL clean-ceiling units |', len(todo), 'to do', flush=True)

    with open(out_path, 'a') as fout:
        for eid, el_ent in tqdm(todo, desc='occlusion-el'):
            ref_qid = consensus.get(eid, {}).get('qid')
            if not ref_qid:
                continue
            attr = attribution[eid]
            spans = attr.get('edit_spans', [])
            m = len(spans)
            ner_ent = ner_outcomes.get(eid, {})
            doc_id = ner_ent.get('document_id') or eid.rsplit('_E', 1)[0]
            rec = text_by_id.get(doc_id)
            if m == 0 or rec is None:
                continue
            lang = el_ent.get('language') or ner_ent.get('language', 'en')

            gt_text = rec['ground_truth']['transcription_unit']
            ocr_text = rec['ocr_hypothesis']['transcription_unit']
            gt_win = gt_text[attr['gt_win_start']:attr['gt_win_end']]
            ocr_win = ocr_text[attr['ocr_win_start']:attr['ocr_win_end']]
            g_s, g_e = attr['ent_win_start'], attr['ent_win_end']

            # gain of span s = score on gt minus score with only s corrupted
            base_gt = el_score(model, tokenizer, title2qid, device, gt_win, g_s, g_e, ref_qid, lang)
            raw = []
            for i in range(m):
                z = np.zeros(m)
                z[i] = 1.0
                t = apply_corruptions(gt_win, ocr_win, spans, z)
                ns, ne = track_entity(ocr_win, spans, z, g_s, g_e)
                raw.append(base_gt - el_score(model, tokenizer, title2qid, device, t, ns, ne, ref_qid, lang))
            gains = [round(float(g), 6) for g in raw]

            row = {
                'entity_id': eid,
                'document_id': doc_id,
                'n_edit_spans': m,
                'score_gt': round(float(base_gt), 6),
                'gains': gains,
                'occ_damage': round(float(sum(max(0.0, g) for g in raw)), 6),
                'el_failed': int(el_ent['outcome'] != 'correct'),
            }

            # repair order by max(0, gain), clipped like lime, replayed from ocr
            if el_ent['outcome'] != 'correct':
                o_s = attr.get('ent_ocr_win_start')
                o_e = attr.get('ent_ocr_win_end')
                if o_s is None:
                    o_s, o_e = g_s, g_e
                ranked = sorted(range(m), key=lambda i: -max(0.0, gains[i]))
                traj = []
                for k in range(m + 1):
                    r = np.zeros(m)
                    for i in ranked[:k]:
                        r[i] = 1.0
                    t, ns, ne = apply_repairs(ocr_win, gt_win, spans, r, o_s, o_e)
                    traj.append(float(check_recovered(model, tokenizer, title2qid, device,
                                                      t, ns, ne, ref_qid, lang)))
                row['traj_occlusion_gt'] = [round(v, 4) for v in traj]
                row['auc_occlusion_gt'] = round(pop_auc(traj, MAX_J), 4)

            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            fout.flush()

    print('saved', out_path)
