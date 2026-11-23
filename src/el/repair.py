import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import PARAMS
from src.data import load, shard_arg, rho_arg
from src.el.model import load_linker, check_recovered
from src.local_damage import span_repair_scores
from src.spans import apply_repairs, pop_auc, rank_indices

SPLIT = 'test' if '--test' in sys.argv else 'dev'
N_RANDOM_SEEDS = PARAMS['repair']['random_orders']
H_LOCAL = PARAMS['local_damage']['h']
MAX_J = PARAMS['repair']['max_j']
# gt_damage = attr damage, local_damage = local_h
STRATEGIES = ['gt_damage', 'local_damage', 'corruption', 'mention_first', 'random']


def entity_trajectory(linker, gt_win, ocr_win, spans, ranked_idx, ent_s, ent_e, ref_qid, lang, rho) -> list[int]:
    title2qid, tokenizer, model, device = linker
    m = len(spans)
    traj = []
    for k in range(m + 1):
        r = np.zeros(m)
        for i in ranked_idx[:k]:
            r[i] = 1.0
        text, ns, ne = apply_repairs(ocr_win, gt_win, spans, r, ent_s, ent_e, rho)
        traj.append(int(check_recovered(model, tokenizer, title2qid, device, text, ns, ne, ref_qid, lang)))
    return traj


if __name__ == '__main__':
    rho = rho_arg()
    linker = load_linker()

    el_outcomes = {}
    with open('data/results/outcomes_el_' + SPLIT + '.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if r['in_eval'] and r['outcome'] != 'correct':
                el_outcomes[r['entity_id']] = r
    ner_outcomes = {r['entity_id']: r for r in
                    (json.loads(l) for l in open('data/results/outcomes_' + SPLIT + '.jsonl'))}
    attribution = {r['entity_id']: r for r in
                   (json.loads(l) for l in open('data/results/attribution_el_' + SPLIT + '.jsonl'))}
    consensus = {r['entity_id']: r for r in
                 (json.loads(l) for l in open('data/annotations/consensus_el_' + SPLIT + '.jsonl'))}
    text_by_id = {rec['document_metadata']['document_id']: rec for rec in load(SPLIT)}

    shard = shard_arg()
    name = 'repair_el' + ('_rho' + str(rho) if rho else '') + '_' + SPLIT
    suffix = '.shard' + str(shard[0]) + 'of' + str(shard[1]) if shard else ''
    out_path = Path('data/results/' + name + suffix + '.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)['entity_id'])

    # shard the full list before dropping done ones, so a restart keeps each shard's entities
    entities = [(eid, r) for eid, r in el_outcomes.items() if eid in attribution]
    if shard:
        entities = entities[shard[0]::shard[1]]
    entities = [(eid, r) for eid, r in entities if eid not in done]

    with open(out_path, 'a') as f:
        for eid, el_ent in tqdm(entities, desc='repair-el'):
            ref_qid = consensus.get(eid, {}).get('qid')
            if not ref_qid:
                continue
            ner_ent = ner_outcomes.get(eid, {})
            doc_id = ner_ent.get('document_id') or eid.rsplit('_E', 1)[0]
            lang = el_ent.get('language') or ner_ent.get('language', 'en')
            rec = text_by_id.get(doc_id)
            if not rec:
                continue

            ocr_text = rec['ocr_hypothesis']['transcription_unit']
            gt_text = rec['ground_truth']['transcription_unit']
            attr = attribution[eid]
            spans = attr.get('edit_spans', [])
            m = len(spans)
            ocr_win = ocr_text[attr['ocr_win_start']:attr['ocr_win_end']]
            gt_win = gt_text[attr['gt_win_start']:attr['gt_win_end']]

            ent_ocr_s = attr.get('ent_ocr_win_start')
            ent_ocr_e = attr.get('ent_ocr_win_end')
            if ent_ocr_s is None:
                ent_ocr_s = attr['ent_win_start']
                ent_ocr_e = attr['ent_win_end']

            row = {'entity_id': eid, 'document_id': doc_id, 'label': el_ent.get('label'),
                   'outcome': el_ent.get('outcome'), 'n_edit_spans': m}

            if m == 0:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                continue

            gt_damages = np.array([sp.get('damage', 0.0) for sp in spans])
            local_scores = span_repair_scores(gt_win, ocr_win, spans,
                                              attr['ent_win_start'], attr['ent_win_end'], H_LOCAL)

            for strategy in STRATEGIES:
                if strategy == 'random':
                    trajs = []
                    for _ in range(N_RANDOM_SEEDS):
                        idx = rank_indices(spans, 'random')
                        trajs.append(entity_trajectory(linker, gt_win, ocr_win, spans, idx,
                                                       ent_ocr_s, ent_ocr_e, ref_qid, lang, rho))
                    traj = [float(np.mean([t[k] for t in trajs])) for k in range(len(trajs[0]))]
                else:
                    idx = rank_indices(spans, strategy, local=local_scores, damages=gt_damages)
                    traj = [float(v) for v in entity_trajectory(linker, gt_win, ocr_win, spans, idx,
                                                                ent_ocr_s, ent_ocr_e, ref_qid, lang, rho)]
                row['traj_' + strategy] = [round(v, 4) for v in traj]
                row['auc_' + strategy] = round(pop_auc(traj, MAX_J), 4)

            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()

    print('saved', out_path)
