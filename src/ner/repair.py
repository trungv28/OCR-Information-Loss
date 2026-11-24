import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import PARAMS
from src.data import load, shard_arg, rho_arg
from src.local_damage import span_repair_scores
from src.ner.tagger import MODELS, load_model, check_recovered
from src.spans import apply_repairs, pop_auc, rank_indices

N_RANDOM_SEEDS = PARAMS['repair']['random_orders']
H_LOCAL = PARAMS['local_damage']['h']
MAX_J = PARAMS['repair']['max_j']
# gt_damage = attr damage, local_damage = local_h
STRATEGIES = ['gt_damage', 'local_damage', 'corruption', 'mention_first', 'random']


def entity_trajectory(tagger, ocr_win, gt_win, spans, ranked_idx, ent_s, ent_e, label, rho) -> list[int]:
    m = len(spans)
    traj = []
    for k in range(m + 1):
        r = np.zeros(m)
        for i in ranked_idx[:k]:
            r[i] = 1.0
        text, ns, ne = apply_repairs(ocr_win, gt_win, spans, r, ent_s, ent_e, rho)
        traj.append(int(check_recovered(tagger, text, ns, ne, label)))
    return traj


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'
    rho = rho_arg()
    print('device', 'cuda' if torch.cuda.is_available() else 'cpu', 'split', split, 'rho', rho)

    ner = load_model()
    taggers = {lang: ner for lang in MODELS}

    outcomes = {}
    with open('data/results/outcomes_' + split + '.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if r['in_eval'] and r['outcome'] != 'correct':
                outcomes[r['entity_id']] = r

    attribution = {}
    with open('data/results/attribution_' + split + '.jsonl') as f:
        for line in f:
            r = json.loads(line)
            attribution[r['entity_id']] = r

    text_by_id = {rec['document_metadata']['document_id']: rec for rec in load(split)}

    shard = shard_arg()
    name = 'repair' + ('_rho' + str(rho) if rho else '') + '_' + split
    suffix = '.shard' + str(shard[0]) + 'of' + str(shard[1]) if shard else ''
    out_path = Path('data/results/' + name + suffix + '.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)['entity_id'])

    # shard the full list before dropping done ones, so a restart keeps each shard's entities
    entities = list(outcomes.items())
    if shard:
        entities = entities[shard[0]::shard[1]]
    entities = [(eid, r) for eid, r in entities if eid not in done and eid in attribution]

    with open(out_path, 'a') as f:
        for eid, ent in tqdm(entities, desc='repair'):
            doc_id = ent['document_id']
            lang = ent.get('language', 'en')
            if lang not in taggers or doc_id not in text_by_id:
                continue

            rec = text_by_id[doc_id]
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

            tagger = taggers[lang]
            label = ent['label']
            row = {'entity_id': eid, 'document_id': doc_id, 'label': label,
                   'outcome': ent['outcome'], 'n_edit_spans': m}

            if m == 0:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                continue

            # gt_damage, gt-anchored lime attribution
            gt_damages = np.array([sp.get('damage', 0.0) for sp in spans])
            # local_damage, entity-anchored contribution to local_h(p_ent)
            local_scores = span_repair_scores(gt_win, ocr_win, spans,
                                              attr['ent_win_start'], attr['ent_win_end'], H_LOCAL)

            for strategy in STRATEGIES:
                if strategy == 'random':
                    trajs = []
                    rng = random.Random(PARAMS['seed'])
                    for _ in range(N_RANDOM_SEEDS):
                        idx = rank_indices(spans, 'random', rng=rng)
                        trajs.append(entity_trajectory(tagger, ocr_win, gt_win, spans, idx,
                                                       ent_ocr_s, ent_ocr_e, label, rho))
                    traj = [float(np.mean([t[k] for t in trajs])) for k in range(len(trajs[0]))]
                else:
                    idx = rank_indices(spans, strategy, local=local_scores, damages=gt_damages)
                    traj = [float(v) for v in entity_trajectory(tagger, ocr_win, gt_win, spans, idx,
                                                                ent_ocr_s, ent_ocr_e, label, rho)]
                row['traj_' + strategy] = [round(v, 4) for v in traj]
                row['auc_' + strategy] = round(pop_auc(traj, MAX_J), 4)

            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()

    print('saved', out_path)
