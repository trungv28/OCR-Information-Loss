import json
import sys
from collections import defaultdict
from pathlib import Path

SPLIT = 'test' if '--test' in sys.argv else 'dev'


def load_jsonl_by_id(path: str, key: str = 'entity_id') -> dict:
    out = {}
    with open(path) as f:
        for l in f:
            r = json.loads(l)
            out[r[key]] = r
    return out


if __name__ == '__main__':
    consensus = load_jsonl_by_id(f'data/annotations/consensus_el_{SPLIT}.jsonl')
    nel_gt = load_jsonl_by_id(f'data/annotations/nel_gt_{SPLIT}.jsonl')
    nel_ocr = load_jsonl_by_id(f'data/annotations/nel_ocr_{SPLIT}.jsonl')
    projected = {}
    for l in open(f'data/annotations/projected_{SPLIT}.jsonl'):
        d = json.loads(l)
        for e in d['entities']:
            projected[e['entity_id']] = {'label': e['label'], 'dataset': d['dataset'], 'language': d['language']}

    out_path = Path(f'data/results/outcomes_el_{SPLIT}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    type_eval = defaultdict(int)
    type_correct = defaultdict(int)
    type_wrong = defaultdict(int)
    type_nil = defaultdict(int)

    with open(out_path, 'w') as f:
        for eid, ref in consensus.items():
            if ref['status'] != 'accepted':
                continue

            ref_qid = ref['qid']
            gt_pred = nel_gt.get(eid, {}).get('qid')
            ocr_pred = nel_ocr.get(eid, {}).get('qid')

            in_eval = (gt_pred == ref_qid)

            meta = projected.get(eid, {})
            label = meta.get('label', '?')

            row = {
                'entity_id': eid,
                'label': label,
                'dataset': meta.get('dataset'),
                'language': meta.get('language'),
                'ref_qid': ref_qid,
                'gt_pred': gt_pred,
                'ocr_pred': ocr_pred,
                'in_eval': in_eval,
                'outcome': None,
            }

            if in_eval:
                type_eval[label] += 1
                if ocr_pred == ref_qid:
                    outcome = 'correct'
                    type_correct[label] += 1
                elif ocr_pred is None:
                    outcome = 'nil'
                    type_nil[label] += 1
                else:
                    outcome = 'wrong'
                    type_wrong[label] += 1
                row['outcome'] = outcome

            f.write(json.dumps(row) + '\n')

    LABELS = ['pers', 'loc', 'org', 'prod']
    total_eval = sum(type_eval.values())
    total_correct = sum(type_correct.values())
    total_wrong = sum(type_wrong.values())
    total_nil = sum(type_nil.values())

    print(f'\n=== EL Outcome Metrics -- {SPLIT} ===')
    print(f"\n{'label'} {'eval'} {'correct'} {'wrong'} {'nil'} {'LEF'} {'loss'}")
    for lbl in LABELS:
        n = type_eval[lbl]
        if n == 0:
            continue
        lef = type_correct[lbl] / n
        loss = (type_wrong[lbl] + type_nil[lbl]) / n
        print(f'{lbl} {n} {type_correct[lbl]} {type_wrong[lbl]} {type_nil[lbl]} {round(lef, 3)} {round(loss, 3)}')

    if total_eval:
        lef = total_correct / total_eval
        loss = (total_wrong + total_nil) / total_eval
        print(f'total {total_eval} {total_correct} {total_wrong} {total_nil} {round(lef, 3)} {round(loss, 3)}')

    print(f'\nSaved to {out_path}')
